#!/usr/bin/env python
"""Interactive labelling server for pose clips.

Run:  $PY label_server.py --out out_pose --port 8000
Then open http://127.0.0.1:8000

Reads clips/recordings/bursts + shard overviews + source npz (for 3D) from
out/index.db (emg_label.store). Per-clip gesture labels, per-clip invalid marks
and whole-recording invalid marks are written to the db `annotations` table
(scope clip|recording, kind label|invalid), carrying a re-segmentation-stable
anchor (clip_start_sample, clip_end_sample, seg_version). /api/export writes
clips_labeled.csv (clips JOIN annotations). /api/drop_recording permanently
deletes a recording (rows + annotations + tombstone + shard). Uses matplotlib +
emg2pose for server-side 3D hand rendering.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import warnings
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers 3d proj)

# Some npz paths don't follow the {subject}/{date-hand}/{stamp} layout, so
# parse_file_info can't recover subject/hand and warns. The labelling server
# handles that gracefully (folder grouping by dirname, skeleton key fallback),
# so the warning is just startup noise here -- silence it.
warnings.filterwarnings("ignore", message="Cannot parse filename")

from emg_label import io_utils, plotting  # noqa: E402
from emg_label import store as dbmod  # noqa: E402  (db layer; handler's `store` = Store instance)

# NOTE: hand rendering is the emg2pose Plotly Mesh3d path (skin_vertices_np ->
# per-frame vertex JSON, drawn client-side). The old matplotlib skeleton path
# (emg_label.skeleton.draw_skeleton) is no longer used here.

FS = 2000
OVERVIEW_REGEN_V = 10         # bump to invalidate regenerated overview_cache
                              # (v3: rad/s; v4: y-axis cap; v5: live rad pose_thr;
                              #  v6: NaN-interp curve + persisted move_exit line;
                              #  v7: y-cap 4x->2x move_enter so holds are legible;
                              #  v8: move_enter/exit recomputed live (units-robust
                              #  vs stale deg pose_thresh in old recordings.csv);
                              #  v9: xlim=[0,n/fs] -> no left/right margin, playhead
                              #  flush; v10: pass hold_*/static_out_* so the blue
                              #  hold span is the REAL hold, not the capped fallback)


def _seg_pose_speed(ja):
    """pose_speed exactly as segment.py computes it for the static overview:
    NaN-interpolate short Manus gaps FIRST (otherwise occlusion drops smear into
    ~250ms speed artifacts that the baked image doesn't have), then the shared
    rad/s pose_speed (joint angles are radians post-load). Keeps the regen +
    frontend curves identical to the static overview so threshold lines align."""
    from emg_label.pose_segmentation import pose_speed
    from emg_label.action_segmentation import interpolate_short_nan_gaps
    return np.asarray(pose_speed(interpolate_short_nan_gaps(ja, FS), FS),
                      dtype=float)
# Whole-recording hand playback is sampled at POSE_FPS frames/sec (env-tunable),
# bounded by POSE_REC_MAX_FRAMES so very long recordings stay cheap. Higher fps
# = smoother motion but proportionally more emg2pose skinning + disk cache per
# recording. 15fps over a ~2min recording is ~1800 frames.
POSE_FPS = float(os.environ.get("POSE_FPS", "15"))
POSE_REC_MAX_FRAMES = int(os.environ.get("POSE_REC_MAX_FRAMES", "3000"))
CURVE_MAX_POINTS = 2000       # downsample target for overview curves
WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")


def _frame_step(n: int) -> int:
    """Sample stride for whole-recording hand playback: aim for POSE_FPS, but
    widen the stride if that would exceed POSE_REC_MAX_FRAMES so the *whole*
    recording is still covered (never truncated to the first N frames)."""
    step = max(1, round(FS / POSE_FPS))
    if (n + step - 1) // step > POSE_REC_MAX_FRAMES:
        step = max(1, -(-n // POSE_REC_MAX_FRAMES))   # ceil(n / cap)
    return step


def _frame_indices(n: int) -> list:
    """Sample indices (one pose per playback frame). Shared by the recording
    payload, the mesh renderer, and the cache-count estimate so frame i always
    maps to the same pose."""
    return list(range(0, n, _frame_step(n))) if n > 0 else []


def _frame_count(n: int) -> int:
    return len(range(0, n, _frame_step(n))) if n > 0 else 0


def _group_folder(source_path: str) -> str:
    """Folder a recording is grouped under in the left tree (and per-folder
    export). Pooled/processed data is a flat set of symlinks under work_pool/,
    so dirname alone collapses every recording into one node -- resolve the
    symlink to recover the real source dir (e.g. processed_data/<batch>), so it
    groups like raw data groups by its {subject}/{session} dir.

    Only the pool case is resolved: realpath() stats every path component, which
    is slow over network mounts (raw data lives on /mnt) and pointless for real
    files. We gate on the cheap string check "is this under a work_pool* dir?"
    so non-pooled recordings skip the filesystem hit entirely."""
    d = os.path.dirname(source_path)
    if os.path.basename(d).startswith("work_pool"):
        try:
            return os.path.dirname(os.path.realpath(source_path))
        except OSError:
            pass
    return d

# matplotlib (pyplot-free Figure/Agg) is not thread-safe; serialize all hand
# renders behind one lock. _HAND_PREP caches the per-recording normalized-mm
# skeleton subset + axis cube so we load each npz only once.
RENDER_LOCK = threading.Lock()
_HAND_PREP: dict = {}
_HAND_PREP_LOCK = threading.Lock()
_HAND_MODEL = None               # (emg2pose.visualization, base_profile, mirrored)
_HAND_MODEL_LOCK = threading.Lock()


def _hand_model():
    """emg2pose hand model (loaded once). Returns (viz_module, base_profile,
    mirrored_profile). Skinning with a cached profile is ~6ms/frame; without it
    the model reloads every call (~30-50ms)."""
    global _HAND_MODEL
    if _HAND_MODEL is None:
        with _HAND_MODEL_LOCK:
            if _HAND_MODEL is None:
                import emg2pose.visualization as ev
                base = ev.load_default_hand_model()
                _HAND_MODEL = (ev, base, ev.mirror_profile(base))
    return _HAND_MODEL


def _clip_key(stem: str, clip_id) -> str:
    return f"{stem}__c{int(clip_id):04d}"


def _jnum(x):
    """JSON-safe number: finite float, else None (NaN / inf / '' / non-numeric)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _jstr(x) -> str:
    """JSON-safe string ('' for None / NaN)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return str(x)


# ----------------------------------------------------------------------------
# State: load clips.csv once, build key->row index, load existing labels.
# A single lock guards label writes (server is the only writer).
# ----------------------------------------------------------------------------
class Store:
    def __init__(self, out_dir: str):
        self.out = out_dir
        self.lock = threading.Lock()
        # The index db is the source of clips/bursts/recordings; the annotations
        # table is the label/invalid truth. One shared connection (writes are
        # serialized behind self.lock).
        self.conn = dbmod.connect(out_dir)
        # All render caches live under one umbrella dir (out/cache/) so the
        # output root stays tidy; makedirs creates the parent on first run.
        cache_root = os.path.join(out_dir, "cache")
        self.rec_cache_dir = os.path.join(cache_root, "recording_cache")
        os.makedirs(self.rec_cache_dir, exist_ok=True)
        # v2 = deg->rad fix; v3 = variable-fps frame grid (POSE_FPS) -- bumping
        # the dir invalidates frames cached under the old fixed-600 grid.
        self.hand_dir = os.path.join(cache_root, "mesh_cache_v3")
        os.makedirs(self.hand_dir, exist_ok=True)
        self.overview_cache_dir = os.path.join(cache_root, "overview_cache")
        os.makedirs(self.overview_cache_dir, exist_ok=True)
        self.regen_always = os.environ.get("OVERVIEW_REGEN") == "1"
        self._segs_by_source = None      # lazy burst spans from db bursts table
        self._segs_lock = threading.Lock()
        self._overview_scan = None       # lazy source_file -> overview.png map
        self._overview_scan_lock = threading.Lock()
        self._stem_cache: dict[str, str] = {}

        # Clips from the db (+ source_path joined from recordings). Rename the
        # new schema columns to the names this server already uses internally, so
        # the geometry/overview/export code is unchanged.
        df = pd.read_sql_query(
            "SELECT c.*, r.source_path FROM clips c "
            "LEFT JOIN recordings r ON r.source_file = c.source_file", self.conn)
        if df.empty:
            raise SystemExit(f"no clips in {dbmod.db_path(out_dir)}; run segment.py first")
        df = df.rename(columns={"start_sample": "clip_start_sample",
                                "end_sample": "clip_end_sample",
                                "matched_burst_idx": "matched_emg_seg_idx"})
        df["clip_id"] = df["clip_id"].astype(int)
        df["source_file"] = df["source_file"].astype(str)
        self.df = df

        # recording durations / thresholds / lag QC / seg_version for the
        # overview info line (from the recordings table).
        rec_dur, rec_pose_thr, rec_enter, rec_exit = {}, {}, {}, {}
        rec_pose_exit = {}
        rec_lag, rec_corr, rec_lagflag, rec_segver = {}, {}, {}, {}
        r = pd.read_sql_query("SELECT * FROM recordings", self.conn)
        if not r.empty:
            sfk = r["source_file"].astype(str)
            rec_dur = dict(zip(sfk, r["duration_s"]))
            for col, dst in [("pose_thresh", rec_pose_thr),
                             ("pose_exit_thresh", rec_pose_exit),
                             ("enter_thresh", rec_enter),
                             ("exit_thresh", rec_exit),
                             ("emg_pose_lag_s", rec_lag),
                             ("emg_pose_corr", rec_corr),
                             ("lag_flag", rec_lagflag),
                             ("seg_version", rec_segver)]:
                if col in r.columns:
                    dst.update(zip(sfk, r[col]))

        # Per-recording metadata via a vectorized groupby -- NO per-clip Python
        # loop, so startup stays fast even at 200k+ clips. Each recording's
        # clips are built lazily (only when that recording is opened). Group by
        # source_path (unique per recording).
        df["_review"] = df["matched_emg_seg_idx"].to_numpy() < 0
        groups = df.groupby("source_path", sort=False)
        agg = groups.agg(
            source_file=("source_file", "first"),
            hand=("hand", "first"), subject=("subject", "first"),
            group=("group", "first"), n_clips=("clip_id", "size"),
            n_review=("_review", "sum"))
        gidx = groups.groups
        self.by_stem: dict[str, dict] = {}
        self.stem_to_source: dict[str, str] = {}
        for source_path, a in agg.iterrows():
            stem = self._stem_of(str(source_path))
            sf = str(a["source_file"])
            hand = str(a["hand"])
            self.by_stem[stem] = {
                "source_path": str(source_path), "source_file": sf,
                "folder": _group_folder(str(source_path)),
                "hand": (hand if hand and hand.lower() != "nan" else None),
                "subject": str(a["subject"]), "group": str(a["group"]),
                "rec_duration_s": float(rec_dur.get(sf, 0.0)),
                "pose_thresh": float(rec_pose_thr.get(sf, float("nan"))),
                "pose_exit_thresh": float(rec_pose_exit.get(sf, float("nan"))),
                "enter_thresh": float(rec_enter.get(sf, float("nan"))),
                "exit_thresh": float(rec_exit.get(sf, float("nan"))),
                "emg_pose_lag_s": rec_lag.get(sf),
                "emg_pose_corr": rec_corr.get(sf),
                "lag_flag": rec_lagflag.get(sf),
                "seg_version": rec_segver.get(sf),
                "row_idx": gidx[source_path],
                "n_clips": int(a["n_clips"]), "n_review": int(a["n_review"]),
                # expected hand-frame count (== len(fidx)) from duration alone,
                # so we can flag the 3D cache without loading the npz
                "n_frames": _frame_count(
                    int(round(float(rec_dur.get(sf, 0.0)) * FS))),
            }
            self.stem_to_source[stem] = sf

        # (source_file, clip_id) -> (clip_start, clip_end, seg_version) for the
        # re-segmentation-stable label anchor written into annotations.
        self._clip_meta = {
            (str(r.source_file), int(r.clip_id)):
                (int(r.clip_start_sample), int(r.clip_end_sample), r.seg_version)
            for r in self.df.itertuples()}

        self.labels: dict = {}           # (source_file, clip_id) -> label record
        self.labeled_count: dict = {}    # source_file -> non-empty gesture count
        self.invalid_count: dict = {}    # source_file -> clips marked invalid
        self._load_labels()

        # Whole-recording invalidation: source_file -> {labeler, marked_at, note}.
        # Such recordings drop out of labelling, export, clustering and eval.
        # Stored as scope='recording', kind='invalid' annotations in the db.
        self.invalid_recordings: dict = {}
        self._load_invalid_recordings()

    def _stem_of(self, source_path: str) -> str:
        s = self._stem_cache.get(source_path)
        if s is None:
            s = io_utils.parse_file_info(source_path).stem
            self._stem_cache[source_path] = s
        return s

    def _resolve_key(self, key: str):
        """clip key '{stem}__c{cid:04d}' -> (stem, source_file, clip_id)."""
        stem, _, cids = key.rpartition("__c")
        sf = self.stem_to_source.get(stem)
        return stem, sf, int(cids)

    def _load_labels(self):
        """Populate the in-memory label dict from clip-scope annotations in the
        db (kind 'label' or 'invalid' -- mutually exclusive per clip)."""
        for a in dbmod.get_annotations(self.conn, scope="clip"):
            sf, cid = str(a["source_file"]), int(a["clip_id"])
            rec = self.labels.setdefault(
                (sf, cid), {"gesture_label": "", "invalid": False,
                            "labeler": "", "labeled_at": "", "note": ""})
            if a["kind"] == "label":
                rec["gesture_label"] = a["value"] or ""
            elif a["kind"] == "invalid":
                rec["invalid"] = True
            rec["labeler"] = a["labeler"] or rec["labeler"]
            rec["labeled_at"] = a["labeled_at"] or rec["labeled_at"]
            rec["note"] = a["note"] or rec["note"]
        for (sf, _cid), rec in self.labels.items():
            if rec["gesture_label"]:
                self.labeled_count[sf] = self.labeled_count.get(sf, 0) + 1
            if rec["invalid"]:
                self.invalid_count[sf] = self.invalid_count.get(sf, 0) + 1

    def _hand_count(self, stem: str) -> int:
        d = os.path.join(self.hand_dir, stem)
        return (len([f for f in os.listdir(d) if f.endswith(".json")])
                if os.path.isdir(d) else 0)

    # --- recordings list (lazy: clips load per recording) -------------------
    def recordings_payload(self) -> dict:
        used = sorted({l["gesture_label"] for l in self.labels.values()
                       if l["gesture_label"]})
        recs = []
        for stem, st in self.by_stem.items():
            sf = st["source_file"]
            nfr = st["n_frames"]
            cnt = self._hand_count(stem)
            cached = 2 if (nfr and cnt >= nfr) else (1 if cnt > 0 else 0)
            recs.append({
                "stem": stem, "source_file": sf, "folder": st["folder"],
                "source_path": st["source_path"],
                "subject": st["subject"], "hand": st["hand"],
                "group": st["group"], "n_clips": st["n_clips"],
                "n_review": st["n_review"],
                "n_labeled": self.labeled_count.get(sf, 0),
                "n_invalid": self.invalid_count.get(sf, 0),
                "rec_invalid": sf in self.invalid_recordings,
                "cached": cached,    # 0 none / 1 partial / 2 full 3D cache
                # recording-level segmentation QC for the overview info line
                "emg_pose_lag_s": _jnum(st.get("emg_pose_lag_s")),
                "emg_pose_corr": _jnum(st.get("emg_pose_corr")),
                "lag_flag": _jstr(st.get("lag_flag")),
                "enter_thresh": _jnum(st.get("enter_thresh")),
                "exit_thresh": _jnum(st.get("exit_thresh")),
                "pose_thresh": _jnum(st.get("pose_thresh")),
                "pose_exit_thresh": _jnum(st.get("pose_exit_thresh")),
                "seg_version": _jstr(st.get("seg_version")),
            })
        return {"fs": FS, "labels_used": used, "recordings": recs,
                "total_clips": int(len(self.df)),
                "total_labeled": sum(self.labeled_count.values()),
                "total_invalid": sum(self.invalid_count.values()),
                "total_invalid_rec": len(self.invalid_recordings)}

    # --- upsert one label + atomic flush ------------------------------------
    def set_label(self, key, gesture_label, labeler, note, invalid=False) -> dict:
        stem, sf, cid = self._resolve_key(key)
        if sf is None:
            raise KeyError(key)
        inv = bool(invalid)
        gl = "" if inv else gesture_label.strip()   # invalid & gesture are exclusive
        with self.lock:
            prev = self.labels.get((sf, cid), {})
            pg, pi = prev.get("gesture_label", ""), prev.get("invalid", False)
            self.labels[(sf, cid)] = {
                "gesture_label": gl, "invalid": inv, "labeler": labeler.strip(),
                "labeled_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
                "note": note.strip(),
            }
            if gl and not pg:
                self.labeled_count[sf] = self.labeled_count.get(sf, 0) + 1
            elif pg and not gl:
                self.labeled_count[sf] = max(0, self.labeled_count.get(sf, 0) - 1)
            if inv and not pi:
                self.invalid_count[sf] = self.invalid_count.get(sf, 0) + 1
            elif pi and not inv:
                self.invalid_count[sf] = max(0, self.invalid_count.get(sf, 0) - 1)
            # Persist as a clip annotation (label & invalid are exclusive kinds);
            # carry the sample interval + seg_version as a re-seg-stable anchor.
            now = self.labels[(sf, cid)]["labeled_at"]
            start, end, sv = self._clip_meta.get((sf, cid), (None, None, None))
            dbmod.delete_annotation(self.conn, sf, cid, "clip", "label")
            dbmod.delete_annotation(self.conn, sf, cid, "clip", "invalid")
            if inv:
                dbmod.set_annotation(self.conn, sf, cid, "clip", "invalid",
                                     clip_start_sample=start, clip_end_sample=end,
                                     seg_version=sv, labeler=labeler.strip(),
                                     labeled_at=now, note=note.strip())
            elif gl:
                dbmod.set_annotation(self.conn, sf, cid, "clip", "label", value=gl,
                                     clip_start_sample=start, clip_end_sample=end,
                                     seg_version=sv, labeler=labeler.strip(),
                                     labeled_at=now, note=note.strip())
        return {"ok": True, "total": int(len(self.df)),
                "total_labeled": sum(self.labeled_count.values()),
                "total_invalid": sum(self.invalid_count.values()),
                "stem": stem, "stem_labeled": self.labeled_count.get(sf, 0),
                "stem_invalid": self.invalid_count.get(sf, 0)}

    # --- whole-recording invalid flag (scope='recording' annotation) --------
    def _load_invalid_recordings(self):
        for a in dbmod.get_annotations(self.conn, scope="recording", kind="invalid"):
            self.invalid_recordings[str(a["source_file"])] = {
                "labeler": a["labeler"] or "", "marked_at": a["labeled_at"] or "",
                "note": a["note"] or ""}

    def set_recording_invalid(self, stem, invalid, labeler="", note="") -> dict:
        """Mark/unmark a WHOLE recording invalid (soft exclude -- distinct from
        drop_recording, which permanently deletes). Invalid recordings drop out
        of labelling, export, clustering and eval. Keyed by source_file."""
        sf = self.stem_to_source.get(stem)
        if sf is None:
            raise KeyError(stem)
        with self.lock:
            if invalid:
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self.invalid_recordings[sf] = {
                    "labeler": labeler.strip(), "marked_at": now,
                    "note": note.strip()}
                dbmod.set_annotation(self.conn, sf, -1, "recording", "invalid",
                                     labeler=labeler.strip(), labeled_at=now,
                                     note=note.strip())
            else:
                self.invalid_recordings.pop(sf, None)
                dbmod.delete_annotation(self.conn, sf, -1, "recording", "invalid")
        return {"ok": True, "stem": stem,
                "rec_invalid": sf in self.invalid_recordings,
                "total_invalid_rec": len(self.invalid_recordings)}

    def drop_recording(self, stem) -> dict:
        """PERMANENTLY delete a recording: remove its db rows + annotations,
        tombstone it (so re-segmentation will not resurrect it), delete the
        on-disk shard + feature cache, and drop it from the in-memory state."""
        sf = self.stem_to_source.get(stem)
        if sf is None:
            raise KeyError(stem)
        with self.lock:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.conn.close()                      # drop_recording opens its own
            dbmod.drop_recording(self.out, sf, dropped_at=now, note="dropped via UI")
            self.conn = dbmod.connect(self.out)
            import shutil
            shutil.rmtree(os.path.join(self.out, "shards", stem), ignore_errors=True)
            fc = os.path.join(self.out, "features", stem + ".npz")
            if os.path.isfile(fc):
                os.remove(fc)
            # drop from in-memory state. Do NOT reset_index -- by_stem['row_idx']
            # holds original index labels into self.df, used label-wise elsewhere.
            self.df = self.df[self.df["source_file"] != sf]
            self.by_stem.pop(stem, None)
            self.stem_to_source.pop(stem, None)
            self.invalid_recordings.pop(sf, None)
            self.labeled_count.pop(sf, None)
            self.invalid_count.pop(sf, None)
            for k in [k for k in self.labels if k[0] == sf]:
                self.labels.pop(k, None)
            self._segs_by_source = None            # invalidate lazy burst cache
        return {"ok": True, "stem": stem, "dropped": sf,
                "total_clips": int(len(self.df))}

    @staticmethod
    def _folder_key(folder: str) -> str:
        """Source folder path -> safe filename stem ({subject}__{session})."""
        parts = [x for x in str(folder).split("/") if x]
        base = "__".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "root")
        return re.sub(r"[^A-Za-z0-9._-]", "_", base) or "folder"

    # --- export: one CSV per source folder + a merged clips_labeled.csv ------
    def export(self) -> dict:
        df = self.df.drop(columns=["_review", "gesture_label", "label_note"],
                          errors="ignore").copy()
        if self.labels:
            rows = [(sf, cid, l["gesture_label"], l["note"])
                    for (sf, cid), l in self.labels.items()]
            ldf = pd.DataFrame(rows, columns=["source_file", "clip_id",
                                              "gesture_label", "label_note"])
            df = df.merge(ldf, on=["source_file", "clip_id"], how="left")
        else:
            df["gesture_label"] = ""; df["label_note"] = ""
        df["gesture_label"] = df["gesture_label"].fillna("")
        df["label_note"] = df["label_note"].fillna("")
        # Drop whole recordings marked invalid, then keep only labelled clips
        # (empty gesture_label = discard, per convention).
        if self.invalid_recordings:
            df = df[~df["source_file"].astype(str).isin(self.invalid_recordings)]
        df = df[df["gesture_label"].astype(str).str.strip() != ""].copy()

        # one CSV per source folder (= {subject}/{session} dir, like the tree)
        sub_dir = os.path.join(self.out, "clips_labeled")
        os.makedirs(sub_dir, exist_ok=True)
        folders = df["source_path"].map(lambda p: _group_folder(str(p)))
        seen, n_files = {}, 0
        for folder, g in df.groupby(folders, sort=True):
            key = self._folder_key(folder)
            if key in seen:                    # distinct folders, same safe name
                seen[key] += 1; key = f"{key}_{seen[key]}"
            else:
                seen[key] = 0
            path = os.path.join(sub_dir, key + ".csv")
            tmp = path + ".tmp"; g.to_csv(tmp, index=False); os.replace(tmp, path)
            n_files += 1

        merged = os.path.join(self.out, "clips_labeled.csv")
        tmp = merged + ".tmp"; df.to_csv(tmp, index=False); os.replace(tmp, merged)
        return {"n_labeled": int(len(df)), "n_folders": n_files,
                "dir": sub_dir, "merged": merged}

    def _overview_xrange(self, png):
        """Exact data-axis left/right edges (image fractions) of overview.png,
        read from the navy pose-speed curve which spans x0..x1. plot_overview_dual
        uses tight_layout, whose margins shift with y-tick width AND fall back to
        matplotlib defaults (0.125/0.9) when it can't fit -- so a fixed constant
        is wrong. Detecting the rendered curve is exact (~1px) for any case."""
        try:
            import matplotlib.image as mpimg
            im = mpimg.imread(png)[..., :3]
            w = im.shape[1]
            R, G, B = im[..., 0], im[..., 1], im[..., 2]
            navy = (B > 0.3) & (R < 0.35) & (G < 0.35) & (B > R + 0.15) & (B > G + 0.15)
            cols = np.where(navy.any(axis=0))[0]
            if not len(cols):
                return (None, None)
            return (round(float(cols.min()) / w, 4), round(float(cols.max()) / w, 4))
        except Exception:
            return (None, None)

    def _load_bursts(self, stem: str):
        """Lazy burst spans from the db bursts table -> (spans, apexes,
        burst_idx->pos)."""
        if self._segs_by_source is None:
            with self._segs_lock:                  # double-checked; build then publish
                if self._segs_by_source is None:
                    segs: dict = {}
                    sdf = pd.read_sql_query(
                        "SELECT source_file, burst_idx, start_sample, "
                        "end_sample, apex_sample FROM bursts", self.conn)
                    if not sdf.empty:
                        sdf["source_file"] = sdf["source_file"].astype(str)
                        for sf, g in sdf.groupby("source_file", sort=False):
                            g = g.sort_values("burst_idx")
                            segs[sf] = (
                                list(zip(g["start_sample"].astype(int),
                                         g["end_sample"].astype(int))),
                                list(g["apex_sample"].astype(int)),
                                {int(s): i for i, s
                                 in enumerate(g["burst_idx"].astype(int))})
                    self._segs_by_source = segs    # publish fully-built map
        sf = self.by_stem[stem]["source_file"]
        return self._segs_by_source.get(sf, ([], [], {}))

    def _regen_overview(self, stem: str, env, ps):
        """Regenerate overview.png from intermediate data (segments.csv bursts +
        clips.csv clips + recordings.csv thresholds) via the SAME
        plot_overview_dual segment.py uses -> identical look. Caches the png +
        the exact axes box (captured at render, so the playhead is pixel-exact
        with no PNG detection). Returns (png_path, x0, x1)."""
        png = os.path.join(self.overview_cache_dir, stem + ".png")
        side = os.path.join(self.overview_cache_dir, stem + ".json")
        if os.path.isfile(png) and os.path.isfile(side):
            with open(side) as f:
                b = json.load(f)
            if b.get("v") == OVERVIEW_REGEN_V:
                return png, b.get("x0"), b.get("x1")
        st = self.by_stem[stem]
        bursts, apex, seg2pos = self._load_bursts(stem)
        sub = self.df.loc[st["row_idx"]]
        cols = set(sub.columns)

        def _clip_dict(r):
            d = dict(clip_id=int(r.clip_id),
                     clip_start=int(r.clip_start_sample),
                     clip_end=int(r.clip_end_sample),
                     motion_start=int(r.motion_start_sample),
                     motion_end=int(r.motion_end_sample))
            # The blue hold span: pass the REAL hold so plot_overview_dual draws
            # it, not the [motion_end, clip_end] fallback. clip_end is post-static
            # capped (~0.55s in the old SMC schema) so the fallback under-draws a
            # ~1s hold to half. Prefer hold_* (new schema), else static_out_* (old).
            if "hold_start_sample" in cols:
                d["hold_start"] = int(r.hold_start_sample)
                d["hold_end"] = int(r.hold_end_sample)
            if "static_out_start_sample" in cols:
                d["static_out_start"] = int(r.static_out_start_sample)
                d["static_out_end"] = int(r.static_out_end_sample)
            return d

        clips = [_clip_dict(r) for r in sub.itertuples()]
        c2b = [seg2pos.get(int(r.matched_emg_seg_idx), -1) for r in sub.itertuples()]
        thr = lambda v: (None if v is None or not np.isfinite(v) else float(v))
        # Recompute move_enter/move_exit LIVE from THIS curve, exactly the way
        # action_segmentation.resolve_move_thresholds does in auto mode:
        # move_enter = robust_threshold(spd); move_exit = move_exit_frac (0.5) *
        # move_enter. This MUST be live, not read from recordings.csv: load_npz
        # normalizes joint angles to radians, so ps is rad/s, but recordings.csv's
        # pose_thresh can be in stale deg units (written by an older run pre the
        # deg->rad fix) -- ~57x larger, which would push the y-cap (pose_thr*2)
        # off the top and squash the curve to ~0. A live threshold is always in
        # the curve's own units, so curve + lines + y-cap can't drift apart.
        # (EMG enter/exit are unaffected -- EMG units never changed -- so those
        # still come from recordings.csv.)
        from emg_label.pose_segmentation import robust_threshold
        MOVE_EXIT_FRAC = 0.5           # == config.Config.move_exit_frac default
        move_enter = float(robust_threshold(np.asarray(ps, dtype=float)))
        if not np.isfinite(move_enter) or move_enter <= 0:
            move_enter = None          # near-static: let plotting percentile-cap
        move_exit = MOVE_EXIT_FRAC * move_enter if move_enter else None
        import matplotlib.figure as _mf
        cap = {}
        with RENDER_LOCK:                          # plot_overview_dual uses pyplot
            orig = _mf.Figure.savefig
            def _cap(self_, *a, **k):
                self_.canvas.draw()
                bb = self_.axes[0].get_position()
                cap["x0"], cap["x1"] = round(bb.x0, 4), round(bb.x1, 4)
                return orig(self_, *a, **k)
            _mf.Figure.savefig = _cap
            tmp = png + ".tmp.png"
            try:
                plotting.plot_overview_dual(
                    env, ps, FS, tmp, burst_segments=bursts, clips=clips,
                    clip_to_burst=c2b, enter_thr=thr(st["enter_thresh"]),
                    exit_thr=thr(st["exit_thresh"]),
                    pose_thr=move_enter, pose_exit_thr=move_exit,
                    apex_samples=apex)
            finally:
                _mf.Figure.savefig = orig
        os.replace(tmp, png)
        with open(side, "w") as f:
            json.dump({"v": OVERVIEW_REGEN_V, "x0": cap.get("x0"),
                       "x1": cap.get("x1")}, f)
        return png, cap.get("x0"), cap.get("x1")

    def overview_image(self, stem: str) -> str | None:
        """Path of the overview png to SERVE (used by /api/overview).

        In regen mode, rebuild the cached png when it's missing or version-stale
        (OVERVIEW_REGEN_V) -- this is independent of the recording-geometry
        cache, so an OVERVIEW_REGEN_V bump actually invalidates served images
        (the bug before: the handler served any existing png without checking
        its version). In static mode, prefer the tuned shard overview.png."""
        if not self.regen_always:
            static = self.overview_path(stem)
            if static:
                return static
        png = os.path.join(self.overview_cache_dir, stem + ".png")
        side = os.path.join(self.overview_cache_dir, stem + ".json")
        try:
            if os.path.isfile(png) and os.path.isfile(side):
                with open(side) as f:
                    if json.load(f).get("v") == OVERVIEW_REGEN_V:
                        return png
        except Exception:
            pass
        # stale or missing -> recompute curves and regen (version-stamped)
        from emg_label.segmentation import emg_envelope
        st = self.by_stem[stem]
        side_hand = (st["hand"]
                     or io_utils.parse_file_info(st["source_path"]).hand or "left")
        emg, ja = io_utils.load_npz(st["source_path"], hand=side_hand)
        ps = _seg_pose_speed(ja)
        env = np.asarray(emg_envelope(emg, FS), dtype=float)
        p, _, _ = self._regen_overview(stem, env, ps)
        return p

    # --- whole-recording overview curves + frame grid (cached geometry) ----
    def _recording_geometry(self, stem: str) -> dict:
        cache = os.path.join(self.rec_cache_dir, stem + ".json")
        if os.path.isfile(cache):
            with open(cache) as f:
                d = json.load(f)
            if d.get("v") == OVERVIEW_REGEN_V:   # tied to overview version so an
                return d                         # overview bump invalidates ov_x0/x1
        st = self.by_stem[stem]
        side = (st["hand"]
                or io_utils.parse_file_info(st["source_path"]).hand or "left")

        emg, ja = io_utils.load_npz(st["source_path"], hand=side)
        n = len(ja)
        # Hand frames are rendered server-side as emg2pose mesh vertices (JSON,
        # via /api/handmesh) and indexed by this same downsampled grid -- the
        # payload only carries frame_times so the scrubber maps time -> frame i.
        fidx = _frame_indices(n)

        # Overview curves: pose-speed + EMG envelope, downsampled for plotting.
        from emg_label.segmentation import emg_envelope
        ps = _seg_pose_speed(ja)
        env = np.asarray(emg_envelope(emg, FS), dtype=float)
        cstep = max(1, n // CURVE_MAX_POINTS)
        cidx = list(range(0, n, cstep))
        cs_t = (np.asarray(cidx) / FS).round(3).tolist()

        def clean(a):  # NaN -> None for JSON
            return [None if not np.isfinite(x) else round(float(x), 3)
                    for x in a[cidx]]

        # Segments (clips) in time order -> s1..sN geometry (labels added live).
        # Built lazily from this recording's df rows only.
        sub = self.df.loc[st["row_idx"]]
        segs = []
        for _, r in sub.iterrows():
            cid = int(r["clip_id"])
            segs.append({
                "key": _clip_key(stem, cid), "clip_id": cid,
                "start_s": int(r["clip_start_sample"]) / FS,
                "end_s": int(r["clip_end_sample"]) / FS,
                "motion_start_s": int(r["motion_start_sample"]) / FS,
                "motion_end_s": int(r["motion_end_sample"]) / FS,
                "apex_s": int(r["apex_sample"]) / FS,
                "review": int(r["matched_emg_seg_idx"]) < 0,
                "envelope_peak": float(r["envelope_peak"]),
                "pose_range": float(r["pose_range"]),
                "motion_duration_s": float(r["motion_duration_s"]),
                "duration_s": float(r["duration_s"]),
                "matched_emg_seg_idx": int(r["matched_emg_seg_idx"]),
            })
        segs.sort(key=lambda s: s["start_s"])
        for i, s in enumerate(segs, start=1):
            s["s"] = i

        pthr = st["pose_thresh"]
        # Overview: prefer the tuned static overview.png; regenerate from
        # intermediate data when it's missing (or when OVERVIEW_REGEN=1).
        static = None if self.regen_always else self.overview_path(stem)
        if static:
            ov_x0, ov_x1 = self._overview_xrange(static)
        else:
            try:
                _, ov_x0, ov_x1 = self._regen_overview(stem, env, ps)
            except Exception as ex:
                print(f"overview regen failed for {stem}: {ex}")
                ov_x0 = ov_x1 = None
        out = {
            "v": OVERVIEW_REGEN_V, "stem": stem, "fps": POSE_FPS,  # share the
                                                     # overview version: ov_x0/x1
                                                     # come from the overview, so
                                                     # an overview bump invalidates
                                                     # this geometry cache too
            "duration_s": round(n / FS, 3),
            "n_frames": len(fidx),
            "ov_x0": ov_x0, "ov_x1": ov_x1,   # overview data-axis edges (img frac)
            "frame_times": (np.asarray(fidx) / FS).round(3).tolist(),
            "curves": {
                "t": cs_t, "pose_speed": clean(ps), "emg_env": clean(env),
                "pose_thr": (None if not np.isfinite(pthr) else round(pthr, 3)),
            },
            "segments": segs,
        }
        with open(cache, "w") as f:
            json.dump(out, f)
        return out

    def recording(self, stem: str) -> dict:
        if stem not in self.by_stem:
            raise KeyError(stem)
        geo = self._recording_geometry(stem)
        sf = self.by_stem[stem]["source_file"]
        # Overlay current labels live (geometry cache is label-agnostic).
        out = dict(geo)
        out["segments"] = [
            {**s, "gesture_label": self.labels.get((sf, s["clip_id"]), {}).get(
                "gesture_label", ""),
             "note": self.labels.get((sf, s["clip_id"]), {}).get("note", ""),
             "invalid": self.labels.get((sf, s["clip_id"]), {}).get("invalid", False)}
            for s in geo["segments"]]
        return out

    # --- server-side hand MESH (emg2pose skinning; Plotly Mesh3d on frontend,
    #     so it's a clear shaded mesh AND rotatable, vs the old flat PNG) -------
    def _mesh_prep(self, stem: str):
        """Load joint_angles once -> (ja_sub, profile, n_frames, viz) on the
        SAME downsampled grid as frame_times. profile = mirrored model for the
        left hand (emg2pose flip convention)."""
        prep = _HAND_PREP.get(stem)
        if prep is not None:
            return prep
        with _HAND_PREP_LOCK:
            prep = _HAND_PREP.get(stem)
            if prep is not None:
                return prep
            st = self.by_stem[stem]
            side = (st["hand"]
                    or io_utils.parse_file_info(st["source_path"]).hand
                    or "left")
            _, ja = io_utils.load_npz(st["source_path"], hand=side)
            n = len(ja)
            fidx = _frame_indices(n)
            ja_sub = np.asarray(ja[fidx], dtype=np.float32)
            # Manus ergonomics are in DEGREES (range ~±60); emg2pose skinning
            # expects RADIANS. Feeding degrees twists the fingers into a blob.
            # Convert when it clearly looks like degrees (leaves radian data
            # like emg2pose's own joint_angles untouched).
            finite = ja_sub[np.isfinite(ja_sub)]
            if finite.size and np.nanmax(np.abs(finite)) > 2 * np.pi:
                ja_sub = ja_sub * (np.pi / 180.0)
            ev, base, mirror = _hand_model()
            profile = mirror if str(side).lower().startswith("left") else base
            prep = (ja_sub, profile, len(fidx), ev)
            _HAND_PREP[stem] = prep
            return prep

    def _hand_path(self, stem: str, i: int) -> str:
        d = os.path.join(self.hand_dir, stem)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"f{i:04d}.json")

    def hand_frame(self, stem: str, i: int) -> bytes:
        """Per-frame hand-mesh vertices (int mm) as JSON bytes; cached on disk."""
        if stem not in self.by_stem:
            raise KeyError(stem)
        path = self._hand_path(stem, i)
        if not os.path.isfile(path):
            ja_sub, profile, nfr, ev = self._mesh_prep(stem)
            i = max(0, min(nfr - 1, i))
            path = self._hand_path(stem, i)
            if not os.path.isfile(path):
                # emg2pose skinning isn't verified re-entrant; serialize the
                # compute behind RENDER_LOCK (same lock the overview render uses).
                with RENDER_LOCK:
                    verts = np.asarray(ev.skin_vertices_np(profile, ja_sub[i]))
                body = json.dumps({"v": np.rint(verts).astype(int).tolist()})
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    f.write(body)
                os.replace(tmp, path)
                return body.encode()
        with open(path, "rb") as f:
            return f.read()

    def hand_faces(self) -> dict:
        """Mesh triangle indices (same for left/right) -> Plotly Mesh3d i,j,k."""
        ev, base, _ = _hand_model()
        tris = np.asarray(base.mesh_triangles)
        return {"i": tris[:, 0].tolist(), "j": tris[:, 1].tolist(),
                "k": tris[:, 2].tolist()}

    def overview_path(self, stem: str):
        """Resolve shards/<dir>/overview.png for a recording. The shard dir is
        named by segment.py's FileInfo.stem, which for --meta runs is the meta
        `sample_id` (or the raw filename stem) and may NOT equal the synthesized
        parse_file_info stem this server keys on. Try the likely names, then
        fall back to scanning every shard's recording.csv by source_file."""
        r = self.by_stem.get(stem)
        # --out pointed at a single shard dir (overview.png sits directly under
        # it, not under shards/<stem>/) -- common mistake, handle it.
        direct = os.path.join(self.out, "overview.png")
        if os.path.isfile(direct):
            return direct
        root = os.path.join(self.out, "shards")
        cands = [stem]
        if r:
            cands.append(os.path.splitext(os.path.basename(r["source_path"]))[0])
            cands.append(os.path.splitext(r["source_file"])[0])
        for c in cands:
            p = os.path.join(root, c, "overview.png")
            if os.path.isfile(p):
                return p
        if self._overview_scan is None:        # one-time scan, source_file -> png
            with self._overview_scan_lock:     # double-checked; build then publish
                if self._overview_scan is None:
                    scan: dict = {}
                    if os.path.isdir(root):
                        for name in os.listdir(root):
                            ov = os.path.join(root, name, "overview.png")
                            if not os.path.isfile(ov):
                                continue
                            rc = os.path.join(root, name, "recording.csv")
                            if os.path.isfile(rc):
                                try:
                                    rr = pd.read_csv(rc, nrows=1)
                                    if "source_file" in rr.columns:
                                        scan[str(rr["source_file"].iloc[0])] = ov
                                except Exception:
                                    pass
                    self._overview_scan = scan  # publish fully-built map
        return self._overview_scan.get(r["source_file"]) if r else None

    def hand_status(self, stem: str) -> dict:
        """Render progress = PNGs on disk. Rendering is driven by the browser's
        preloader (paced by image onload), not a background thread: a continuous
        server thread would hold the GIL in matplotlib and starve the on-demand
        request for the frame the user is actually viewing -> the hand only
        updated on pause. Counting files keeps the progress label working."""
        if stem not in self.by_stem:
            raise KeyError(stem)
        return {"done": self._hand_count(stem)}


# ----------------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------------
def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quieter
            pass

        def _send(self, code, body, ctype="application/json", no_cache=False):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if no_cache:                 # so frontend edits always take effect
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path, ctype, no_cache=False):
            if not os.path.isfile(path):
                return self._send(404, {"error": "not found"})
            with open(path, "rb") as f:
                self._send(200, f.read(), ctype, no_cache=no_cache)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            p = u.path
            if p == "/":
                return self._file(
                    os.path.join(WEBUI_DIR, "index.html"), "text/html",
                    no_cache=True)
            if p in ("/app.js", "/plotly.min.js"):
                return self._file(
                    os.path.join(WEBUI_DIR, p.lstrip("/")), "text/javascript",
                    no_cache=True)
            if p == "/api/recordings":
                return self._send(200, store.recordings_payload())
            if p == "/api/overview":
                stem = q.get("stem", [""])[0]
                # overview_image() version-checks the regen cache and rebuilds
                # when stale; no_cache because the URL is keyed only by stem, so
                # without it a browser serves a stale (e.g. wrong-unit) overview.
                try:
                    png = store.overview_image(stem)
                except Exception as ex:
                    print(f"WARN /api/overview regen {stem}: {ex}")
                    png = None
                if png and os.path.isfile(png):
                    return self._file(png, "image/png", no_cache=True)
                print(f"WARN /api/overview: no overview for stem={stem}")
                return self._send(404, {"error": "overview not found",
                                        "stem": stem})
            if p == "/api/recording":
                stem = q.get("stem", [""])[0]
                try:
                    payload = store.recording(stem)
                    return self._send(200, payload)
                except Exception as ex:
                    return self._send(
                        500, {"error": f"{type(ex).__name__}: {ex}"})
            if p == "/api/handmesh":
                stem = q.get("stem", [""])[0]
                try:
                    i = int(q.get("i", ["0"])[0])
                    return self._send(200, store.hand_frame(stem, i),
                                      "application/json")
                except Exception as ex:
                    return self._send(
                        500, {"error": f"{type(ex).__name__}: {ex}"})
            if p == "/api/handfaces":
                try:
                    return self._send(200, store.hand_faces())
                except Exception as ex:
                    return self._send(
                        500, {"error": f"{type(ex).__name__}: {ex}"})
            if p == "/api/handstatus":
                stem = q.get("stem", [""])[0]
                try:
                    return self._send(200, store.hand_status(stem))
                except Exception as ex:
                    return self._send(
                        500, {"error": f"{type(ex).__name__}: {ex}"})
            if p == "/api/export":
                return self._send(200, store.export())
            return self._send(404, {"error": "no route"})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path not in ("/api/label", "/api/invalid_recording",
                              "/api/drop_recording"):
                return self._send(404, {"error": "no route"})
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            try:
                if u.path == "/api/drop_recording":
                    res = store.drop_recording(data["stem"])
                elif u.path == "/api/invalid_recording":
                    res = store.set_recording_invalid(
                        data["stem"], bool(data.get("invalid", False)),
                        data.get("labeler", ""), data.get("note", ""))
                else:
                    res = store.set_label(
                        data["key"], data.get("gesture_label", ""),
                        data.get("labeler", ""), data.get("note", ""),
                        invalid=bool(data.get("invalid", False)))
                return self._send(200, res)
            except KeyError as ex:
                return self._send(400, {"error": f"unknown key {ex}"})
    return H


def _ensure_plotly_vendored():
    """Vendor webui/plotly.min.js from the installed plotly package if it's
    missing. The frontend prefers this local copy and only falls back to the
    CDN when it's absent -- but the CDN is 4.5MB and unreachable on offline /
    locked-down boxes, leaving the 3D hand panel spinning forever. The plotly
    pip package already ships the same minified bundle, so copy it once on
    startup. Best-effort: a failure just leaves the existing CDN fallback."""
    dst = os.path.join(WEBUI_DIR, "plotly.min.js")
    if os.path.isfile(dst):
        return
    try:
        import importlib.util
        import shutil
        spec = importlib.util.find_spec("plotly")
        if spec is None or not spec.origin:
            return
        src = os.path.join(os.path.dirname(spec.origin),
                           "package_data", "plotly.min.js")
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
            print(f"vendored plotly.min.js -> {dst}")
    except Exception as ex:
        print(f"note: could not vendor plotly.min.js ({ex}); "
              "3D hand will fall back to the CDN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_pose")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    _ensure_plotly_vendored()
    store = Store(args.out)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Labelling {len(store.df)} clips across {len(store.by_stem)} "
          f"recordings from {args.out}")
    print(f"Open http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
