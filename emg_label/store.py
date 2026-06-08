from __future__ import annotations

"""SQLite data layer for the EMG-gesture pipeline (the derived index + the
annotation truth + cluster-run results).

Design: docs/数据组织重构.md. The per-recording shards under ``OUT/shards/{stem}/``
remain the segmentation *truth*; ``OUT/index.db`` is a derived, queryable index
rebuildable from those shards at any time (``build_index``), PLUS two things that
are themselves truth and do NOT live in shards:

  * ``annotations`` -- the per-clip / per-recording human labels & invalid marks
    (one table, ``scope`` + ``kind``), carrying a re-segmentation-stable anchor
    (``clip_start_sample, clip_end_sample, seg_version``).
  * ``cluster_runs`` / ``cluster_assignments`` -- one row-set per clustering run
    (apex or trajectory channel), keyed on the same ``(source_file, clip_id)`` as
    clips and annotations, so label-driven evaluation is a single JOIN.

The clustering unit, the labelling unit and the evaluation unit are all the
**clip** ``(source_file, clip_id)``. ``bursts`` (EMG activations, formerly the
mis-named ``segments.csv``) are auxiliary.
"""

import datetime
import glob
import hashlib
import json
import os
import re
import sqlite3

import pandas as pd

DB_NAME = "index.db"

# Column order for each table, also used by export_csv() and (Phase 2) by
# segment.py when it writes the shard CSVs. source_path is kept on recordings
# (where to find the npz); bursts/clips reference the recording by source_file.
REC_COLS = [
    "source_file", "source_path", "group", "subject", "hand",
    "n_samples", "duration_s", "n_bursts", "n_clips",
    "n_burst_only", "n_clip_only",
    "emg_pose_lag_s", "emg_pose_corr", "lag_flag",
    "pose_nan_frac", "emg_nan_frac",
    "enter_thresh", "exit_thresh", "pose_thresh", "pose_exit_thresh",
    "rec_pose_range", "pose_static", "seg_version",
]
BURST_COLS = [
    "source_file", "burst_idx", "group",
    "start_sample", "end_sample", "hold_end_sample", "apex_sample",
    "duration_s", "emg_rms", "envelope_peak", "pose_range", "matched_clip_id",
]
CLIP_COLS = [
    "source_file", "clip_id", "group", "subject", "hand",
    "start_sample", "end_sample",
    "motion_start_sample", "motion_end_sample",
    "hold_start_sample", "hold_end_sample", "apex_sample",
    "duration_s", "motion_duration_s", "hold_duration_s",
    "emg_rms", "envelope_peak", "mean_pose_speed", "max_pose_speed",
    "pose_range", "matched_burst_idx", "fusion_type", "review_flag",
    "seg_version",
]
ANNOT_COLS = [
    "source_file", "clip_id", "scope", "kind", "value",
    "clip_start_sample", "clip_end_sample", "seg_version",
    "labeler", "labeled_at", "note",
]

def _qcols(cols: list) -> str:
    """Comma-joined, double-quoted identifiers (some columns like `group` are
    SQL reserved words, so every column is quoted)."""
    return ", ".join(f'"{c}"' for c in cols)


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS recordings (
  {_qcols(REC_COLS)},
  PRIMARY KEY (source_file)
);
CREATE TABLE IF NOT EXISTS bursts (
  {_qcols(BURST_COLS)},
  PRIMARY KEY (source_file, burst_idx)
);
CREATE TABLE IF NOT EXISTS clips (
  {_qcols(CLIP_COLS)},
  PRIMARY KEY (source_file, clip_id)
);
CREATE TABLE IF NOT EXISTS annotations (
  {_qcols(ANNOT_COLS)},
  PRIMARY KEY (source_file, clip_id, scope, kind)
);
CREATE TABLE IF NOT EXISTS cluster_runs (
  run_id TEXT PRIMARY KEY, created_at TEXT, channel TEXT, params_json TEXT
);
CREATE TABLE IF NOT EXISTS cluster_assignments (
  run_id TEXT, source_file TEXT, clip_id INTEGER, cluster_id INTEGER,
  PRIMARY KEY (run_id, source_file, clip_id)
);
CREATE TABLE IF NOT EXISTS tombstones (
  source_file TEXT PRIMARY KEY, dropped_at TEXT, note TEXT
);
CREATE INDEX IF NOT EXISTS ix_clips_sf ON clips(source_file);
CREATE INDEX IF NOT EXISTS ix_bursts_sf ON bursts(source_file);
CREATE INDEX IF NOT EXISTS ix_assign_run ON cluster_assignments(run_id);
"""


# ---------- connection ------------------------------------------------------

def db_path(out_dir: str) -> str:
    return os.path.join(out_dir, DB_NAME)


def connect(out_dir: str) -> sqlite3.Connection:
    """Open (creating if needed) ``OUT/index.db`` with the schema applied.

    check_same_thread=False so the threaded label server can share one
    connection; callers must serialize writes behind a lock (reads under WAL are
    fine). Row access is by name via sqlite3.Row.
    """
    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(db_path(out_dir), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    return conn


def _rows(df: pd.DataFrame, cols: list) -> list:
    """DataFrame -> list of tuples in `cols` order, filling missing cols with
    None so a shard CSV that lacks an optional column still inserts."""
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return list(df[cols].itertuples(index=False, name=None))


def _insert(conn, table: str, cols: list, rows: list) -> None:
    if not rows:
        return
    ph = ",".join(["?"] * len(cols))
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({_qcols(cols)}) VALUES ({ph})", rows)


# ---------- build the index from shards -------------------------------------

def build_index(out_dir: str, only_stems=None) -> dict:
    """(Re)build recordings/bursts/clips from ``OUT/shards/*/`` into index.db.

    Each shard's recording.csv is the commit marker. Tombstoned source_files are
    skipped (so a dropped recording is not resurrected by a re-segmentation that
    left its shard on disk). For each indexed recording the existing rows are
    deleted first, so a re-segmentation with fewer clips leaves no stale rows.
    Annotations / cluster_runs are NOT touched. Returns simple counts.
    """
    conn = connect(out_dir)
    try:
        tomb = {r[0] for r in conn.execute("SELECT source_file FROM tombstones")}
        shards = sorted(glob.glob(os.path.join(out_dir, "shards", "*", "recording.csv")))
        n_rec = n_burst = n_clip = 0
        with conn:  # one transaction
            for rec_csv in shards:
                shard = os.path.dirname(rec_csv)
                stem = os.path.basename(shard)
                if only_stems is not None and stem not in only_stems:
                    continue
                rec = pd.read_csv(rec_csv, dtype={"source_file": str})
                if rec.empty:
                    continue
                sf = str(rec["source_file"].iloc[0])
                if sf in tomb:
                    continue
                for t in ("recordings", "bursts", "clips"):
                    conn.execute(f"DELETE FROM {t} WHERE source_file=?", (sf,))
                _insert(conn, "recordings", REC_COLS, _rows(rec, REC_COLS))
                n_rec += 1
                bpath = os.path.join(shard, "bursts.csv")
                if os.path.isfile(bpath):
                    b = pd.read_csv(bpath, dtype={"source_file": str})
                    _insert(conn, "bursts", BURST_COLS, _rows(b, BURST_COLS))
                    n_burst += len(b)
                cpath = os.path.join(shard, "clips.csv")
                if os.path.isfile(cpath):
                    c = pd.read_csv(cpath, dtype={"source_file": str})
                    _insert(conn, "clips", CLIP_COLS, _rows(c, CLIP_COLS))
                    n_clip += len(c)
        return {"recordings": n_rec, "bursts": n_burst, "clips": n_clip}
    finally:
        conn.close()


def export_csv(out_dir: str) -> None:
    """Dump recordings/bursts/clips tables to top-level convenience CSVs
    (recordings.csv / bursts.csv / clips.csv) for dev tools that read flat files
    (diag_seg, cluster_traj, visualize_segment). source_path is normalized onto
    recordings in the db, so it is JOINed back into the bursts/clips dumps (those
    readers locate the npz via source_path). The db is the authoritative index;
    these CSVs are a derived courtesy."""
    conn = connect(out_dir)
    try:
        # bursts/clips: re-attach source_path from recordings (denormalize for
        # flat readers); place it right after source_file for readability.
        for table, name in (("bursts", "bursts.csv"), ("clips", "clips.csv")):
            df = pd.read_sql_query(
                f"SELECT t.*, r.source_path FROM {table} t "
                "LEFT JOIN recordings r ON r.source_file = t.source_file", conn)
            if "source_path" in df.columns:
                cols = (["source_file", "source_path"]
                        + [c for c in df.columns
                           if c not in ("source_file", "source_path")])
                df = df[cols]
            _atomic_csv(out_dir, name, df)
        _atomic_csv(out_dir, "recordings.csv",
                    pd.read_sql_query("SELECT * FROM recordings", conn))
    finally:
        conn.close()


def _atomic_csv(out_dir: str, name: str, df: pd.DataFrame) -> None:
    tmp = os.path.join(out_dir, name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(out_dir, name))


# ---------- queries ---------------------------------------------------------

def tombstoned(out_dir: str) -> set:
    """source_files that were permanently dropped (hard 'do not re-process').
    Opens the db read-only-ish; returns empty if no db yet."""
    if not os.path.isfile(db_path(out_dir)):
        return set()
    conn = connect(out_dir)
    try:
        return {r[0] for r in conn.execute("SELECT source_file FROM tombstones")}
    finally:
        conn.close()


def excluded_recordings(conn) -> set:
    """source_files excluded downstream: recording-scope invalid annotations
    UNION tombstones. Clustering / eval / export filter these out."""
    q = ("SELECT source_file FROM annotations "
         "WHERE scope='recording' AND kind='invalid' "
         "UNION SELECT source_file FROM tombstones")
    return {r[0] for r in conn.execute(q)}


def recordings_summary(conn) -> list:
    """Per-recording rows + n_labeled / n_invalid (clip-scope) + rec_invalid."""
    base = {r["source_file"]: dict(r)
            for r in conn.execute("SELECT * FROM recordings")}
    for r in conn.execute(
            "SELECT source_file, "
            "SUM(kind='label' AND value!='') AS n_labeled, "
            "SUM(kind='invalid') AS n_invalid "
            "FROM annotations WHERE scope='clip' GROUP BY source_file"):
        if r["source_file"] in base:
            base[r["source_file"]]["n_labeled"] = int(r["n_labeled"] or 0)
            base[r["source_file"]]["n_invalid"] = int(r["n_invalid"] or 0)
    rec_inv = {r[0] for r in conn.execute(
        "SELECT source_file FROM annotations "
        "WHERE scope='recording' AND kind='invalid'")}
    out = []
    for sf, d in base.items():
        d.setdefault("n_labeled", 0)
        d.setdefault("n_invalid", 0)
        d["rec_invalid"] = sf in rec_inv
        out.append(d)
    return out


def clips_for_recording(conn, source_file: str) -> list:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM clips WHERE source_file=? ORDER BY start_sample",
        (str(source_file),))]


def labeled_view(conn, include_invalid: bool = False) -> pd.DataFrame:
    """clips LEFT JOIN clip-scope label annotations -> a always-fresh labelled
    view (one row per clip, gesture_label filled from annotations)."""
    df = pd.read_sql_query(
        "SELECT c.*, a.value AS gesture_label, a.labeler, a.labeled_at, a.note "
        "FROM clips c LEFT JOIN annotations a "
        "ON a.source_file=c.source_file AND a.clip_id=c.clip_id "
        "AND a.scope='clip' AND a.kind='label'", conn)
    if not include_invalid:
        excl = excluded_recordings(conn)
        inv_clip = pd.read_sql_query(
            "SELECT source_file, clip_id FROM annotations "
            "WHERE scope='clip' AND kind='invalid'", conn)
        if not df.empty:
            df = df[~df["source_file"].isin(excl)]
            if not inv_clip.empty:
                key = df["source_file"].astype(str) + "#" + df["clip_id"].astype(str)
                bad = set(inv_clip["source_file"].astype(str) + "#"
                          + inv_clip["clip_id"].astype(str))
                df = df[~key.isin(bad)]
    return df.reset_index(drop=True)


# ---------- annotations (the labelling truth) -------------------------------

def set_annotation(conn, source_file, clip_id, scope, kind, value="",
                   clip_start_sample=None, clip_end_sample=None,
                   seg_version=None, labeler="", labeled_at="", note="") -> None:
    """Upsert one annotation. clip_id=-1 for scope='recording'. Pass the clip's
    sample interval + seg_version so a later re-segmentation can remap labels."""
    cid = -1 if scope == "recording" else int(clip_id)
    with conn:
        conn.execute(
            f"INSERT OR REPLACE INTO annotations ({_qcols(ANNOT_COLS)}) "
            f"VALUES ({','.join(['?'] * len(ANNOT_COLS))})",
            (str(source_file), cid, scope, kind, value,
             clip_start_sample, clip_end_sample, seg_version,
             labeler, labeled_at, note))


def delete_annotation(conn, source_file, clip_id, scope, kind) -> None:
    cid = -1 if scope == "recording" else int(clip_id)
    with conn:
        conn.execute(
            "DELETE FROM annotations WHERE source_file=? AND clip_id=? "
            "AND scope=? AND kind=?", (str(source_file), cid, scope, kind))


def get_annotations(conn, scope=None, kind=None) -> list:
    q, args = "SELECT * FROM annotations", []
    cond = []
    if scope is not None:
        cond.append("scope=?"); args.append(scope)
    if kind is not None:
        cond.append("kind=?"); args.append(kind)
    if cond:
        q += " WHERE " + " AND ".join(cond)
    return [dict(r) for r in conn.execute(q, args)]


# ---------- delete primitive (drop a whole recording) -----------------------

def drop_recording(out_dir: str, source_file: str, dropped_at: str = "",
                   note: str = "") -> None:
    """Permanently drop one recording: remove its index rows + annotations and
    write a tombstone so re-segmentation will not resurrect it. The caller is
    responsible for removing the on-disk shard dir (the stem is not known here)."""
    conn = connect(out_dir)
    try:
        with conn:
            for t in ("recordings", "bursts", "clips", "annotations",
                      "cluster_assignments"):
                conn.execute(f"DELETE FROM {t} WHERE source_file=?",
                             (str(source_file),))
            conn.execute(
                "INSERT OR REPLACE INTO tombstones (source_file, dropped_at, note) "
                "VALUES (?,?,?)", (str(source_file), dropped_at, note))
    finally:
        conn.close()


# ---------- cluster runs ----------------------------------------------------

def write_cluster_run(out_dir: str, run_id: str, channel: str, params: dict,
                      assignments, created_at: str = "") -> None:
    """Register a clustering run + its (source_file, clip_id, cluster_id) rows.
    `assignments` is a DataFrame or list of (source_file, clip_id, cluster_id)."""
    if isinstance(assignments, pd.DataFrame):
        rows = list(assignments[["source_file", "clip_id", "cluster_id"]]
                    .itertuples(index=False, name=None))
    else:
        rows = list(assignments)
    conn = connect(out_dir)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO cluster_runs "
                "(run_id, created_at, channel, params_json) VALUES (?,?,?,?)",
                (run_id, created_at, channel, json.dumps(params, default=str)))
            conn.execute("DELETE FROM cluster_assignments WHERE run_id=?", (run_id,))
            conn.executemany(
                "INSERT OR REPLACE INTO cluster_assignments "
                "(run_id, source_file, clip_id, cluster_id) VALUES (?,?,?,?)",
                [(run_id, str(sf), int(cid), int(cl)) for sf, cid, cl in rows])
    finally:
        conn.close()


def drop_run(out_dir: str, run_id: str, remove_dir: bool = True) -> bool:
    """Delete a clustering run from the db (cluster_runs + cluster_assignments)
    and, by default, its on-disk cluster_runs/{run_id}/ dir. Returns True if the
    run had any db rows. Deleting only the directory leaves stale db rows, so use
    this (or prune_runs) to keep the db and disk in sync."""
    conn = connect(out_dir)
    try:
        had = conn.execute("SELECT 1 FROM cluster_runs WHERE run_id=?",
                           (run_id,)).fetchone() is not None
        with conn:
            conn.execute("DELETE FROM cluster_assignments WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM cluster_runs WHERE run_id=?", (run_id,))
    finally:
        conn.close()
    if remove_dir:
        import shutil
        shutil.rmtree(os.path.join(out_dir, "cluster_runs", run_id),
                      ignore_errors=True)
    return had


def prune_runs(out_dir: str) -> list:
    """Drop db rows for runs whose cluster_runs/{run_id}/ dir no longer exists
    (e.g. deleted by hand). Returns the list of pruned run_ids. Dir-less runs in
    cluster_assignments-only (no cluster_runs row) are caught too."""
    rundir = os.path.join(out_dir, "cluster_runs")
    conn = connect(out_dir)
    try:
        ids = {r[0] for r in conn.execute("SELECT run_id FROM cluster_runs")}
        ids |= {r[0] for r in conn.execute(
            "SELECT DISTINCT run_id FROM cluster_assignments")}
    finally:
        conn.close()
    orphans = sorted(rid for rid in ids
                     if not os.path.isdir(os.path.join(rundir, rid)))
    for rid in orphans:
        drop_run(out_dir, rid, remove_dir=False)
    return orphans


def new_run_id(channel: str, params: dict, label: str = "") -> tuple:
    """(run_id, created_at). run_id = '{YYYYMMDD-HHMMSS}[__{label}]__{paramhash8}'
    -- readable (timestamp + a data/method label like 'apex__ghd1108_0501-left__k17')
    AND reproducible (params hash, which includes the data scope)."""
    created_at = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    h = hashlib.sha1(
        json.dumps({"channel": channel, **params}, sort_keys=True,
                   default=str).encode()).hexdigest()[:8]
    safe = re.sub(r"[^A-Za-z0-9._+-]", "", label).strip("_")
    return (f"{created_at}__{safe}__{h}" if safe else f"{created_at}__{h}"), created_at


def save_run(out_dir: str, run_id: str, channel: str, params: dict,
             clusters_df: pd.DataFrame, created_at: str) -> str:
    """Persist one clustering run: write cluster_runs/{run_id}/params.json +
    clusters.csv AND register it in the db (cluster_runs + cluster_assignments).
    `clusters_df` must have columns source_file, clip_id, cluster_id. Returns the
    run directory (gallery rendering is the caller's job -> rundir/gallery/)."""
    rundir = os.path.join(out_dir, "cluster_runs", run_id)
    os.makedirs(rundir, exist_ok=True)
    with open(os.path.join(rundir, "params.json"), "w") as f:
        json.dump({"run_id": run_id, "channel": channel,
                   "created_at": created_at, "params": params},
                  f, indent=2, default=str)
    cdf = clusters_df[["source_file", "clip_id", "cluster_id"]].copy()
    cdf.to_csv(os.path.join(rundir, "clusters.csv"), index=False)
    write_cluster_run(out_dir, run_id, channel, params, cdf, created_at)
    return rundir


def latest_run(conn, channel=None):
    q = "SELECT run_id FROM cluster_runs"
    args = []
    if channel:
        q += " WHERE channel=?"; args.append(channel)
    q += " ORDER BY created_at DESC, run_id DESC LIMIT 1"
    row = conn.execute(q, args).fetchone()
    return row[0] if row else None


def clusters_with_labels(conn, run_id: str) -> pd.DataFrame:
    """Cluster assignments JOIN clip-scope labels -> the table label-driven
    evaluate.py scores. gesture_label is NULL/'' for unlabelled clips."""
    return pd.read_sql_query(
        "SELECT ca.source_file, ca.clip_id, ca.cluster_id, "
        "       a.value AS gesture_label, c.subject "
        "FROM cluster_assignments ca "
        "LEFT JOIN annotations a ON a.source_file=ca.source_file "
        "  AND a.clip_id=ca.clip_id AND a.scope='clip' AND a.kind='label' "
        "LEFT JOIN clips c ON c.source_file=ca.source_file AND c.clip_id=ca.clip_id "
        "WHERE ca.run_id=?", conn, params=(run_id,))


# ---------- re-segmentation: remap labels onto new clips --------------------

def remap_annotations(conn, current_seg_version: str,
                      min_overlap: float = 0.5) -> dict:
    """After a re-segmentation, carry clip-scope annotations whose seg_version is
    stale onto the new clips by maximum sample-interval overlap (IoU-style:
    intersection / union of [clip_start_sample, clip_end_sample]). Below
    `min_overlap` the annotation is kept but flagged note='待复核' (do NOT
    silently drop). Returns counts {remapped, flagged, unchanged}."""
    clips = pd.read_sql_query(
        "SELECT source_file, clip_id, start_sample, end_sample FROM clips", conn)
    by_sf = {sf: g for sf, g in clips.groupby("source_file")} if not clips.empty else {}
    remapped = flagged = unchanged = 0
    stale = [dict(r) for r in conn.execute(
        "SELECT * FROM annotations WHERE scope='clip' "
        "AND (seg_version IS NULL OR seg_version!=?)", (current_seg_version,))]
    with conn:
        for a in stale:
            g = by_sf.get(a["source_file"])
            a0, a1 = a["clip_start_sample"], a["clip_end_sample"]
            if g is None or a0 is None or a1 is None:
                conn.execute(
                    "UPDATE annotations SET note=? WHERE source_file=? AND "
                    "clip_id=? AND scope=? AND kind=?",
                    ((a["note"] or "") + " 待复核(无新clip)", a["source_file"],
                     a["clip_id"], a["scope"], a["kind"]))
                flagged += 1
                continue
            inter = (g["end_sample"].clip(upper=a1) - g["start_sample"].clip(lower=a0)).clip(lower=0)
            union = (g[["end_sample"]].max(axis=1).clip(lower=a1)
                     - g[["start_sample"]].min(axis=1).clip(upper=a0))
            iou = (inter / union.replace(0, 1)).to_numpy()
            j = int(iou.argmax()) if len(iou) else -1
            if j >= 0 and iou[j] >= min_overlap:
                new = g.iloc[j]
                new_cid = int(new["clip_id"])
                if new_cid != int(a["clip_id"]):
                    # move row to the new clip_id (delete old PK, insert new)
                    conn.execute(
                        "DELETE FROM annotations WHERE source_file=? AND clip_id=? "
                        "AND scope=? AND kind=?",
                        (a["source_file"], a["clip_id"], a["scope"], a["kind"]))
                conn.execute(
                    f"INSERT OR REPLACE INTO annotations ({_qcols(ANNOT_COLS)}) "
                    f"VALUES ({','.join(['?'] * len(ANNOT_COLS))})",
                    (a["source_file"], new_cid, a["scope"], a["kind"], a["value"],
                     int(new["start_sample"]), int(new["end_sample"]),
                     current_seg_version, a["labeler"], a["labeled_at"], a["note"]))
                remapped += 1
            else:
                conn.execute(
                    "UPDATE annotations SET note=? WHERE source_file=? AND "
                    "clip_id=? AND scope=? AND kind=?",
                    ((a["note"] or "") + " 待复核(重叠不足)", a["source_file"],
                     a["clip_id"], a["scope"], a["kind"]))
                flagged += 1
    return {"remapped": remapped, "flagged": flagged, "unchanged": unchanged}
