from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import os

import numpy as np

from emg_label import (action_segmentation, features, io_utils, plotting,
                       pose_segmentation, qc, segmentation, store)
from emg_label.config import Config


# Shard CSV schemas are the single source of truth in emg_label.store (the db
# tables mirror them). bursts.csv = EMG activations (formerly the mis-named
# segments.csv); clips.csv = pose action segments = the labelling/clustering unit.
BURST_FIELDS = store.BURST_COLS   # source_file, burst_idx, group, start/end, ...
CLIP_FIELDS = store.CLIP_COLS     # source_file, clip_id, start/end, hold_*, ...
REC_FIELDS = store.REC_COLS       # ... + seg_version


def _seg_version(cfg: Config) -> str:
    """Short deterministic hash of the segmentation-relevant Config params, so a
    label captured under one parameterization can be detected as stale and
    remapped after a re-segmentation (see store.remap_annotations)."""
    keys = ("fs", "smooth_ms", "min_action_s", "min_rest_gap_s", "emg_enter_k",
            "emg_exit_k", "pose_smooth_ms", "pose_pct", "pose_mad",
            "min_static_s", "min_motion_s", "move_enter", "move_exit",
            "settle_frac", "auto_move_thresh", "move_exit_frac", "enable_r2",
            "enable_r7", "max_hold_s", "snap_window_s", "nan_max_gap_s",
            "min_finite_frac", "pose_min_range", "pose_long_seg_s")
    blob = json.dumps({k: getattr(cfg, k) for k in keys}, sort_keys=True,
                      default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


# ---------- atomic write helpers --------------------------------------------

def _atomic_write_csv(path: str, fields: list, rows: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _atomic_savez(path: str, **arrays) -> None:
    # numpy appends .npz when missing; force the tmp name to already end in .npz
    # so we control the final path explicitly.
    tmp = path + "._tmp_.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_plot_overview(out_path: str, **plot_kwargs) -> None:
    tmp = out_path + ".tmp.png"
    plotting.plot_overview_dual(out_path=tmp, **plot_kwargs)
    os.replace(tmp, out_path)


# ---------- per-window QC ---------------------------------------------------

def _qc(window, emg, env, ja, pose_speed_sig, fs):
    """QC dict for an arbitrary [s, e) window over the source signals."""
    s, e = int(window[0]), int(window[1])
    s, e = max(0, s), min(len(emg), e)
    if e <= s:
        return {"duration_s": 0.0, "emg_rms": 0.0, "envelope_peak": 0.0,
                "mean_pose_speed": 0.0, "max_pose_speed": 0.0,
                "pose_range": 0.0}
    em = emg[s:e]
    return {
        "duration_s": round((e - s) / fs, 4),
        "emg_rms": float(np.sqrt(np.mean(np.square(em)))),
        "envelope_peak": float(env[s:e].max()),
        "mean_pose_speed": float(pose_speed_sig[s:e].mean()),
        "max_pose_speed": float(pose_speed_sig[s:e].max()),
        "pose_range": float(np.linalg.norm(
            ja[s:e].max(axis=0) - ja[s:e].min(axis=0))),
    }


# ---------- per-recording processing ----------------------------------------

def _shard_paths(out_dir: str, stem: str) -> dict:
    shard_dir = os.path.join(out_dir, "shards", stem)
    return {
        "dir": shard_dir,
        "overview": os.path.join(shard_dir, "overview.png"),
        "bursts": os.path.join(shard_dir, "bursts.csv"),
        "clips": os.path.join(shard_dir, "clips.csv"),
        "recording": os.path.join(shard_dir, "recording.csv"),
        "features": os.path.join(out_dir, "features", stem + ".npz"),
    }


def shard_complete(out_dir: str, stem: str) -> bool:
    """recording.csv is the commit point -- exists ⇒ all sibling shard files
    were written before it (Stage 1's process_one writes it last)."""
    return os.path.isfile(_shard_paths(out_dir, stem)["recording"])


def process_one(path: str, info, cfg: Config,
                write_overview: bool = True) -> bool:
    """Process one recording into a shard under out/shards/{stem}/ + features
    cache under out/features/{stem}.npz.

    Returns True on success, False on skip (missing file / load error).
    """
    paths = _shard_paths(cfg.out_dir, info.stem)
    os.makedirs(paths["dir"], exist_ok=True)
    os.makedirs(os.path.dirname(paths["features"]), exist_ok=True)

    if not os.path.isfile(path):
        print(f"SKIP missing file: {path}")
        return False
    try:
        emg, ja = io_utils.load_npz(path, hand=info.hand)
    except Exception as ex:
        # Wide net: KeyError/ValueError (non-sample npz like norm_stats.npz,
        # bad shape), UnpicklingError + BadZipFile + EOFError (truncated or
        # zero-filled file -- seen on /mnt/pose_data over flaky NFS), OSError
        # (transient FS errors). One bad recording must not abort a parallel
        # run; log the full path so the user can inspect/delete after.
        print(f"SKIP {info.stem} ({type(ex).__name__}: {ex}) path={path}")
        return False
    n = len(emg)

    # --- Unified pose+EMG action segmentation ------------------------------
    # The protocol is move -> hold -> move -> hold with NO neutral return: each
    # 静止/hold is the just-formed gesture pose itself. pose-speed (the spine)
    # draws every onset and where the hand settles; EMG re-splits over-long
    # never-settling runs (R2). One action = a motion run + its following REAL
    # stable hold, right boundary at the next onset (segments tile seamlessly,
    # no artificial pad). move_enter/move_exit are adaptive per-recording
    # (robust_threshold valley) -- pose-speed scale varies ~4x across recordings
    # so a fixed rad/s does not work. Diagnostic-verified on jm-0503/ax-0819:
    # ~1 segment per natural gesture, real hold median 0.6-1.0s, apex in hold
    # >90%. See docs/检测切分统一设计.md + emg_label/action_segmentation.py.
    actions, dbg = action_segmentation.segment_recording(emg, ja, cfg)
    segs, env = dbg["bursts"], dbg["env"]
    enter, exit_thr = dbg["enter"], dbg["exit_e"]
    spd, pose_thr, pose_exit_thr = dbg["spd"], dbg["move_enter"], dbg["move_exit"]
    rec_pose_range, pose_static = dbg["rec_pose_range"], bool(dbg["pose_static"])

    # --- EMG burst path (cross-reference + Stage-2 features cache only) ------
    hold_ends = segmentation.hold_windows(segs, n, cfg.fs)
    rest = np.nanmedian(ja, axis=0)
    burst_apexes = [features.apex_index(ja, s, he, rest, cfg.fs)
                    for (s, _), he in zip(segs, hold_ends)]

    clips, clip_apexes = [], []
    for cid, a in enumerate(actions):
        clips.append({
            "clip_id": cid,
            "clip_start": a["start"], "clip_end": a["end"],
            # static_in zero-width (no pre-motion pad); static_out IS the real
            # hold (aliased as hold_*), restoring the original SMC semantics.
            "static_in_start": a["start"], "static_in_end": a["start"],
            "motion_start": a["motion_start"], "motion_end": a["motion_end"],
            "static_out_start": a["hold_start"], "static_out_end": a["hold_end"],
            "hold_start": a["hold_start"], "hold_end": a["hold_end"],
            "fusion_type": a["fusion_type"],
            "review_flag": a["review_flag"],
        })
        clip_apexes.append(a["apex"])

    # --- Cross-reference ----------------------------------------------------
    burst_to_clip = pose_segmentation.match_bursts_to_clips(segs, clips)
    clip_to_burst = pose_segmentation.match_clips_to_bursts(clips, segs)

    # --- Build CSV rows -----------------------------------------------------
    seg_version = _seg_version(cfg)
    burst_rows = []
    for burst_idx, ((s, e), he, ap, mc) in enumerate(
            zip(segs, hold_ends, burst_apexes, burst_to_clip)):
        m = _qc((s, e), emg, env, ja, spd, cfg.fs)
        burst_rows.append({
            "source_file": info.stem + ".npz",
            "group": info.group,
            "burst_idx": burst_idx,
            "start_sample": s,
            "end_sample": e,
            "hold_end_sample": he,
            "apex_sample": ap,
            "duration_s": m["duration_s"],
            "emg_rms": round(m["emg_rms"], 4),
            "envelope_peak": round(m["envelope_peak"], 4),
            "pose_range": round(m["pose_range"], 4),
            "matched_clip_id": mc,
        })

    clip_rows = []
    for c, ap, mb in zip(clips, clip_apexes, clip_to_burst):
        m_clip = _qc((c["clip_start"], c["clip_end"]),
                     emg, env, ja, spd, cfg.fs)
        m_motion = _qc((c["motion_start"], c["motion_end"]),
                       emg, env, ja, spd, cfg.fs)
        m_hold = _qc((c["hold_start"], c["hold_end"]),
                     emg, env, ja, spd, cfg.fs)
        clip_rows.append({
            "source_file": info.stem + ".npz",
            "group": info.group,
            "subject": info.subject or "",
            "hand": info.hand or "",
            "clip_id": c["clip_id"],
            "start_sample": c["clip_start"],
            "end_sample": c["clip_end"],
            "motion_start_sample": c["motion_start"],
            "motion_end_sample": c["motion_end"],
            "hold_start_sample": c["hold_start"],
            "hold_end_sample": c["hold_end"],
            "apex_sample": ap,
            "duration_s": m_clip["duration_s"],
            "motion_duration_s": m_motion["duration_s"],
            "hold_duration_s": m_hold["duration_s"],
            "emg_rms": round(m_clip["emg_rms"], 4),
            "envelope_peak": round(m_clip["envelope_peak"], 4),
            "mean_pose_speed": round(m_clip["mean_pose_speed"], 4),
            "max_pose_speed": round(m_clip["max_pose_speed"], 4),
            "pose_range": round(m_clip["pose_range"], 4),
            "matched_burst_idx": mb,
            "fusion_type": c["fusion_type"],
            "review_flag": c["review_flag"],
            "seg_version": seg_version,
        })

    # --- Recording-level QC -------------------------------------------------
    lag_s, lag_corr = qc.estimate_emg_pose_lag(env, spd, cfg.fs)
    lag_flag = qc.lag_status(lag_s)
    n_burst_only = sum(1 for x in burst_to_clip if x < 0)
    n_clip_only = sum(1 for x in clip_to_burst if x < 0)
    pose_nan_frac = float(np.mean(np.any(~np.isfinite(ja), axis=1)))
    emg_nan_frac = float(np.mean(np.any(~np.isfinite(emg), axis=1)))
    rec_row = {
        "source_file": info.stem + ".npz",
        "source_path": path,
        "group": info.group,
        "subject": info.subject or "",
        "hand": info.hand or "",
        "n_samples": n,
        "duration_s": round(n / cfg.fs, 4),
        "n_bursts": len(segs),
        "n_clips": len(clips),
        "n_burst_only": n_burst_only,
        "n_clip_only": n_clip_only,
        "emg_pose_lag_s": round(lag_s, 4) if np.isfinite(lag_s) else "",
        "emg_pose_corr": round(lag_corr, 4),
        "lag_flag": lag_flag,
        "pose_nan_frac": round(pose_nan_frac, 4),
        "emg_nan_frac": round(emg_nan_frac, 4),
        "enter_thresh": round(float(enter), 6),
        "exit_thresh": round(float(exit_thr), 6),
        "pose_thresh": round(float(pose_thr), 6),
        "pose_exit_thresh": round(float(pose_exit_thr), 6),
        "rec_pose_range": round(rec_pose_range, 4),
        "pose_static": int(pose_static),
        "seg_version": seg_version,
    }

    # --- Stage-2 features cache (apex pose vector, one per CLIP) -----------
    # The clustering unit is the clip, so cache the clip-apex feature keyed by
    # clip_id; cluster.py reads it without reloading the npz. (Computed from each
    # clip's already-located apex_sample within its hold.)
    if clips:
        feat_arr = np.stack([
            features.clip_apex_feature(ja, ap, cfg.fs) for ap in clip_apexes
        ]).astype(np.float32)
    else:
        feat_arr = np.empty((0, ja.shape[1]), dtype=np.float32)
    _atomic_savez(
        paths["features"],
        clip_id=np.arange(len(clips), dtype=np.int64),
        feature=feat_arr,
        rest=rest.astype(np.float32),
    )

    # --- Shard atomic writes (recording.csv LAST = commit point) -----------
    _atomic_write_csv(paths["bursts"], BURST_FIELDS, burst_rows)
    _atomic_write_csv(paths["clips"], CLIP_FIELDS, clip_rows)
    if write_overview:
        _atomic_plot_overview(
            paths["overview"],
            env=env, pose_speed=spd, fs=cfg.fs,
            burst_segments=segs, clips=clips,
            enter_thr=enter, exit_thr=exit_thr, pose_thr=pose_thr,
            pose_exit_thr=pose_exit_thr,
            apex_samples=clip_apexes,  # clip apex = each movement's formed-pose frame
        )
    _atomic_write_csv(paths["recording"], REC_FIELDS, [rec_row])

    lag_disp = f"{lag_s:+.3f}s" if np.isfinite(lag_s) else "n/a"
    nan_warn = (f" pose_nan={pose_nan_frac:.0%}"
                if pose_nan_frac > 0.01 else "")
    print(f"{info.stem}: {len(segs)} bursts, {len(clips)} clips, "
          f"{n_burst_only} burst-only, {n_clip_only} clip-only, "
          f"lag={lag_disp} (corr={lag_corr:.2f}, {lag_flag}){nan_warn}")
    return True


# ---------- top-level index assembly ---------------------------------------

def rebuild_index(out_dir: str) -> None:
    """Rebuild the SQLite index (OUT/index.db) from every shard, then dump the
    top-level convenience CSVs (recordings/bursts/clips.csv). Tombstoned
    recordings are skipped by store.build_index. The db is the authoritative
    index; the CSVs are a derived courtesy for flat-file dev tools."""
    counts = store.build_index(out_dir)
    store.export_csv(out_dir)
    print(f"Indexed {counts['recordings']} recordings, {counts['bursts']} "
          f"bursts, {counts['clips']} clips into {store.db_path(out_dir)}")
    print(f"Wrote convenience CSVs: {out_dir}/recordings.csv, bursts.csv, clips.csv")


# ---------- CLI -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Stage 1: dual segmentation (EMG burst + pose-speed clip). "
                    "Each recording lands as an independent shard under "
                    "out/shards/{stem}/ -- existing shards are skipped on rerun.")
    ap.add_argument("input_dir", nargs="?", default=None,
                    help="folder containing .npz files (optional if --meta given)")
    ap.add_argument("--meta", default=None,
                    help="sample_meta.csv-style catalogue; each row is one "
                         "recording. Mutually exclusive with input_dir scan.")
    ap.add_argument("--recursive", action="store_true",
                    help="recursive glob under input_dir")
    ap.add_argument("--allow-force", action="store_true",
                    help="include /mnt/force_data/ recordings.")
    ap.add_argument("--out", default="out")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--min-action-s", type=float, default=0.4)
    ap.add_argument("--min-rest-gap-s", type=float, default=0.2)
    ap.add_argument("--smooth-ms", type=float, default=150.0)
    ap.add_argument("--enter-thresh", type=float, default=None)
    ap.add_argument("--exit-thresh", type=float, default=None)
    ap.add_argument("--emg-enter-k", type=float, default=0.8,
                    help="enter threshold placement rest_center(0)->Otsu valley(1). "
                         "Lower to catch shorter/weaker bursts (1.0 = old behaviour).")
    ap.add_argument("--emg-exit-k", type=float, default=0.4,
                    help="hysteresis exit placement (must be < --emg-enter-k).")
    ap.add_argument("--pose-smooth-ms", type=float, default=250.0)
    ap.add_argument("--pose-pct", type=float, default=35.0)
    ap.add_argument("--pose-mad", type=float, default=1.5)
    ap.add_argument("--min-static-s", type=float, default=0.35)
    ap.add_argument("--min-motion-s", type=float, default=0.20)
    # --- unified pose-hysteresis + EMG-split detector (action_segmentation) ---
    # These are the knobs segment_recording actually consumes. pose-speed is the
    # spine; move-enter/move-exit are auto-derived per recording by default.
    ap.add_argument("--auto-move-thresh", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="derive move-enter/exit per recording from "
                         "robust_threshold(pose-speed) using --pose-pct/--pose-mad "
                         "(default). --no-auto-move-thresh uses fixed "
                         "--move-enter/--move-exit instead.")
    ap.add_argument("--move-enter", type=float, default=np.deg2rad(60.0),
                    help="STATIC->MOVING onset, rad/s (~60 deg/s). Only used with "
                         "--no-auto-move-thresh.")
    ap.add_argument("--move-exit", type=float, default=np.deg2rad(25.0),
                    help="MOVING->STATIC settle, rad/s (~25 deg/s). Only used with "
                         "--no-auto-move-thresh.")
    ap.add_argument("--move-exit-frac", type=float, default=0.5,
                    help="move_exit = frac * move_enter when --auto-move-thresh.")
    ap.add_argument("--settle-frac", type=float, default=0.25,
                    help="relative-settle: spd <= max(move_exit, frac*peak_run).")
    ap.add_argument("--enable-r2", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="split over-long never-settling runs at clean EMG-rest gaps.")
    ap.add_argument("--enable-r7", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="back-to-back hold veto. OFF by default (over-splits on "
                         "regular data); opt-in for genuinely under-cut recordings.")
    ap.add_argument("--max-hold-s", type=float, default=4.0,
                    help="hold longer than this gets review_flag=long_static.")
    ap.add_argument("--snap-window-s", type=float, default=0.15,
                    help="half-window for snapping onset/cut to the speed foot/min.")
    ap.add_argument("--nan-max-gap-s", type=float, default=0.20,
                    help="Manus pose dropouts shorter than this are linearly filled.")
    ap.add_argument("--min-finite-frac", type=float, default=0.5,
                    help="window with fewer finite rows than this fails the gate.")
    ap.add_argument("--pose-min-range", type=float, default=np.deg2rad(15.0),
                    help="absolute joint-excursion floor in RADIANS (default "
                         "~15 deg-equiv); segments/recordings below it are "
                         "dropped as static (no real gesture)")
    ap.add_argument("--pose-long-seg-s", type=float, default=2.5,
                    help="segments longer than this get review_flag=long")
    # --- incremental-mode flags ---
    ap.add_argument("--force", action="store_true",
                    help="reprocess recordings even if their shard exists "
                         "(out/shards/{stem}/recording.csv). Default skips.")
    ap.add_argument("--no-overview", action="store_true",
                    help="skip overview.png generation (CSV-only). Existing "
                         "PNGs are kept.")
    ap.add_argument("--index-only", action="store_true",
                    help="skip per-recording processing; just rebuild the "
                         "top-level segments/clips/recordings.csv from "
                         "existing shards.")
    ap.add_argument("--subjects", default=None,
                    help="comma-separated subject ids to keep. Matched on the "
                         "NORMALIZED id (hyphen/case-insensitive), so 'fgw-0917' "
                         "== 'fgw0917'. Works identically for --meta and dir "
                         "input -- the unified id-based selector.")
    ap.add_argument("--only-subject", default=None,
                    help="(alias for a single --subjects id; kept for back-compat)")
    ap.add_argument("--only-hand", choices=["left", "right"], default=None,
                    help="process only recordings with this hand.")
    ap.add_argument("--dates", default=None,
                    help="keep only recordings whose date (YYYYMMDD) starts with "
                         "any of these comma-separated prefixes: a full day "
                         "'20260502', a month '202605', or a year '2026'.")
    ap.add_argument("--date-from", default=None,
                    help="keep recordings on/after this date (YYYYMMDD, inclusive).")
    ap.add_argument("--date-to", default=None,
                    help="keep recordings on/before this date (YYYYMMDD, inclusive).")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes (one per recording). "
                         "Each recording is fully independent so this scales "
                         "near-linearly until disk IO saturates. Default 1 "
                         "(sequential).")
    args = ap.parse_args()

    cfg = Config(
        fs=args.fs, out_dir=args.out, min_action_s=args.min_action_s,
        min_rest_gap_s=args.min_rest_gap_s, smooth_ms=args.smooth_ms,
        enter_thresh=args.enter_thresh, exit_thresh=args.exit_thresh,
        emg_enter_k=args.emg_enter_k, emg_exit_k=args.emg_exit_k,
        pose_smooth_ms=args.pose_smooth_ms, pose_pct=args.pose_pct,
        pose_mad=args.pose_mad, min_static_s=args.min_static_s,
        min_motion_s=args.min_motion_s,
        pose_min_range=args.pose_min_range, pose_long_seg_s=args.pose_long_seg_s,
        auto_move_thresh=args.auto_move_thresh, move_enter=args.move_enter,
        move_exit=args.move_exit, move_exit_frac=args.move_exit_frac,
        settle_frac=args.settle_frac, enable_r2=args.enable_r2,
        enable_r7=args.enable_r7, max_hold_s=args.max_hold_s,
        snap_window_s=args.snap_window_s, nan_max_gap_s=args.nan_max_gap_s,
        min_finite_frac=args.min_finite_frac,
    )
    os.makedirs(os.path.join(cfg.out_dir, "shards"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "features"), exist_ok=True)

    if args.index_only:
        rebuild_index(cfg.out_dir)
        return

    if args.meta:
        work = [(info.path, info)
                for info in io_utils.parse_meta_csv(args.meta)]
    else:
        if not args.input_dir:
            ap.error("either input_dir or --meta is required")
        paths = io_utils.discover_npz(args.input_dir, recursive=args.recursive)
        work = [(p, io_utils.parse_file_info(p)) for p in paths]
    if not work:
        print(f"No .npz files found (input_dir={args.input_dir}, "
              f"meta={args.meta})")
        return

    if not args.allow_force:
        force = [w for w in work if "/force_data/" in w[0].lower()]
        if force:
            print(f"SKIP {len(force)} force_data file(s) "
                  f"(pass --allow-force to include them)")
            work = [w for w in work if "/force_data/" not in w[0].lower()]
        if not work:
            print("All input was force_data; nothing to do.")
            return

    sel = []
    if args.subjects:
        sel += [s for s in args.subjects.split(",") if s.strip()]
    if args.only_subject:
        sel.append(args.only_subject)
    if sel:
        want = {io_utils.normalize_subject(s) for s in sel}
        before = len(work)
        work = [(p, i) for p, i in work
                if io_utils.normalize_subject(i.subject) in want]
        print(f"--subjects {','.join(sel)}: kept {len(work)}/{before} recordings")
    if args.only_hand:
        work = [(p, i) for p, i in work if i.hand == args.only_hand]
    # --- date selection (by time), mirroring --subjects / --only-hand ---
    if args.dates:
        pref = tuple(s.strip() for s in args.dates.split(",") if s.strip())
        before = len(work)
        work = [(p, i) for p, i in work if i.date and i.date.startswith(pref)]
        print(f"--dates {args.dates}: kept {len(work)}/{before} recordings")
    if args.date_from or args.date_to:
        lo = args.date_from or "00000000"
        hi = args.date_to or "99999999"
        before = len(work)
        work = [(p, i) for p, i in work if i.date and lo <= i.date <= hi]
        print(f"--date-from/to [{lo},{hi}]: kept {len(work)}/{before} recordings")
    if not work:
        print("No work after --only-* / --dates filters.")
        return

    # Skip tombstoned recordings (permanently dropped via the label UI) so a
    # re-segmentation never resurrects them. Tombstone = hard "do not process".
    tomb = store.tombstoned(cfg.out_dir)
    if tomb:
        before = len(work)
        work = [(p, i) for p, i in work if (i.stem + ".npz") not in tomb]
        if before != len(work):
            print(f"SKIP {before - len(work)} tombstoned recording(s)")
        if not work:
            print("All work tombstoned; nothing to do.")
            return

    # Pre-filter cached shards so workers never spawn for already-done items
    # (each pool worker startup costs ~50-100ms; not worth paying for a no-op).
    todo = []
    n_cached = n_done = n_skip = 0
    for path, info in work:
        if not args.force and shard_complete(cfg.out_dir, info.stem):
            print(f"CACHED {info.stem}")
            n_cached += 1
        else:
            todo.append((path, info))

    write_overview = not args.no_overview
    if todo and args.workers > 1:
        # multiprocessing: process_one is fully self-contained -- it only
        # reads cfg/info/path and writes to shards/{stem}/ + features/{stem}.npz
        # -- so workers never contend for the same output paths. cfg is a
        # frozen-ish dataclass (picklable); FileInfo is too.
        from multiprocessing import Pool
        worker_fn = functools.partial(
            _worker, cfg=cfg, write_overview=write_overview)
        with Pool(args.workers) as pool:
            for ok in pool.imap_unordered(worker_fn, todo):
                if ok:
                    n_done += 1
                else:
                    n_skip += 1
    else:
        for path, info in todo:
            ok = process_one(path, info, cfg,
                             write_overview=write_overview)
            if ok:
                n_done += 1
            else:
                n_skip += 1

    print(f"\nshards: {n_done} new, {n_cached} cached, {n_skip} skipped "
          f"({n_done + n_cached + n_skip} total)")

    rebuild_index(cfg.out_dir)


def _worker(item, cfg, write_overview):
    """multiprocessing.Pool entry. Top-level so it pickles cleanly.

    Catches *any* unexpected exception so one rogue recording can't kill the
    whole pool (Pool.imap_unordered re-raises and shuts down on first miss).
    process_one already catches file-load errors; this is the belt-and-braces
    for anything else (matplotlib memory hiccup, numpy ufunc edge case).
    """
    # Cap intra-process BLAS threading when we're running many workers --
    # otherwise N workers x OMP_NUM_THREADS easily oversubscribes the box.
    # threadpoolctl is best-effort; numpy's already initialised when fork()s
    # inherit it, so env vars set here wouldn't take effect.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    path, info = item
    try:
        return process_one(path, info, cfg, write_overview=write_overview)
    except Exception as ex:
        import traceback
        print(f"WORKER ERROR {info.stem} ({type(ex).__name__}: {ex}) "
              f"path={path}\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()
