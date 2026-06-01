from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import pandas as pd

from emg_label import (features, io_utils, plotting, pose_segmentation, qc,
                       segmentation)
from emg_label.config import Config


SEG_FIELDS = ["source_file", "source_path", "group", "seg_idx",
              "start_sample", "end_sample", "hold_end_sample",
              "apex_sample", "duration_s", "emg_rms", "envelope_peak",
              "pose_range", "matched_clip_id"]

CLIP_FIELDS = ["source_file", "source_path", "group", "subject", "hand",
               "clip_id",
               "clip_start_sample", "clip_end_sample",
               "static_in_start_sample", "static_in_end_sample",
               "motion_start_sample", "motion_end_sample",
               "static_out_start_sample", "static_out_end_sample",
               "apex_sample", "duration_s", "motion_duration_s",
               "emg_rms", "envelope_peak",
               "mean_pose_speed", "max_pose_speed", "pose_range",
               "matched_emg_seg_idx", "gesture_label"]

REC_FIELDS = ["source_file", "source_path", "group", "subject", "hand",
              "n_samples", "duration_s", "n_bursts", "n_clips",
              "n_burst_only", "n_clip_only",
              "emg_pose_lag_s", "emg_pose_corr", "lag_flag",
              "pose_nan_frac", "emg_nan_frac",
              "enter_thresh", "exit_thresh", "pose_thresh"]


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
        "segments": os.path.join(shard_dir, "segments.csv"),
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
    except (KeyError, ValueError) as ex:
        # Raw session dirs contain non-sample npz (norm_stats.npz etc.) and
        # broken recordings -- skip rather than abort a long run.
        print(f"SKIP {os.path.basename(path)}: {ex}")
        return False
    n = len(emg)

    # --- EMG burst path -----------------------------------------------------
    segs, env, enter, exit_thr = segmentation.segment_emg(emg, cfg)
    hold_ends = segmentation.hold_windows(segs, n, cfg.fs)
    rest = np.median(ja, axis=0)
    burst_apexes = [features.apex_index(ja, s, he, rest, cfg.fs)
                    for (s, _), he in zip(segs, hold_ends)]

    # --- Pose-speed clip path ----------------------------------------------
    spd = pose_segmentation.pose_speed(ja, cfg.fs, cfg.pose_smooth_ms)
    pose_thr = pose_segmentation.robust_threshold(
        spd, cfg.pose_pct, cfg.pose_mad)
    static, motion = pose_segmentation.static_motion_intervals(
        spd, cfg.fs, pose_thr,
        min_static_s=cfg.min_static_s,
        min_motion_s=cfg.min_motion_s,
        merge_gap_s=cfg.merge_gap_s,
    )
    clips = pose_segmentation.build_smc_clips(
        static, motion, cfg.fs, n,
        pre_static_s=cfg.pre_static_s,
        post_static_s=cfg.post_static_s,
        pad_s=cfg.pad_s,
        min_motion_s=cfg.min_motion_s,
    )
    clip_apexes = []
    for c in clips:
        so_s, so_e = c["static_out_start"], c["static_out_end"]
        if so_e > so_s:
            clip_apexes.append(features.apex_index(
                ja, so_s, so_e, rest, cfg.fs))
        else:
            clip_apexes.append((so_s + so_e) // 2)

    # --- Cross-reference ----------------------------------------------------
    burst_to_clip = pose_segmentation.match_bursts_to_clips(segs, clips)
    clip_to_burst = pose_segmentation.match_clips_to_bursts(clips, segs)

    # --- Build CSV rows -----------------------------------------------------
    seg_rows = []
    for seg_idx, ((s, e), he, ap, mc) in enumerate(
            zip(segs, hold_ends, burst_apexes, burst_to_clip)):
        m = _qc((s, e), emg, env, ja, spd, cfg.fs)
        seg_rows.append({
            "source_file": info.stem + ".npz",
            "source_path": path,
            "group": info.group,
            "seg_idx": seg_idx,
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
        clip_rows.append({
            "source_file": info.stem + ".npz",
            "source_path": path,
            "group": info.group,
            "subject": info.subject or "",
            "hand": info.hand or "",
            "clip_id": c["clip_id"],
            "clip_start_sample": c["clip_start"],
            "clip_end_sample": c["clip_end"],
            "static_in_start_sample": c["static_in_start"],
            "static_in_end_sample": c["static_in_end"],
            "motion_start_sample": c["motion_start"],
            "motion_end_sample": c["motion_end"],
            "static_out_start_sample": c["static_out_start"],
            "static_out_end_sample": c["static_out_end"],
            "apex_sample": ap,
            "duration_s": m_clip["duration_s"],
            "motion_duration_s": m_motion["duration_s"],
            "emg_rms": round(m_clip["emg_rms"], 4),
            "envelope_peak": round(m_clip["envelope_peak"], 4),
            "mean_pose_speed": round(m_clip["mean_pose_speed"], 4),
            "max_pose_speed": round(m_clip["max_pose_speed"], 4),
            "pose_range": round(m_clip["pose_range"], 4),
            "matched_emg_seg_idx": mb,
            "gesture_label": "",
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
    }

    # --- Stage-2 features cache (apex pose vectors, one per burst) ---------
    # Computed here so cluster.py never has to reload the npz to redo this
    # deterministic transform.
    if segs:
        feat_arr = np.stack([
            features.apex_pose_feature(ja, s, he, rest, cfg.fs)
            for (s, _), he in zip(segs, hold_ends)
        ]).astype(np.float32)
    else:
        feat_arr = np.empty((0, ja.shape[1]), dtype=np.float32)
    _atomic_savez(
        paths["features"],
        seg_idx=np.arange(len(segs), dtype=np.int64),
        feature=feat_arr,
        rest=rest.astype(np.float32),
    )

    # --- Shard atomic writes (recording.csv LAST = commit point) -----------
    _atomic_write_csv(paths["segments"], SEG_FIELDS, seg_rows)
    _atomic_write_csv(paths["clips"], CLIP_FIELDS, clip_rows)
    if write_overview:
        _atomic_plot_overview(
            paths["overview"],
            emg=emg, env=env, pose_speed=spd, fs=cfg.fs,
            burst_segments=segs, clips=clips,
            enter_thr=enter, exit_thr=exit_thr, pose_thr=pose_thr,
            apex_samples=burst_apexes,
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
    """Concat all shards/*/{segments,clips,recording}.csv into the three
    top-level CSVs that the rest of the pipeline reads."""
    mapping = [
        ("segments.csv", "segments.csv", SEG_FIELDS,
         ["source_file", "seg_idx"]),
        ("clips.csv", "clips.csv", CLIP_FIELDS,
         ["source_file", "clip_id"]),
        ("recording.csv", "recordings.csv", REC_FIELDS,
         ["source_file"]),
    ]
    n_rec = 0
    for shard_name, top_name, fields, sort_by in mapping:
        paths = sorted(glob.glob(
            os.path.join(out_dir, "shards", "*", shard_name)))
        if paths:
            dfs = [pd.read_csv(p, dtype={"source_file": str}) for p in paths]
            df = pd.concat(dfs, ignore_index=True)
            df = df.sort_values(sort_by, kind="stable").reset_index(drop=True)
            df = df[fields]
        else:
            df = pd.DataFrame(columns=fields)
        df.to_csv(os.path.join(out_dir, top_name), index=False)
        if top_name == "recordings.csv":
            n_rec = len(df)
            n_lag_ok = int((df.get("lag_flag", pd.Series(dtype=str))
                            == "ok").sum())
    print(f"Wrote {os.path.join(out_dir, 'segments.csv')}")
    print(f"Wrote {os.path.join(out_dir, 'clips.csv')}")
    print(f"Wrote {os.path.join(out_dir, 'recordings.csv')} "
          f"({n_rec} recordings, {n_lag_ok}/{n_rec} ok lag)")


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
    ap.add_argument("--pose-smooth-ms", type=float, default=250.0)
    ap.add_argument("--pose-pct", type=float, default=35.0)
    ap.add_argument("--pose-mad", type=float, default=1.5)
    ap.add_argument("--min-static-s", type=float, default=0.35)
    ap.add_argument("--min-motion-s", type=float, default=0.20)
    ap.add_argument("--merge-gap-s", type=float, default=0.20)
    ap.add_argument("--pre-static-s", type=float, default=0.5)
    ap.add_argument("--post-static-s", type=float, default=0.5)
    ap.add_argument("--pad-s", type=float, default=0.05)
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
    ap.add_argument("--only-subject", default=None,
                    help="process only recordings with this subject "
                         "(local rerun).")
    ap.add_argument("--only-hand", choices=["left", "right"], default=None,
                    help="process only recordings with this hand.")
    args = ap.parse_args()

    cfg = Config(
        fs=args.fs, out_dir=args.out, min_action_s=args.min_action_s,
        min_rest_gap_s=args.min_rest_gap_s, smooth_ms=args.smooth_ms,
        enter_thresh=args.enter_thresh, exit_thresh=args.exit_thresh,
        pose_smooth_ms=args.pose_smooth_ms, pose_pct=args.pose_pct,
        pose_mad=args.pose_mad, min_static_s=args.min_static_s,
        min_motion_s=args.min_motion_s, merge_gap_s=args.merge_gap_s,
        pre_static_s=args.pre_static_s, post_static_s=args.post_static_s,
        pad_s=args.pad_s,
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

    if args.only_subject:
        work = [(p, i) for p, i in work if i.subject == args.only_subject]
    if args.only_hand:
        work = [(p, i) for p, i in work if i.hand == args.only_hand]
    if not work:
        print("No work after --only-* filters.")
        return

    n_cached = n_done = n_skip = 0
    for path, info in work:
        # Shard-level skip: recording.csv is the commit-point sentinel from a
        # prior successful run. --force bypasses it.
        if not args.force and shard_complete(cfg.out_dir, info.stem):
            print(f"CACHED {info.stem}")
            n_cached += 1
            continue
        ok = process_one(path, info, cfg,
                         write_overview=not args.no_overview)
        if ok:
            n_done += 1
        else:
            n_skip += 1
    print(f"\nshards: {n_done} new, {n_cached} cached, {n_skip} skipped "
          f"({n_done + n_cached + n_skip} total)")

    rebuild_index(cfg.out_dir)


if __name__ == "__main__":
    main()
