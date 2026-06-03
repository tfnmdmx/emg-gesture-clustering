from __future__ import annotations

"""S0 诊断（只读，不改流水线）：速度峰切分 vs 持姿感知(hold-to-hold)切分对照。

回答路线图 S0 的问题：当前速度峰段是否把"一个自然手势"过切成多段？
对每条录制画 4 条 lane 并量化：
  1) EMG 包络 + burst（一次肌肉激活 ≈ 一次手势尝试）
  2) pose-speed + 当前速度峰段（橙）+ detect/bound 阈值线
  3) hold-to-hold 候选：static_motion_intervals 的 motion 段（绿）= 两次持姿之间的运动
  4) 量化：每个 EMG burst / 每个 hold-to-hold 段被几个速度峰段覆盖（过切因子）

用法：
  ENV=/home/chenglin/anaconda3/envs/emg2pose/bin/python
  $ENV diag_seg.py --out out_peak --stems jm-0503__20260430-left-2__20260430_171403
  $ENV diag_seg.py --out out_pose2 --stems hzy-1217__20260503-left-4__20260503_153603
输出：out_diag_s0/{stem}.png + 终端量化表
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from emg_label import io_utils, pose_segmentation, segmentation  # noqa: E402
from emg_label.config import Config  # noqa: E402


def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _split_factor(units, segs):
    """For each unit (s,e), how many segs overlap it. Returns Counter of counts."""
    counts = []
    for us, ue in units:
        counts.append(sum(1 for ss, se in segs if _overlap(us, ue, ss, se) > 0))
    return Counter(counts)


def process(stem, cfg, args):
    src_csv = os.path.join(args.out, "clips.csv")
    df = pd.read_csv(src_csv, dtype={"source_file": str})
    rows = df[df["source_file"] == stem + ".npz"]
    if rows.empty:
        print(f"SKIP {stem}: not in {src_csv}")
        return
    src = str(rows.iloc[0]["source_path"])
    hand = str(rows.iloc[0].get("hand") or "") or None
    if not os.path.isfile(src):
        print(f"SKIP {stem}: npz missing {src}")
        return
    emg, ja = io_utils.load_npz(src, hand=hand)
    n = len(emg)
    fs = cfg.fs

    # --- EMG bursts ---
    env = segmentation.emg_envelope(emg, fs, cfg.smooth_ms)
    bursts, _, enter, _ = segmentation.segment_emg(emg, cfg)

    # --- pose-speed + current velocity-peak segments (with range gate) ---
    spd = pose_segmentation.pose_speed(ja, fs, cfg.pose_smooth_ms)
    detect = pose_segmentation.robust_threshold(spd, cfg.pose_pct, cfg.pose_mad)
    bound = cfg.pose_bound_frac * detect
    vsegs, _ = pose_segmentation.velocity_peak_segments(
        spd, fs, detect, bound_frac=cfg.pose_bound_frac, prom_k=cfg.pose_prom_k,
        min_gesture_s=cfg.pose_min_gesture_s, merge_gap_s=cfg.pose_peak_merge_gap_s)

    def seg_range(a, b):
        w = ja[a:b]
        return float(np.linalg.norm(np.nanmax(w, 0) - np.nanmin(w, 0))) if b > a else 0.0
    vsegs = [(a, b) for a, b in vsegs if seg_range(a, b) >= cfg.pose_min_range]

    # --- hold-to-hold candidate: static_motion_intervals motion segments ---
    static, motion = pose_segmentation.static_motion_intervals(
        spd, fs, detect, min_static_s=cfg.min_static_s,
        min_motion_s=cfg.min_motion_s, merge_gap_s=cfg.merge_gap_s)

    # --- quantify over-segmentation ---
    sf_burst = _split_factor(bursts, vsegs)
    sf_hold = _split_factor(motion, vsegs)

    def fmt(counter):
        tot = sum(counter.values()) or 1
        return ", ".join(f"{k}->{v}({v/tot:.0%})"
                         for k, v in sorted(counter.items()))

    print(f"\n=== {stem} ===")
    print(f"  duration {n/fs:.1f}s | EMG bursts {len(bursts)} | "
          f"velocity-peak segs {len(vsegs)} | hold-to-hold motion {len(motion)}")
    print(f"  vpeak/burst ratio = {len(vsegs)/max(1,len(bursts)):.2f} | "
          f"vpeak/hold ratio = {len(vsegs)/max(1,len(motion)):.2f}")
    print(f"  per-EMG-burst  -> #vpeak segs overlapping: {fmt(sf_burst)}")
    print(f"  per-hold2hold  -> #vpeak segs overlapping: {fmt(sf_hold)}")

    # --- plot ---
    t = np.arange(n) / fs
    fig, ax = plt.subplots(4, 1, figsize=(18, 10), sharex=True)

    ax[0].plot(t, env, lw=0.6, color="k")
    ax[0].axhline(enter, color="r", ls="--", lw=0.7)
    for bs, be in bursts:
        ax[0].axvspan(bs / fs, be / fs, color="green", alpha=0.15)
    ax[0].set_ylabel(f"EMG env\n{len(bursts)} bursts")

    ax[1].plot(t, np.nan_to_num(spd), lw=0.6, color="navy")
    ax[1].axhline(detect, color="r", ls="--", lw=0.8, label="detect")
    ax[1].axhline(bound, color="gray", ls=":", lw=0.8, label="bound")
    for a, b in vsegs:
        ax[1].axvspan(a / fs, b / fs, color="orange", alpha=0.25)
    ax[1].legend(loc="upper right", fontsize=7)
    ax[1].set_ylabel(f"pose-speed\n{len(vsegs)} vpeak segs")

    ax[2].plot(t, np.nan_to_num(spd), lw=0.6, color="navy")
    ax[2].axhline(detect, color="r", ls="--", lw=0.8)
    for a, b in static:
        ax[2].axvspan(a / fs, b / fs, color="lightblue", alpha=0.35)
    for a, b in motion:
        ax[2].axvspan(a / fs, b / fs, color="green", alpha=0.30)
    ax[2].set_ylabel(f"hold-to-hold\n{len(motion)} motion / {len(static)} hold")

    # comparison lane: bursts (top) vs vpeak (mid) vs hold2hold (bottom)
    ax[3].set_ylim(0, 1)
    ax[3].set_yticks([0.83, 0.5, 0.17])
    ax[3].set_yticklabels(["EMG\nburst", "vpeak", "hold2hold"])
    for bs, be in bursts:
        ax[3].axvspan(bs / fs, be / fs, ymin=0.70, ymax=0.96, color="green", alpha=0.4)
    for a, b in vsegs:
        ax[3].axvspan(a / fs, b / fs, ymin=0.37, ymax=0.63, color="orange", alpha=0.5)
    for a, b in motion:
        ax[3].axvspan(a / fs, b / fs, ymin=0.04, ymax=0.30, color="green", alpha=0.4)
    ax[3].set_xlabel("time (s)")
    ax[3].set_xlim(0, n / fs)
    ax[3].set_title(
        f"vpeak/burst={len(vsegs)/max(1,len(bursts)):.2f}  "
        f"vpeak/hold={len(vsegs)/max(1,len(motion)):.2f}  "
        f"| per-burst split: {fmt(sf_burst)}", fontsize=9, loc="left")

    if args.xmax is not None:
        for a in ax:
            a.set_xlim(args.xmin, args.xmax)
    fig.suptitle(stem, fontsize=11)
    fig.tight_layout()
    os.makedirs(args.proto_out, exist_ok=True)
    suffix = f"_zoom{int(args.xmin)}-{int(args.xmax)}" if args.xmax is not None else ""
    out = os.path.join(args.proto_out, stem + suffix + ".png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out_peak", help="dir with clips.csv")
    ap.add_argument("--proto-out", default="out_diag_s0")
    ap.add_argument("--stems", required=True, help="comma-separated stems")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--pose-smooth-ms", type=float, default=250.0)
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=None, help="zoom end (s)")
    args = ap.parse_args()
    cfg = Config(fs=args.fs, pose_smooth_ms=args.pose_smooth_ms)
    for stem in args.stems.split(","):
        process(stem.strip(), cfg, args)


if __name__ == "__main__":
    main()
