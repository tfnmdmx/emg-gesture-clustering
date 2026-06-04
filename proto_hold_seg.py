from __future__ import annotations

"""原型（只读）：hold-锚定的"清晰手型"切分 vs 现在的 movement 切分。

动机：用户只需要"手势规范、形状边界清晰的动作"。这种数据是 move->hold->move->hold，
hold(稳定持姿)就是清晰手型本身。与其切动态、易受速度抖动、含过渡的 movement，不如直接
锚定 hold：每个稳定持姿 = 一个清晰手势，按质量打分只留干净的，丢掉快速/不稳/模糊的。

对每条录制画 2 lane：
  lane1: pose-speed + detect线 + holds(蓝) + motion(橙)  —— 现状参照
  lane2: 提议的 hold-锚定手势段(绿=保留, 灰=被质量门丢弃)，每段 = [进入运动 + 持姿]，
         apex(持姿中偏离最大帧)标紫线
并打印 现movement数 vs 提议保留手势数。

质量门（deg 单位，从数据标定）：
  hold_dur >= MIN_HOLD_S         持姿够久（真的停住了，不是路过）
  joint_std <= MAX_STD_DEG       持姿够稳（手型不漂移 = 形状清晰）
  dev_from_rest >= MIN_DEV_DEG   手型成型（非中性、是个真手势）

Run:
  ENV=/home/chenglin/anaconda3/envs/emg2pose/bin/python
  $ENV proto_hold_seg.py --stems jm-0503__20260430-left-2__20260430_172323 --out out_peak_s1
  $ENV proto_hold_seg.py --stems <stem> --out out_peak_s1 --xmax 30   # 30s 放大
输出：out_holdseg_proto/{stem}.png
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from emg_label import io_utils, pose_segmentation as ps  # noqa: E402
from emg_label.config import Config  # noqa: E402


def hold_anchored_gestures(spd, ja, fs, detect, rest,
                           min_hold_s=1.0, max_hold_s=4.0,
                           max_std_deg=8.0, min_dev_deg=40.0,
                           min_static_s=0.30, merge_gap_s=0.20, pad_s=0.1):
    """Stable holds -> deliberate-gesture segments.

    A deliberate gesture = the hand truly settles and HOLDS a well-formed shape
    for [min_hold_s, max_hold_s]. Shorter holds are transitional pauses the hand
    passes through between gestures (dropped); longer ones are 长时间静止 abnormal
    rests (dropped). Returns list of dicts; dropped holds are kept (kept=False)
    with a reason so the diagnostic can grey them out.
    """
    static, motion = ps.static_motion_intervals(
        spd, fs, detect, min_static_s=min_static_s, min_motion_s=0.10,
        merge_gap_s=merge_gap_s)
    pad = int(round(pad_s * fs))
    n = len(spd)
    out = []
    for hs, he in static:
        w = ja[hs:he]
        dur = (he - hs) / fs
        std = float(np.mean(np.std(w, axis=0)))
        dev = float(np.linalg.norm(np.median(w, axis=0) - rest))
        kept = (min_hold_s <= dur <= max_hold_s
                and std <= max_std_deg and dev >= min_dev_deg)
        # approach = the motion run ending just before this hold (clip context)
        approach = [ms for ms, me in motion if me <= hs]
        astart = approach and max(approach, key=lambda x: x)  # last motion start
        # last motion run whose end is closest before hs:
        cand = [(ms, me) for ms, me in motion if me <= hs]
        clip_start = (max(0, cand[-1][0] - pad) if cand else max(0, hs - pad))
        # apex = max joint deviation-from-rest frame within the hold
        dev_t = np.linalg.norm(ja[hs:he] - rest, axis=1)
        apex = hs + int(np.argmax(dev_t)) if he > hs else hs
        out.append({"hold_start": hs, "hold_end": he,
                    "clip_start": clip_start, "clip_end": min(n, he + pad),
                    "apex": apex, "dur": dur, "std": std, "dev": dev,
                    "kept": kept})
    return out, static, motion


def process(stem, cfg, args):
    df = pd.read_csv(os.path.join(args.out, "clips.csv"), dtype={"source_file": str})
    rows = df[df["source_file"] == stem + ".npz"]
    if rows.empty:
        print(f"SKIP {stem}: not in clips.csv"); return
    src = str(rows.iloc[0]["source_path"]); hand = str(rows.iloc[0].get("hand") or "") or None
    emg, ja = io_utils.load_npz(src, hand=hand)
    n = len(emg); fs = cfg.fs
    spd = ps.pose_speed(ja, fs, cfg.pose_smooth_ms)
    detect = ps.robust_threshold(spd, cfg.pose_pct, cfg.pose_mad)
    rest = np.median(ja, axis=0)

    gestures, static, motion = hold_anchored_gestures(
        spd, ja, fs, detect, rest,
        min_hold_s=args.min_hold_s, max_hold_s=args.max_hold_s,
        max_std_deg=args.max_std, min_dev_deg=args.min_dev)
    kept = [g for g in gestures if g["kept"]]
    dropped = [g for g in gestures if not g["kept"]]

    print(f"\n=== {stem} ===")
    print(f"  current movement segs: {len(motion)}")
    print(f"  proposed hold-gestures: {len(kept)} kept, {len(dropped)} dropped "
          f"(of {len(static)} holds)")
    if dropped:
        why = {"short": sum(g['dur'] < args.min_hold_s for g in dropped),
               "too-long": sum(g['dur'] > args.max_hold_s for g in dropped),
               "unstable": sum(g['std'] > args.max_std for g in dropped),
               "low-dev": sum(g['dev'] < args.min_dev for g in dropped)}
        print(f"  dropped reasons: {why}")

    # readability cap: pose-speed peaks hit ~thousands deg/s; clip the y-axis to
    # the threshold band so holds and the threshold line stay legible.
    sp = np.nan_to_num(spd)
    ycap = max(float(np.percentile(sp[np.isfinite(sp)], 97)), detect * 2.0) * 1.1

    t = np.arange(n) / fs
    fig, ax = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
    ax[0].plot(t, np.nan_to_num(spd), lw=0.6, color="navy")
    ax[0].set_ylim(0, ycap)
    ax[0].axhline(detect, color="r", ls="--", lw=0.8, label="detect")
    for hs, he in static:
        ax[0].axvspan(hs / fs, he / fs, color="lightblue", alpha=0.5)
    for ms, me in motion:
        ax[0].axvspan(ms / fs, me / fs, color="orange", alpha=0.25)
    ax[0].legend(loc="upper right", fontsize=7)
    ax[0].set_ylabel(f"pose-speed\nholds(blue)={len(static)} motion(orange)={len(motion)}")

    ax[1].plot(t, np.nan_to_num(spd), lw=0.5, color="gray", alpha=0.6)
    ax[1].set_ylim(0, ycap)
    for g in kept:
        ax[1].axvspan(g["clip_start"] / fs, g["clip_end"] / fs, color="green", alpha=0.22)
        ax[1].axvspan(g["hold_start"] / fs, g["hold_end"] / fs, color="green", alpha=0.38)
        ax[1].axvline(g["apex"] / fs, color="purple", lw=0.6, alpha=0.7)
    for g in dropped:
        ax[1].axvspan(g["hold_start"] / fs, g["hold_end"] / fs, color="lightgray", alpha=0.7)
    ax[1].set_ylabel(f"PROPOSED\n{len(kept)} clean gestures")
    ax[1].set_xlabel("time (s)")
    ax[1].set_xlim(0, n / fs)
    ax[1].set_title(
        f"current {len(motion)} movement-segs  ->  proposed {len(kept)} clean "
        f"hold-gestures (+{len(dropped)} dropped: short/unstable/low-dev)",
        fontsize=9, loc="left")

    if args.xmax is not None:
        for a in ax:
            a.set_xlim(args.xmin, args.xmax)
    fig.suptitle(stem, fontsize=10)
    fig.tight_layout()
    os.makedirs(args.proto_out, exist_ok=True)
    suffix = f"_zoom{int(args.xmin)}-{int(args.xmax)}" if args.xmax is not None else ""
    out = os.path.join(args.proto_out, stem + suffix + ".png")
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out_peak_s1")
    ap.add_argument("--proto-out", default="out_holdseg_proto")
    ap.add_argument("--stems", required=True)
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--pose-smooth-ms", type=float, default=250.0)
    ap.add_argument("--min-hold-s", type=float, default=1.0)
    ap.add_argument("--max-hold-s", type=float, default=4.0)
    ap.add_argument("--max-std", type=float, default=8.0)
    ap.add_argument("--min-dev", type=float, default=40.0)
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=None)
    args = ap.parse_args()
    cfg = Config(fs=args.fs, pose_smooth_ms=args.pose_smooth_ms)
    for stem in args.stems.split(","):
        process(stem.strip(), cfg, args)


if __name__ == "__main__":
    main()
