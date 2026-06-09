from __future__ import annotations

"""切分诊断（只读，不改流水线）：旧 velocity-peak / 旧 hold-to-hold / 新统一切分 三方对照。

回答两个问题：
  过切：旧 velocity-peak 是否把"一个自然手势"切成多段？（常规数据 jm-0503）
  欠切：旧 hold-to-hold(static_motion) 是否把"连续手势"并成一段？（连续数据 hzy-1217）
  新统一切分 (emg_label.action_segmentation.segment_recording) 是否两头都对齐 EMG burst？

每条录制画 4 lane：
  1) EMG 包络 + burst（一次肌肉激活 ≈ 一次手势）
  2) pose-speed + 旧 velocity-peak 段（橙）+ detect 线          —— 过切参照
  3) pose-speed + 新统一段：motion(绿) + 真实 hold(蓝) + onset(红) + apex(紫)
       + move_enter/move_exit 线                                —— 提议
  4) 对照条：EMG burst / vpeak(旧) / hold2hold(旧) / NEW，标题打过切/欠切指标

量化（终端）：
  过切 = 每个 EMG burst 被几个新段覆盖（目标≈1，旧 vpeak≈2）
  欠切 = 每个新段含几个 EMG burst（目标≈1，旧 hold2hold 在连续数据会 >1）
  hold 捕获 = 新段真实 hold 时长中位数（旧仅 50ms pad）

用法：（先激活带 sklearn/torch 的环境，或用 ENV=/path/to/python 覆盖）
  ENV=${ENV:-python}
  $ENV diag_seg.py --out out_pose2 --stems jm-0503__20260430-left-2__20260430_164846
  $ENV diag_seg.py --out out_pose2 --stems hzy-1217__20260501-left-6__20260501_172246
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

from emg_label import (action_segmentation, io_utils, pose_segmentation,  # noqa: E402
                       segmentation)
from emg_label.config import Config  # noqa: E402


def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _split_factor(units, segs):
    """For each unit (s,e): how many segs overlap it. Returns Counter of counts."""
    return Counter(sum(1 for ss, se in segs if _overlap(us, ue, ss, se) > 0)
                   for us, ue in units)


def _fmt(counter):
    tot = sum(counter.values()) or 1
    return ", ".join(f"{k}->{v}({v/tot:.0%})" for k, v in sorted(counter.items()))


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

    # --- OLD velocity-peak segments (over-cut reference, with range gate) ---
    spd = pose_segmentation.pose_speed(ja, fs, cfg.pose_smooth_ms)
    detect = pose_segmentation.robust_threshold(spd, cfg.pose_pct, cfg.pose_mad)
    vsegs, _ = pose_segmentation.velocity_peak_segments(
        spd, fs, detect, bound_frac=cfg.pose_bound_frac, prom_k=cfg.pose_prom_k,
        min_gesture_s=cfg.pose_min_gesture_s, merge_gap_s=cfg.pose_peak_merge_gap_s)

    def seg_range(a, b):
        w = ja[a:b]
        return float(np.linalg.norm(np.nanmax(w, 0) - np.nanmin(w, 0))) if b > a else 0.0
    vsegs = [(a, b) for a, b in vsegs if seg_range(a, b) >= cfg.pose_min_range]

    # --- OLD hold-to-hold static_motion motion segments (under-cut reference) ---
    static, motion = pose_segmentation.static_motion_intervals(
        spd, fs, detect, min_static_s=cfg.min_static_s,
        min_motion_s=cfg.min_motion_s, merge_gap_s=cfg.merge_gap_s)

    # --- NEW unified segmentation -------------------------------------------
    actions, dbg = action_segmentation.segment_recording(emg, ja, cfg)
    new_full = [(a["start"], a["end"]) for a in actions]
    new_motion = [(a["motion_start"], a["motion_end"]) for a in actions]
    new_holds = [(a["hold_start"], a["hold_end"]) for a in actions]
    apexes = [a["apex"] for a in actions]
    nspd = dbg["spd"]
    me_thr, mx_thr = dbg["move_enter"], dbg["move_exit"]

    # --- quantify -----------------------------------------------------------
    sf_burst_old = _split_factor(bursts, vsegs)            # over-cut (old vpeak)
    sf_burst_new = _split_factor(bursts, new_full)         # over-cut (new)
    # over-cut vs the natural-gesture proxy (old hold2hold motion run): EMG bursts
    # under-count gestures (gentle ones fire at rest-level EMG), so per-gesture is
    # the honest over-cut measure.
    oc_gest_vp = _split_factor(motion, vsegs)              # old vpeak per gesture
    oc_gest_new = _split_factor(motion, new_full)          # new per gesture
    # under-cut: bursts whose center falls inside each segment
    def bursts_in(segs):
        c = Counter()
        for s, e in segs:
            c[sum(1 for bs, be in bursts if s <= (bs + be) // 2 < e)] += 1
        return c
    uc_old = bursts_in([(s, e) for s, e in motion])        # under-cut (old h2h)
    uc_new = bursts_in(new_full)                           # under-cut (new)

    hold_durs = np.array([(he - hs) / fs for hs, he in new_holds]) if new_holds else np.array([])
    apex_in_hold = (sum(1 for a in actions
                        if a["hold_start"] <= a["apex"] < a["hold_end"])
                    / len(actions)) if actions else float("nan")
    # seamless tiling check
    seamless = all(new_full[i][1] == new_full[i + 1][0]
                   for i in range(len(new_full) - 1))
    flags = Counter(a["review_flag"] for a in actions)

    print(f"\n=== {stem} ===")
    print(f"  duration {n/fs:.1f}s | EMG bursts {len(bursts)} | "
          f"old vpeak {len(vsegs)} | old hold2hold {len(motion)} | "
          f"NEW {len(actions)}")
    print(f"  move_enter={me_thr:.0f} move_exit={mx_thr:.0f} deg/s "
          f"(auto={cfg.auto_move_thresh}) | R2 splits={dbg['n_r2']} "
          f"R7 splits={dbg['n_r7']} | flags={dict(flags)}")
    print(f"  -- 过切 vs 自然手势 (per old-hold2hold motion -> #segs overlapping; want ~1) --")
    print(f"     old vpeak : {_fmt(oc_gest_vp)}  (count {len(vsegs)} vs gestures {len(motion)})")
    print(f"     NEW       : {_fmt(oc_gest_new)}  (count {len(actions)} vs gestures {len(motion)})")
    print(f"  -- 过切 vs EMG burst (per burst -> #segs; bursts under-count gentle gestures) --")
    print(f"     old vpeak : {_fmt(sf_burst_old)}  (ratio {len(vsegs)/max(1,len(bursts)):.2f})")
    print(f"     NEW       : {_fmt(sf_burst_new)}  (ratio {len(actions)/max(1,len(bursts)):.2f})")
    print(f"  -- 欠切 (per seg -> #EMG bursts inside; want ~1) --")
    print(f"     old h2h   : {_fmt(uc_old)}")
    print(f"     NEW       : {_fmt(uc_new)}")
    if hold_durs.size:
        frac_real = float(np.mean(hold_durs >= cfg.min_static_s))
        print(f"  -- hold 捕获 -- median {np.median(hold_durs):.2f}s | "
              f">={cfg.min_static_s}s in {frac_real:.0%} segs | "
              f"apex-in-hold {apex_in_hold:.0%} | seamless={seamless}")

    # --- plot ---------------------------------------------------------------
    t = np.arange(n) / fs
    fig, ax = plt.subplots(4, 1, figsize=(18, 11), sharex=True)

    ax[0].plot(t, env, lw=0.6, color="k")
    ax[0].axhline(enter, color="r", ls="--", lw=0.7)
    for bs, be in bursts:
        ax[0].axvspan(bs / fs, be / fs, color="green", alpha=0.15)
    ax[0].set_ylabel(f"EMG env\n{len(bursts)} bursts")

    sp = np.nan_to_num(spd)
    ycap = max(float(np.percentile(sp, 97)), me_thr * 2.0) * 1.1
    ax[1].plot(t, sp, lw=0.6, color="navy")
    ax[1].set_ylim(0, ycap)
    ax[1].axhline(detect, color="r", ls="--", lw=0.8, label="detect")
    for a, b in vsegs:
        ax[1].axvspan(a / fs, b / fs, color="orange", alpha=0.25)
    ax[1].legend(loc="upper right", fontsize=7)
    ax[1].set_ylabel(f"OLD vpeak\n{len(vsegs)} segs")

    ax[2].plot(t, np.nan_to_num(nspd), lw=0.6, color="navy")
    ax[2].set_ylim(0, ycap)
    ax[2].axhline(me_thr, color="darkorange", ls="--", lw=0.8, label="move_enter")
    ax[2].axhline(mx_thr, color="seagreen", ls=":", lw=0.9, label="move_exit")
    for hs, he in new_holds:
        ax[2].axvspan(hs / fs, he / fs, color="lightblue", alpha=0.5)
    for ms, me in new_motion:
        ax[2].axvspan(ms / fs, me / fs, color="green", alpha=0.30)
    for a in apexes:
        ax[2].axvline(a / fs, color="purple", lw=0.5, alpha=0.6)
    for s, _ in new_full:
        ax[2].axvline(s / fs, color="red", lw=0.5, alpha=0.5)
    ax[2].legend(loc="upper right", fontsize=7)
    ax[2].set_ylabel(f"NEW unified\n{len(actions)} segs (motion+hold)")

    ax[3].set_ylim(0, 1)
    ax[3].set_yticks([0.88, 0.63, 0.38, 0.13])
    ax[3].set_yticklabels(["EMG\nburst", "vpeak\n(old)", "hold2hold\n(old)", "NEW"])
    for bs, be in bursts:
        ax[3].axvspan(bs / fs, be / fs, ymin=0.78, ymax=0.98, color="green", alpha=0.4)
    for a, b in vsegs:
        ax[3].axvspan(a / fs, b / fs, ymin=0.53, ymax=0.73, color="orange", alpha=0.5)
    for a, b in motion:
        ax[3].axvspan(a / fs, b / fs, ymin=0.28, ymax=0.48, color="steelblue", alpha=0.5)
    for s, e in new_full:
        ax[3].axvspan(s / fs, e / fs, ymin=0.03, ymax=0.23, color="purple", alpha=0.35)
    for ms, me in new_motion:
        ax[3].axvspan(ms / fs, me / fs, ymin=0.03, ymax=0.23, color="green", alpha=0.5)
    ax[3].set_xlabel("time (s)")
    ax[3].set_xlim(0, n / fs)
    ax[3].set_title(
        f"过切 vpeak/burst={len(vsegs)/max(1,len(bursts)):.2f} -> NEW/burst="
        f"{len(actions)/max(1,len(bursts)):.2f} | "
        f"欠切 old-h2h per-seg bursts {_fmt(uc_old)} -> NEW {_fmt(uc_new)} | "
        f"R2={dbg['n_r2']} R7={dbg['n_r7']}", fontsize=8, loc="left")

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
    ap.add_argument("--out", default="out_pose2", help="dir with clips.csv")
    ap.add_argument("--proto-out", default="out_diag_s0")
    ap.add_argument("--stems", required=True, help="comma-separated stems")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--pose-smooth-ms", type=float, default=250.0)
    ap.add_argument("--move-enter", type=float, default=None,
                    help="fixed pose move-onset deg/s (disables auto)")
    ap.add_argument("--move-exit", type=float, default=None,
                    help="fixed pose settle deg/s (disables auto)")
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=None, help="zoom end (s)")
    args = ap.parse_args()
    cfg = Config(fs=args.fs, pose_smooth_ms=args.pose_smooth_ms)
    if args.move_enter is not None and args.move_exit is not None:
        cfg.move_enter, cfg.move_exit, cfg.auto_move_thresh = (
            args.move_enter, args.move_exit, False)
    for stem in args.stems.split(","):
        process(stem.strip(), cfg, args)


if __name__ == "__main__":
    main()
