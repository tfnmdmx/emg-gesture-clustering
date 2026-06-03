from __future__ import annotations

"""Prototype: velocity-PEAK dynamic-gesture segmentation (standalone, read-only).

Why this exists
---------------
The deployed `clips.csv` cuts gestures as ``static -> motion -> static``. That
works when gestures are separated by holds, but when a subject gestures
*continuously* (never settling) the whole run collapses into ONE mega-clip
(e.g. hzy-1217 ...153603 = a single 27 s "motion"), and even normal recordings
merge several sub-movements into one long clip (e.g. ax-0819 ...102752 clip c1).

For DYNAMIC gestures the natural unit is a *movement* = one bell in pose-speed,
bounded by the velocity valleys around it. This prototype segments that way and
renders a side-by-side overview (current clips vs new peak-segments) so the
split can be eyeballed before anything in the real pipeline changes.

EMG is demoted to a vote/flag only: each new segment is tagged both-agree
(overlaps an EMG burst) or pose-only; EMG bursts with no overlapping segment are
surfaced as review flags. Nothing here is merged into the segment set.

Run:
  python proto_peak_seg.py                 # the two canonical examples
  python proto_peak_seg.py --stems a,b,c   # explicit recordings
Outputs: out_peakseg_proto/{stem}.png
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from emg_label import io_utils, pose_segmentation, segmentation  # noqa: E402
from emg_label.config import Config  # noqa: E402


# recordings that exposed the two failure modes in the current pipeline
DEFAULT_STEMS = [
    "ax-0819__20260428-left__20260428_102752",        # normal but merges clip c1
    "hzy-1217__20260503-left-4__20260503_153603",      # continuous -> one 27 s clip
]


def velocity_peak_segments(spd, fs, detect_thr, bound_frac=0.5,
                           prom_k=2.0, min_gesture_s=0.25, merge_gap_s=0.12):
    """One dynamic segment per pose-speed peak; split mega-runs at valleys.

    1. peaks above ``detect_thr`` (robust motion floor), >= ``min_gesture_s``
       apart, standing out by >= ``prom_k * MAD`` (a wobble inside one movement
       is not its own gesture);
    2. bound each movement at the lower ``bound_frac * detect_thr`` so the
       rising/falling flanks stay in (hysteresis: detect high, bound low);
    3. when one supra-``bound`` run holds several peaks (continuous gesturing
       that never settles -- exactly what SMC merges) split it at the velocity
       *valleys* between consecutive peaks.

    apex := peak-velocity frame (the most intense instant of the movement).
    Returns (segments, apex_samples, detect_thr, bound_level, prominence).
    """
    s = np.nan_to_num(np.asarray(spd, float), nan=0.0)  # NaN occlusion -> no motion
    finite = s[np.isfinite(s)]
    med = float(np.median(finite)) if finite.size else 0.0
    mad = 1.4826 * float(np.median(np.abs(finite - med))) if finite.size else 0.0
    prom = max(prom_k * mad, 1e-9)
    min_dist = max(1, int(round(min_gesture_s * fs)))
    peaks, _ = find_peaks(s, height=detect_thr, distance=min_dist, prominence=prom)

    bound = bound_frac * detect_thr
    runs = pose_segmentation.mask_to_intervals(s > bound, fs, 0.0, merge_gap_s)
    segs, apexes = [], []
    for rs, re in runs:
        pk = peaks[(peaks >= rs) & (peaks < re)]
        if len(pk) == 0:
            continue
        bnds = [rs]
        for j in range(len(pk) - 1):
            bnds.append(pk[j] + int(np.argmin(s[pk[j]:pk[j + 1] + 1])))
        bnds.append(re)
        for j in range(len(pk)):
            a, b = bnds[j], bnds[j + 1]
            if b - a >= min_dist:
                segs.append((int(a), int(b)))
                apexes.append(int(pk[j]))
    return segs, apexes, float(detect_thr), float(bound), float(prom)


def _ov(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def process(stem, clips_df, cfg, args, out_dir):
    rows = clips_df[clips_df["source_file"] == stem + ".npz"]
    if rows.empty:
        print(f"SKIP {stem}: not in clips.csv")
        return
    src = str(rows.iloc[0]["source_path"])
    hand = str(rows.iloc[0].get("hand") or "") or None
    if not os.path.isfile(src):
        print(f"SKIP {stem}: source npz missing ({src})")
        return
    emg, ja = io_utils.load_npz(src, hand=hand)
    n = len(emg)
    fs = cfg.fs

    # signals
    spd = pose_segmentation.pose_speed(ja, fs, args.pose_smooth_ms)
    detect = pose_segmentation.robust_threshold(spd, cfg.pose_pct, cfg.pose_mad)
    segs0, apexes0, detect, bound, prom = velocity_peak_segments(
        spd, fs, detect, bound_frac=args.bound_frac, prom_k=args.prom_k,
        min_gesture_s=args.min_gesture_s, merge_gap_s=args.merge_gap_s)

    # ---- ABSOLUTE motion gate (the fix for static-recording phantom segs) ----
    # robust_threshold is per-recording relative, so on a near-static recording
    # it descends to the noise floor and manufactures segments. A real dynamic
    # gesture must move the joints an absolute distance, regardless of how
    # quiet the rest of the recording is. pose_range (joint excursion) is the
    # direct, noise-robust measure (unlike speed it's not inflated by a single
    # spiky sample). rec_range = recording-wide excursion -> flags the whole
    # recording as static.
    def seg_range(a, b):
        w = ja[a:b]
        return float(np.linalg.norm(np.nanmax(w, 0) - np.nanmin(w, 0))) if b > a else 0.0
    rec_range = float(np.linalg.norm(np.nanmax(ja, 0) - np.nanmin(ja, 0)))
    ranges = [seg_range(a, b) for a, b in segs0]
    keep = [r >= args.min_range for r in ranges]
    segs = [s for s, k in zip(segs0, keep) if k]
    apexes = [p for p, k in zip(apexes0, keep) if k]
    n_dropped = len(segs0) - len(segs)

    env = segmentation.emg_envelope(emg, fs, cfg.smooth_ms)
    bursts, _, enter, _ = segmentation.segment_emg(emg, cfg)

    # current clips for comparison
    cur = [(int(r.clip_start_sample), int(r.clip_end_sample), int(r.clip_id),
            int(r.motion_start_sample), int(r.motion_end_sample))
           for r in rows.itertuples()]

    # EMG vote
    seg_both = [any(_ov(a, b, bs, be) > 0 for bs, be in bursts) for a, b in segs]
    burst_only = [(bs, be) for bs, be in bursts
                  if not any(_ov(a, b, bs, be) > 0 for a, b in segs)]
    n_both = sum(seg_both)

    # split factor on the longest current clip
    longest = max(cur, key=lambda c: c[4] - c[3]) if cur else None
    split_k = (sum(1 for ap in apexes if longest[0] <= ap < longest[1])
               if longest else 0)

    # ---- plot ----
    t = np.arange(n) / fs
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

    ax[0].plot(t, env, lw=0.7, color="k")
    ax[0].set_ylabel("EMG envelope")
    ax[0].axhline(enter, color="r", ls="--", lw=0.7)
    for bs, be in bursts:
        ax[0].axvspan(bs / fs, be / fs, color="green", alpha=0.12)

    ax[1].plot(t, np.nan_to_num(spd), lw=0.7, color="navy")
    ax[1].set_ylabel("pose speed (rad/s)")
    ax[1].axhline(detect, color="r", ls="--", lw=0.8, label="detect")
    ax[1].axhline(bound, color="gray", ls=":", lw=0.8, label="bound")
    ax[1].legend(loc="upper right", fontsize=7)
    ymax = ax[1].get_ylim()[1]
    for i, ((a, b), kp) in enumerate(zip(segs0, keep)):
        ax[1].axvspan(a / fs, b / fs,
                      color=("orange" if kp else "lightgray"),
                      alpha=(0.22 if kp else 0.5))
        ax[1].axvline(apexes0[i] / fs, color="red", lw=0.7, alpha=0.7)
        ax[1].text((a + b) / 2 / fs, ymax * 0.92, f"p{i}", ha="center",
                   fontsize=6, color=("darkorange" if kp else "gray"))

    # comparison lanes
    ax[2].set_ylim(0, 1)
    ax[2].set_yticks([0.75, 0.3])
    ax[2].set_yticklabels(["current\nclips", "NEW\npeak-seg"])
    ax[2].set_xlabel("time (s)")
    ax[2].set_xlim(0, n / fs)
    for cs, ce, cid, ms, me in cur:
        ax[2].axvspan(cs / fs, ce / fs, ymin=0.55, ymax=0.95,
                      color="steelblue", alpha=0.45)
        ax[2].text((cs + ce) / 2 / fs, 0.83, f"c{cid}", ha="center",
                   va="center", fontsize=6, color="white")
    ki = 0
    for (a, b), kp in zip(segs0, keep):
        if kp:
            col = "green" if seg_both[ki] else "orange"
            ax[2].axvspan(a / fs, b / fs, ymin=0.10, ymax=0.45, color=col, alpha=0.5)
            ax[2].text((a + b) / 2 / fs, 0.27, f"p{ki}", ha="center", va="center",
                       fontsize=6, color="black")
            ki += 1
        else:                            # dropped by absolute gate -> gray
            ax[2].axvspan(a / fs, b / fs, ymin=0.10, ymax=0.45,
                          color="lightgray", alpha=0.6)
    for bs, be in burst_only:           # EMG-only -> review flag (red tick)
        ax[2].plot([(bs + be) / 2 / fs], [0.05], marker="^", color="red",
                   markersize=5)
    static_tag = "  [STATIC RECORDING]" if rec_range < args.min_range else ""
    ax[2].set_title(
        f"current {len(cur)} clips  ->  {len(segs)} kept "
        f"(+{n_dropped} dropped by range>={args.min_range:g} gate)   |   "
        f"vote: {n_both} both, {len(segs)-n_both} pose-only, "
        f"{len(burst_only)} EMG-only   |   rec_range={rec_range:.1f}{static_tag}",
        fontsize=9, loc="left")

    suffix = ""
    if args.xmax is not None:
        for a in ax:
            a.set_xlim(args.xmin, args.xmax)
        suffix = f"_zoom{int(args.xmin)}-{int(args.xmax)}"
    fig.suptitle(stem, fontsize=10)
    fig.tight_layout()
    out = os.path.join(out_dir, stem + suffix + ".png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"{stem}: {len(cur)} clips -> {len(segs0)} peaks -> {len(segs)} KEPT "
          f"(+{n_dropped} dropped by gate); rec_range={rec_range:.2f} "
          f"{'[STATIC]' if rec_range < args.min_range else ''}; "
          f"both {n_both}, pose-only {len(segs)-n_both}, "
          f"EMG-only {len(burst_only)}; -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out_pose2", help="dir with clips.csv to compare against")
    ap.add_argument("--proto-out", default="out_peakseg_proto")
    ap.add_argument("--stems", default=None, help="comma-separated stems (default: 2 examples)")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--pose-smooth-ms", type=float, default=250.0)
    ap.add_argument("--prom-k", type=float, default=2.0, help="peak prominence = k * MAD(speed)")
    ap.add_argument("--min-gesture-s", type=float, default=0.25)
    ap.add_argument("--bound-frac", type=float, default=0.5, help="segment-bound level = frac * detect")
    ap.add_argument("--merge-gap-s", type=float, default=0.12)
    ap.add_argument("--xmin", type=float, default=0.0, help="zoom window start (s)")
    ap.add_argument("--xmax", type=float, default=None, help="zoom window end (s); enables a zoomed render")
    ap.add_argument("--min-range", type=float, default=15.0,
                    help="ABSOLUTE joint-excursion floor (pose_range) for a "
                         "segment to count as a real dynamic gesture. Static "
                         "recordings (ax-0819 ~2) fall far below typical (~177).")
    args = ap.parse_args()

    cfg = Config(fs=args.fs, pose_smooth_ms=args.pose_smooth_ms)
    os.makedirs(args.proto_out, exist_ok=True)
    clips_path = os.path.join(args.out, "clips.csv")
    clips_df = pd.read_csv(clips_path, dtype={"source_file": str})

    stems = (args.stems.split(",") if args.stems else DEFAULT_STEMS)
    for stem in stems:
        process(stem.strip(), clips_df, cfg, args, args.proto_out)


if __name__ == "__main__":
    main()
