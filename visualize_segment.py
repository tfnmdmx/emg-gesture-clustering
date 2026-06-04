from __future__ import annotations

"""Visualize one exported gesture segment npz.

Renders a single PNG with four views of the segment:
  1. 16-channel EMG (stacked, offset traces)
  2. EMG envelope (same signal segmentation uses) + apex marker
  3. 20 joint-angle channels over time
  4. 3D hand pose (emg2pose FK) at start / apex / end frames

Usage:
    python visualize_segment.py <segment.npz> [-o out.png] [--side left|right]

The apex frame is the moment of maximum joint deviation from the segment's
mean pose -- the most "settled" gesture pose. It is computed with the exact
shared helper used for the clustering feature (features.apex_index).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from emg_label import features  # noqa: E402
from emg_label.hand3d import angles_to_landmarks, draw_hand  # noqa: E402
from emg_label.io_utils import side_from_meta  # noqa: E402
from emg_label.segmentation import emg_envelope  # noqa: E402
from emg_label.skeleton import axis_limits  # noqa: E402


def visualize(npz_path: str, out_path: str, side: str | None = None):
    d = np.load(npz_path, allow_pickle=True)
    emg = np.asarray(d["emg"], dtype=float)            # (T, 16)
    ja = np.asarray(d["joint_angles"], dtype=float)    # (T, 20)
    fs = int(d["fs"]) if "fs" in d.files else 2000
    label = str(d["label"]) if "label" in d.files else "?"
    cid = str(d["cluster_id"]) if "cluster_id" in d.files else "?"
    src = str(d["source_file"]) if "source_file" in d.files else ""
    seg_idx = str(d["seg_idx"]) if "seg_idx" in d.files else "?"
    if side is None:
        side = side_from_meta(d)

    T = emg.shape[0]
    t = np.arange(T) / fs
    apex = features.apex_index(ja, 0, T, np.nanmedian(ja, axis=0), fs)

    # Envelope = baseline-centered RMS across channels (matches segmentation).
    env = emg_envelope(emg, fs=fs, smooth_ms=50.0)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 3, width_ratios=[2, 2, 1.6],
                          hspace=0.35, wspace=0.25)

    # --- (1) 16-channel EMG, stacked ----------------------------------------
    ax_emg = fig.add_subplot(gs[0, :2])
    # Offset each channel by a fixed step so all 16 are visible without overlap.
    step = np.percentile(np.abs(emg), 99) * 2.2 + 1e-6
    for ch in range(16):
        ax_emg.plot(t, emg[:, ch] + ch * step, lw=0.4)
    ax_emg.axvline(apex / fs, color="r", ls="--", lw=0.8)
    ax_emg.set_yticks([ch * step for ch in range(16)])
    ax_emg.set_yticklabels([f"ch{ch}" for ch in range(16)], fontsize=6)
    ax_emg.set_title(f"16-channel EMG (stacked)   |   {T} samples = "
                     f"{T / fs:.2f}s @ {fs}Hz", fontsize=10)
    ax_emg.set_xlim(0, t[-1])

    # --- (2) EMG envelope ----------------------------------------------------
    ax_env = fig.add_subplot(gs[1, :2])
    ax_env.plot(t, env, color="k", lw=1.0)
    ax_env.fill_between(t, env, color="green", alpha=0.12)
    ax_env.axvline(apex / fs, color="r", ls="--", lw=0.8,
                   label=f"apex @ {apex / fs:.2f}s")
    ax_env.set_ylabel("EMG |mean| env")
    ax_env.set_title("EMG envelope (baseline-centered RMS of 16 ch)", fontsize=10)
    ax_env.set_xlim(0, t[-1])
    ax_env.legend(fontsize=8, loc="upper right")

    # --- (3) 20 joint angles -------------------------------------------------
    ax_ja = fig.add_subplot(gs[2, :2])
    for j in range(20):
        ax_ja.plot(t, ja[:, j], lw=0.7)
    ax_ja.axvline(apex / fs, color="r", ls="--", lw=0.8)
    ax_ja.set_xlabel("time (s)")
    ax_ja.set_ylabel("joint angle (rad)")
    ax_ja.set_title("20 joint-angle channels", fontsize=10)
    ax_ja.set_xlim(0, t[-1])

    # --- (4) 3D hand poses at start / apex / end -----------------------------
    frames = [("start", 0), ("apex", apex), ("end", T - 1)]
    lms = [angles_to_landmarks(ja[f], side=side) for _, f in frames]
    lims = axis_limits(np.concatenate(lms, axis=0), margin_ratio=0.025)
    for i, (name, f) in enumerate(frames):
        ax = fig.add_subplot(gs[i, 2], projection="3d")
        draw_hand(ax, lms[i])
        ax.set_title(f"hand @ {name} (t={f / fs:.2f}s)", fontsize=9)
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
        ax.view_init(elev=25, azim=-60)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.suptitle(
        f"label={label}  cluster_id={cid}  side={side}   |   "
        f"{src}  seg{seg_idx}",
        fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"  T={T} ({T / fs:.2f}s)  apex frame={apex} ({apex / fs:.2f}s)  side={side}")


def main():
    ap = argparse.ArgumentParser(description="Visualize one exported segment npz")
    ap.add_argument("npz", help="path to a segment .npz")
    ap.add_argument("-o", "--out", default=None,
                    help="output PNG (default: <npz stem>_viz.png next to it)")
    ap.add_argument("--side", choices=["left", "right"], default=None,
                    help="hand side (default: inferred from metadata)")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.npz)[0] + "_viz.png"
    visualize(args.npz, out, side=args.side)


if __name__ == "__main__":
    main()
