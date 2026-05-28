from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_overview(emg, activity, segments, fs, out_path,
                  enter_thr=None, exit_thr=None, labels=None):
    n = len(activity)
    t = np.arange(n) / fs
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    env = np.abs(np.asarray(emg)).mean(axis=1)
    axes[0].plot(t, env, lw=0.5)
    axes[0].set_ylabel("EMG |mean| (raw)")
    axes[1].plot(t, np.asarray(activity), lw=0.8, color="k")
    axes[1].set_ylabel("EMG envelope")
    axes[1].set_xlabel("time (s)")
    if enter_thr is not None:
        axes[1].axhline(enter_thr, color="r", ls="--", lw=0.8)
    if exit_thr is not None:
        axes[1].axhline(exit_thr, color="orange", ls="--", lw=0.8)
    ymax = axes[1].get_ylim()[1]
    for idx, (s, e) in enumerate(segments):
        for ax in axes:
            ax.axvspan(s / fs, e / fs, color="green", alpha=0.15)
        lab = str(labels[idx]) if labels is not None else str(idx)
        axes[1].text((s + e) / 2 / fs, ymax * 0.9, lab, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_cluster_preview(centroids, counts, ids, out_path):
    k = len(centroids)
    ncol = 4
    nrow = max(1, int(np.ceil(k / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.2 * nrow),
                             squeeze=False)
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i < k:
            ax.bar(range(len(centroids[i])), centroids[i])
            ax.set_title(f"cluster {ids[i]} (n={counts[i]})", fontsize=9)
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_cluster_hands(centroids, counts, ids, out_path, side="left"):
    from emg_label.hand3d import angles_batch_to_landmarks, draw_hand

    # Batched FK is one torch call instead of k.
    all_lm = angles_batch_to_landmarks(np.asarray(centroids), side=side)
    pts = all_lm.reshape(-1, 3)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    # Center each axis around midpoint with the same half-range so the
    # aspect stays true and small clusters don't get visually inflated.
    mid = (mn + mx) / 2.0
    half = (mx - mn).max() / 2.0 * 1.05
    lo = mid - half
    hi = mid + half
    k = len(centroids)
    ncol = 4
    nrow = max(1, int(np.ceil(k / ncol)))
    fig = plt.figure(figsize=(3.6 * ncol, 3.4 * nrow))
    for i in range(k):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        draw_hand(ax, all_lm[i])
        ax.set_title(f"cluster {ids[i]} (n={counts[i]})", fontsize=9)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        # emg2pose coords: +X = wrist->fingertips, Y = palm normal, Z = finger spread.
        # 3/4 view from above the back of the hand makes fingers and thumb readable.
        ax.view_init(elev=25, azim=-60)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
