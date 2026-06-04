from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_overview(emg, activity, segments, fs, out_path,
                  enter_thr=None, exit_thr=None, labels=None,
                  apex_samples=None):
    n = len(activity)
    t = np.arange(n) / fs
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    env = np.abs(np.asarray(emg)).mean(axis=1)
    axes[0].plot(t, env, lw=0.5)
    axes[0].set_ylabel("EMG |mean| (raw)")
    act = np.asarray(activity)
    axes[1].plot(t, act, lw=0.8, color="k")
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
    # Apex = the held-pose frame used as the clustering feature (max joint
    # deviation from rest, usually just after the EMG burst). Purple line on
    # both axes + a marker on the envelope at that instant.
    if apex_samples is not None:
        for a in apex_samples:
            ta = a / fs
            for ax in axes:
                ax.axvline(ta, color="purple", ls="--", lw=0.7, alpha=0.6)
            ai = min(max(int(a), 0), n - 1)
            axes[1].plot([ta], [act[ai]], marker="v", color="purple",
                         markersize=5, zorder=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_overview_dual(env, pose_speed, fs, out_path,
                       burst_segments=None, clips=None,
                       clip_to_burst=None,
                       enter_thr=None, exit_thr=None, pose_thr=None,
                       pose_exit_thr=None, apex_samples=None):
    """Two-row overview for ground-truth QA.

    Row 0: EMG envelope + burst spans (green, labelled b{idx}) + enter/exit
           thresholds. A green band is one detected muscle activation.
    Row 1: pose speed + clip spans (orange, labelled c{clip_id}); the motion
           sub-window (the actual movement = one clip) is filled darker, and
           each clip's apex (formed-pose frame) is a purple line.

    Burst (row 0) and clip (row 1) are the two independent segmenters; a green
    band with no orange one below it (or vice versa) is a disagreement worth a
    manual look. ``clip_to_burst`` is accepted for call-site compatibility but
    no longer drawn as a separate lane.
    """
    n = len(env)
    t = np.arange(n) / fs
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(t, np.asarray(env), lw=0.8, color="k")
    axes[0].set_ylabel("EMG envelope")
    if enter_thr is not None:
        axes[0].axhline(enter_thr, color="r", ls="--", lw=0.8)
    if exit_thr is not None:
        axes[0].axhline(exit_thr, color="orange", ls="--", lw=0.8)

    ps_arr = np.asarray(pose_speed)
    axes[1].plot(t, ps_arr, lw=0.8, color="navy")
    axes[1].set_ylabel("pose speed (deg/s)\n(‖dθ/dt‖ over 20 joints; peaks clipped)")
    axes[1].set_xlabel("time (s)")
    # move_enter (onset) and move_exit (settle) thresholds -- the band between
    # them is the hysteresis dead-zone; holds sit below move_exit.
    if pose_thr is not None:
        axes[1].axhline(pose_thr, color="r", ls="--", lw=0.8,
                        label="move_enter")
    if pose_exit_thr is not None:
        axes[1].axhline(pose_exit_thr, color="seagreen", ls=":", lw=1.0,
                        label="move_exit")
    # Transition peaks reach ~thousands of deg/s and would squash the threshold
    # band + the low-speed holds into an unreadable sliver. Cap the y-axis at a
    # few x the onset threshold so the segmentation-relevant band (holds, the two
    # thresholds, the rising flank) stays legible; tall peaks clip off-screen --
    # we only need to see WHERE speed crosses the thresholds, not how tall it got.
    finite = ps_arr[np.isfinite(ps_arr)]
    cap = 0.0
    if pose_thr:
        cap = float(pose_thr) * 4.0
    elif finite.size:
        cap = float(np.percentile(finite, 90))
    if cap > 0:
        axes[1].set_ylim(0, cap)
    if pose_thr is not None or pose_exit_thr is not None:
        axes[1].legend(loc="upper right", fontsize=7)

    if burst_segments:
        for idx, (s, e) in enumerate(burst_segments):
            axes[0].axvspan(s / fs, e / fs, color="green", alpha=0.15)
            axes[0].text((s + e) / 2 / fs, axes[0].get_ylim()[1] * 0.9,
                         f"b{idx}", ha="center", fontsize=7, color="darkgreen")

    if clips:
        for c in clips:
            # one action = motion run (orange) + its following real hold (blue).
            axes[1].axvspan(c["motion_start"] / fs, c["motion_end"] / fs,
                            color="orange", alpha=0.30)
            hs = c.get("hold_start", c.get("static_out_start", c["motion_end"]))
            he = c.get("hold_end", c.get("static_out_end", c["clip_end"]))
            axes[1].axvspan(hs / fs, he / fs, color="steelblue", alpha=0.22)
            axes[1].text((c["clip_start"] + c["clip_end"]) / 2 / fs,
                         axes[1].get_ylim()[1] * 0.9,
                         f"c{c['clip_id']}", ha="center", fontsize=7,
                         color="darkorange")

    if apex_samples is not None:
        for a in apex_samples:
            axes[1].axvline(a / fs, color="purple", ls="--", lw=0.6, alpha=0.55)

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
