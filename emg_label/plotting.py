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
                       apex_samples=None):
    """Three-row overview for ground-truth QA.

    Row 0: EMG envelope + burst spans (green, labelled b{idx}).
    Row 1: pose speed + clip spans (orange, labelled c{clip_id}); the motion
           sub-window inside each clip is filled darker.
    Row 2: matched ("kept") segments only -- clips with a paired EMG burst
           (clip_to_burst[i] >= 0). Each kept segment is a vertical stack of
           the orange clip span + the green burst span, numbered s1, s2, ...
           in time order. This is the high-confidence subset both segmenters
           agree on -- the primary ground-truth candidate set.

    Disagreement still surfaces in rows 0/1: a green band without an orange
    one (or vice versa) flags a manual-review candidate, and is absent from
    row 2 by construction.
    """
    n = len(env)
    t = np.arange(n) / fs
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, np.asarray(env), lw=0.8, color="k")
    axes[0].set_ylabel("EMG envelope")
    if enter_thr is not None:
        axes[0].axhline(enter_thr, color="r", ls="--", lw=0.8)
    if exit_thr is not None:
        axes[0].axhline(exit_thr, color="orange", ls="--", lw=0.8)

    axes[1].plot(t, np.asarray(pose_speed), lw=0.8, color="navy")
    axes[1].set_ylabel("pose speed (rad/s)")
    if pose_thr is not None:
        axes[1].axhline(pose_thr, color="r", ls="--", lw=0.8)
    # pose_speed is a derivative -> a few rare spikes otherwise autoscale the
    # axis tall and flatten the curve. Cap at the 98.5th pct (clip those spikes
    # off the top), but keep the motion threshold band visible.
    _psf = np.asarray(pose_speed, dtype=float)
    _psf = _psf[np.isfinite(_psf)]
    if _psf.size:
        _top = float(np.nanpercentile(_psf, 98.5))
        if pose_thr is not None and np.isfinite(pose_thr):
            _top = max(_top, float(pose_thr) * 3.0)
        if _top > 0:
            axes[1].set_ylim(0, _top * 1.1)

    if burst_segments:
        for idx, (s, e) in enumerate(burst_segments):
            for ax in axes[:2]:
                ax.axvspan(s / fs, e / fs, color="green", alpha=0.13)
            axes[0].text((s + e) / 2 / fs, axes[0].get_ylim()[1] * 0.9,
                         f"b{idx}", ha="center", fontsize=7, color="darkgreen")

    if clips:
        for c in clips:
            for ax in axes[:2]:
                ax.axvspan(c["clip_start"] / fs, c["clip_end"] / fs,
                           color="orange", alpha=0.10)
            axes[1].axvspan(c["motion_start"] / fs, c["motion_end"] / fs,
                            color="orange", alpha=0.30)
            axes[1].text((c["clip_start"] + c["clip_end"]) / 2 / fs,
                         axes[1].get_ylim()[1] * 0.9,
                         f"c{c['clip_id']}", ha="center", fontsize=7,
                         color="darkorange")

    if apex_samples is not None:
        for a in apex_samples:
            ta = a / fs
            for ax in axes[:2]:
                ax.axvline(ta, color="purple", ls="--", lw=0.6, alpha=0.55)

    # --- Row 2: ALL clips (top, s1..sN in time order = labelling chips) and
    #            ALL bursts (bottom, b0..). Matched clips get a darker, edged
    #            span; clip-only / burst-only ones still appear (faint / green).
    ax_m = axes[2]
    ax_m.set_ylabel("clips / bursts")
    ax_m.set_xlabel("time (s)")
    ax_m.set_ylim(0, 1)
    ax_m.set_yticks([])
    ax_m.set_xlim(0, n / fs)
    ax_m.axhline(0.5, color="lightgray", lw=0.5, zorder=0)

    n_c = n_b = 0
    if clips:
        order = []
        for ci, c in enumerate(clips):
            bi = int(clip_to_burst[ci]) if clip_to_burst is not None else -1
            matched = bool(burst_segments) and 0 <= bi < len(burst_segments)
            order.append((c["clip_start"], c, matched))
        order.sort(key=lambda x: x[0])
        show_lab = len(order) <= 40        # skip labels when too dense to read
        for s_idx, (_, c, matched) in enumerate(order, start=1):
            ax_m.axvspan(c["clip_start"] / fs, c["clip_end"] / fs,
                         ymin=0.55, ymax=0.95, color="orange",
                         alpha=0.45 if matched else 0.18,
                         ec="darkorange" if matched else "none", lw=0.6)
            if show_lab:
                ax_m.text((c["clip_start"] + c["clip_end"]) / 2 / fs, 0.75,
                          f"s{s_idx}", ha="center", va="center", fontsize=7,
                          fontweight="bold", color="black",
                          bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                    ec="gray", lw=0.4, alpha=0.85))
        n_c = len(order)
    if burst_segments:
        show_b = len(burst_segments) <= 40
        for b_idx, (bs, be) in enumerate(burst_segments):
            ax_m.axvspan(bs / fs, be / fs, ymin=0.05, ymax=0.45,
                         color="green", alpha=0.35)
            if show_b:
                ax_m.text((bs + be) / 2 / fs, 0.25, f"b{b_idx}", ha="center",
                          va="center", fontsize=6, color="darkgreen")
        n_b = len(burst_segments)
    ax_m.set_title(f"{n_c} clips (s1..s{n_c}) / {n_b} bursts "
                   f"(darker = burst∩clip matched)", fontsize=9, loc="left")

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
