from __future__ import annotations

"""Visualize the clustering feature space, per group.

Recomputes the exact apex-pose features cluster.py used (20-dim per segment),
then for each group produces a multi-panel figure:
  1. PCA 2D scatter, points colored by cluster, centroids marked.
  2. t-SNE 2D scatter, same coloring (better at showing separation).
  3. Centroid heatmap: k clusters x 20 joint channels (z-scored).
  4. Per-cluster silhouette + size bar (cluster compactness / population).

Features are cached to <out>/feature_cache/<group>.npz so reruns are fast.

Usage:
    python plot_cluster_features.py <input_dir> [--out out_fgw] [--fs 2000]
        [--no-tsne] [--force]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_samples  # noqa: E402

from emg_label import features, io_utils  # noqa: E402


def _group_features(input_dir, gdf, fs):
    """Recompute (X, cluster_ids) for one group -- identical to cluster.py."""
    feats, cids = [], []
    for fname, fdf in gdf.groupby("source_file"):
        _, ja = io_utils.load_npz(os.path.join(input_dir, fname))
        rest = np.median(ja, axis=0)
        fdf = fdf.sort_values("start_sample")
        starts = [int(v) for v in fdf["start_sample"].tolist()]
        ends = [int(v) for v in fdf["end_sample"].tolist()]
        clusters = [int(v) for v in fdf["cluster_id"].tolist()]
        for j in range(len(starts)):
            s = starts[j]
            win_end = (starts[j + 1] if j + 1 < len(starts)
                       else min(len(ja), ends[j] + fs))
            feats.append(features.apex_pose_feature(ja, s, win_end, rest, fs))
            cids.append(clusters[j])
        del ja
    return np.asarray(feats), np.asarray(cids)


def _scatter_by_cluster(ax, XY, cids, title, centroids_xy=None):
    uniq = sorted(set(int(c) for c in cids))
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(uniq):
        m = cids == c
        ax.scatter(XY[m, 0], XY[m, 1], s=8, alpha=0.55,
                   color=cmap(i % 20), label=str(c), linewidths=0)
    if centroids_xy is not None:
        for i, c in enumerate(uniq):
            x, y = centroids_xy[i]
            ax.scatter([x], [y], s=140, marker="*", color=cmap(i % 20),
                       edgecolor="k", linewidths=0.8, zorder=5)
            ax.text(x, y, str(c), fontsize=7, ha="center", va="center",
                    zorder=6)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def plot_group(group, X, cids, out_path, do_tsne=True):
    uniq = sorted(set(int(c) for c in cids))
    k = len(uniq)
    # z-score the 20-dim features (same transform clustering used).
    Xz, _, _ = features.zscore(X)

    # PCA 2D + project centroids into the same PCA space.
    pca = PCA(n_components=2, random_state=0)
    XY_pca = pca.fit_transform(Xz)
    cent_z = np.stack([Xz[cids == c].mean(axis=0) for c in uniq])
    cent_pca = pca.transform(cent_z)
    var = pca.explained_variance_ratio_

    # Centroid heatmap in z-scored joint space.
    # Silhouette per sample (needs >=2 clusters).
    try:
        sil = silhouette_samples(Xz, cids)
    except Exception:
        sil = np.zeros(len(cids))

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    _scatter_by_cluster(
        ax1, XY_pca, cids,
        f"PCA (PC1 {var[0]*100:.0f}% / PC2 {var[1]*100:.0f}%)", cent_pca)
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.legend(title="cluster", fontsize=6, ncol=2, loc="best",
               framealpha=0.5, markerscale=1.2)

    ax2 = fig.add_subplot(gs[0, 1])
    if do_tsne and len(X) > 5:
        perp = max(5, min(30, (len(X) - 1) // 3))
        tsne = TSNE(n_components=2, random_state=0, perplexity=perp,
                    init="pca", learning_rate="auto")
        XY_t = tsne.fit_transform(Xz)
        _scatter_by_cluster(ax2, XY_t, cids, f"t-SNE (perplexity={perp})")
    else:
        ax2.text(0.5, 0.5, "t-SNE skipped", ha="center", va="center")
        ax2.set_xticks([]); ax2.set_yticks([])

    # Heatmap: clusters (rows) x 20 joints (cols), z-scored centroids.
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(cent_z, aspect="auto", cmap="coolwarm",
                    vmin=-2.5, vmax=2.5)
    ax3.set_yticks(range(k)); ax3.set_yticklabels(uniq, fontsize=7)
    ax3.set_xticks(range(20))
    ax3.set_xticklabels([f"j{j}" for j in range(20)], fontsize=6, rotation=90)
    ax3.set_ylabel("cluster"); ax3.set_xlabel("joint channel")
    ax3.set_title("centroid heatmap (z-scored joint angles)", fontsize=11)
    fig.colorbar(im, ax=ax3, fraction=0.025, pad=0.01)

    # Per-cluster mean silhouette + size.
    ax4 = fig.add_subplot(gs[1, 1])
    sizes = [int((cids == c).sum()) for c in uniq]
    msil = [float(sil[cids == c].mean()) for c in uniq]
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(k)]
    ax4b = ax4.twinx()
    ax4.bar([i - 0.2 for i in range(k)], sizes, width=0.4,
            color=colors, alpha=0.85, label="size")
    ax4b.bar([i + 0.2 for i in range(k)], msil, width=0.4,
             color="gray", alpha=0.7, label="mean silhouette")
    ax4.set_xticks(range(k)); ax4.set_xticklabels(uniq, fontsize=7)
    ax4.set_xlabel("cluster"); ax4.set_ylabel("size (segments)")
    ax4b.set_ylabel("mean silhouette")
    ax4b.axhline(0, color="r", lw=0.6, ls="--")
    ax4.set_title(f"cluster size & silhouette "
                  f"(overall sil={sil.mean():.3f})", fontsize=11)

    fig.suptitle(f"{group}: {len(X)} segments, k={k} "
                 f"clusters (apex-pose feature space)", fontsize=13)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (overall silhouette={sil.mean():.3f})")


def main():
    ap = argparse.ArgumentParser(
        description="Plot clustering feature space per group")
    ap.add_argument("input_dir", help="folder of source .npz (stage-1 input)")
    ap.add_argument("--out", default="out_fgw")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--no-tsne", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompute features even if cache exists")
    args = ap.parse_args()

    clu = pd.read_csv(os.path.join(args.out, "segments_clustered.csv"))
    fmap_dir = os.path.join(args.out, "feature_maps")
    cache_dir = os.path.join(args.out, "feature_cache")
    os.makedirs(fmap_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    for group, gdf in clu.groupby("group"):
        cache = os.path.join(cache_dir, f"{group}.npz")
        if os.path.exists(cache) and not args.force:
            z = np.load(cache)
            X, cids = z["X"], z["cids"]
            print(f"[{group}] loaded cached features {X.shape}")
        else:
            print(f"[{group}] computing features ...")
            X, cids = _group_features(args.input_dir, gdf, args.fs)
            np.savez(cache, X=X, cids=cids)
        out_path = os.path.join(fmap_dir, f"{group}_features.png")
        plot_group(group, X, cids, out_path, do_tsne=not args.no_tsne)


if __name__ == "__main__":
    main()
