from __future__ import annotations

"""v1 dynamic-gesture TRAJECTORY clustering for peak-segmented clips.

Reads ``<out>/clips.csv`` (the peak-segment dynamic clips segment.py now writes)
and builds ONE fixed-length joint-angle TRAJECTORY feature per clip over its
motion interval -- the whole movement participates, which is what makes this a
*dynamic*-gesture clusterer (unlike the old cluster.py, which clusters a single
apex-pose frame). Then z-score -> PCA -> MiniBatchKMeans (scales to the 200k+
clip range). Writes:

  <out>/clips_clustered.csv     clips.csv + a cluster_id column
  <out>/cluster_summary.csv     per cluster: size + medoid clip reference
  <out>/clusters_traj/          per-cluster medoid 5-keyframe strip + index.html

so the result can be eyeballed and hand-labelled (fill gesture_label on the
medoid / a sample, then measure cluster purity).

Feature representations (--repr):
  centered (default) joint-angle trajectory minus its per-clip mean pose --
           keeps the movement PATH + dynamics, drops the absolute pose offset
           (also reduces per-subject bias for later cross-subject runs).
  velocity d(joint-angle)/dt trajectory -- pure dynamics.
  raw      absolute joint-angle trajectory.

Per-joint weighting hook: --joint-weights w0,..,w19 scales each joint's columns
by sqrt(w) so the L2 distance respects which joints matter for the gestures.

sklearn is in both the emg2pose env and base conda; run with either:
  /home/chenglin/anaconda3/envs/emg2pose/bin/python cluster_traj.py --out out_peak
"""

import argparse
import html
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from emg_label import features, io_utils  # noqa: E402
from emg_label.skeleton import (axis_limits, draw_skeleton,  # noqa: E402
                                normalize_skeleton)


# ---------- feature extraction ----------------------------------------------

def _fill_nan(col: np.ndarray) -> np.ndarray:
    """Linear-interp interior NaNs (Manus occlusion), edge-fill the ends."""
    m = np.isfinite(col)
    if m.all():
        return col
    if not m.any():
        return np.zeros_like(col)
    x = np.arange(len(col))
    return np.interp(x, x[m], col[m])


def clip_traj_feature(ja: np.ndarray, ms: int, me: int, L: int,
                      repr_: str) -> np.ndarray:
    """One clip's motion interval -> fixed-length (flattened) trajectory.

    Resampling to ``L`` frames normalises duration so a fast and a slow version
    of the same gesture line up; ``repr_`` then chooses what aspect to keep.
    """
    seg = np.asarray(ja[ms:me], dtype=float)
    if len(seg) < 2:
        base = ja[ms:ms + 1] if 0 <= ms < len(ja) else np.zeros((1, ja.shape[1]))
        seg = np.repeat(np.asarray(base, float), 2, axis=0)
    D = seg.shape[1]
    src = np.linspace(0.0, 1.0, len(seg))
    dst = np.linspace(0.0, 1.0, L)
    res = np.empty((L, D), dtype=float)
    for d in range(D):
        res[:, d] = np.interp(dst, src, _fill_nan(seg[:, d]))
    if repr_ == "centered":
        res = res - res.mean(axis=0, keepdims=True)
    elif repr_ == "velocity":
        res = np.diff(res, axis=0)
    # 'raw' -> as-is
    return res.reshape(-1)


def build_features(clips_df: pd.DataFrame, out_dir: str, L: int, repr_: str):
    """Per-clip trajectory features, cached per source_file under traj_features/.

    Returns (X[n,D], rows[n] row-index into clips_df). The cache key encodes
    (repr, L) so re-running with different params recomputes cleanly while the
    same params reload instantly.
    """
    feat_dir = os.path.join(out_dir, "traj_features")
    os.makedirs(feat_dir, exist_ok=True)
    feats, rows = [], []
    hit = miss = 0
    for stem_npz, fdf in clips_df.groupby("source_file"):
        stem = str(stem_npz)[:-4] if str(stem_npz).endswith(".npz") else str(stem_npz)
        cache = os.path.join(feat_dir, f"{stem}.{repr_}_L{L}.npz")
        want = {int(c) for c in fdf["clip_id"]}
        feat_by_clip = None
        if os.path.isfile(cache):
            d = np.load(cache)
            fbc = {int(k): v for k, v in zip(d["clip_id"], d["feature"])}
            if want.issubset(fbc):
                feat_by_clip = fbc
                hit += 1
        if feat_by_clip is None:
            src = str(fdf.iloc[0]["source_path"])
            hand = str(fdf.iloc[0].get("hand") or "") or None
            _, ja = io_utils.load_npz(src, hand=hand)
            feat_by_clip = {}
            for _, r in fdf.iterrows():
                feat_by_clip[int(r["clip_id"])] = clip_traj_feature(
                    ja, int(r["motion_start_sample"]),
                    int(r["motion_end_sample"]), L, repr_)
            del ja
            ids = sorted(feat_by_clip)
            np.savez(cache, clip_id=np.array(ids, dtype=np.int64),
                     feature=np.stack([feat_by_clip[i] for i in ids]).astype(np.float32))
            miss += 1
        for ridx, r in fdf.iterrows():
            feats.append(feat_by_clip[int(r["clip_id"])])
            rows.append(ridx)
    print(f"features: {hit} files cached, {miss} computed; {len(feats)} clips")
    return np.asarray(feats, dtype=float), np.asarray(rows)


# ---------- clustering ------------------------------------------------------

def cluster(Xp, k, k_min, k_max, sil_sample, seed=0):
    """MiniBatchKMeans; if k is None, pick k by silhouette on a subsample."""
    n = Xp.shape[0]
    rng = np.random.default_rng(seed)

    def fit(kk):
        km = MiniBatchKMeans(n_clusters=kk, random_state=seed, n_init=3,
                             batch_size=min(4096, n))
        return km.fit_predict(Xp), km

    if k is not None:
        labels, km = fit(k)
        return labels, km, k, None
    idx = (rng.choice(n, sil_sample, replace=False) if n > sil_sample
           else np.arange(n))
    best = (-np.inf, None, None, None)
    scores = {}
    for kk in range(max(2, k_min), min(k_max, n - 1) + 1):
        labels, km = fit(kk)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(Xp[idx], labels[idx])
        scores[kk] = s
        if s > best[0]:
            best = (s, labels, km, kk)
    return best[1], best[2], best[3], scores


# ---------- per-cluster medoid keyframe gallery -----------------------------

def render_medoid_strip(row, out_path, fs) -> bool:
    """5-keyframe trajectory strip (start->1/3->apex->2/3->end) from raw Manus
    skeleton XYZ. Returns False if the recording has no skeleton."""
    skel = io_utils.load_skeleton(str(row["source_path"]),
                                  hand=(str(row.get("hand") or "") or None))
    if skel is None:
        return False
    cs, ce = int(row["clip_start_sample"]), int(row["clip_end_sample"])
    ms, me = int(row["motion_start_sample"]), int(row["motion_end_sample"])
    ap = int(row["apex_sample"])
    md = max(1, me - ms)
    idx = [cs, ms + md // 3, ap, ms + 2 * md // 3, max(cs, ce - 1)]
    idx = [min(max(0, i), len(skel) - 1) for i in idx]
    titles = ["start", "1/3", "apex", "2/3", "end"]
    frames = normalize_skeleton(skel[idx])
    limits = axis_limits(frames)
    fig = plt.figure(figsize=(2.2 * 5, 2.5))
    for i, (fr, t) in enumerate(zip(frames, titles)):
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        draw_skeleton(ax, fr)
        ax.set_title(t, fontsize=8)
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
        ax.view_init(elev=18, azim=-72)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return True


def cluster_gallery(clips_df, rows, labels, Xp, out_dir, fs):
    """One medoid strip per cluster + an index.html, sorted by size."""
    gdir = os.path.join(out_dir, "clusters_traj")
    os.makedirs(gdir, exist_ok=True)
    entries = []
    for cid in sorted(set(int(x) for x in labels)):
        mask = labels == cid
        members = np.where(mask)[0]
        centroid = Xp[members].mean(axis=0)
        medoid = members[int(np.argmin(((Xp[members] - centroid) ** 2).sum(1)))]
        row = clips_df.loc[rows[medoid]]
        png = f"cluster{cid:02d}_n{int(mask.sum())}.png"
        ok = render_medoid_strip(row, os.path.join(gdir, png), fs)
        entries.append({"cid": cid, "size": int(mask.sum()),
                        "png": png if ok else None, "row": row})
    entries.sort(key=lambda e: -e["size"])
    parts = ["<!doctype html><meta charset=utf-8><title>Trajectory clusters</title>",
             "<style>body{font-family:sans-serif;margin:16px}"
             "h2{font-size:14px;margin-top:18px} img{border:1px solid #ddd}"
             ".m{font-family:monospace;font-size:11px;color:#666}</style>",
             f"<h1>{len(entries)} trajectory clusters &mdash; medoid keyframe "
             f"strip per cluster (start &rarr; 1/3 &rarr; apex &rarr; 2/3 &rarr; "
             f"end)</h1>"]
    for e in entries:
        r = e["row"]
        parts.append(f"<h2>cluster {e['cid']} &mdash; {e['size']} clips</h2>")
        parts.append(f"<div class=m>medoid: {html.escape(str(r['source_file']))}"
                     f" clip {int(r['clip_id'])} &middot; motion "
                     f"{float(r['motion_duration_s']):.2f}s</div>")
        if e["png"]:
            parts.append(f"<img src='{e['png']}' width=1000>")
        else:
            parts.append("<div class=m>(no skeleton in source npz)</div>")
    with open(os.path.join(gdir, "index.html"), "w") as f:
        f.write("\n".join(parts))
    return entries


# ---------- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out_peak", help="dir with clips.csv")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--L", type=int, default=32, help="resampled trajectory length")
    ap.add_argument("--repr", default="centered",
                    choices=["centered", "velocity", "raw"])
    ap.add_argument("--pca", type=int, default=30, help="PCA components")
    ap.add_argument("--k", type=int, default=None,
                    help="fixed cluster count; omit to scan k-min..k-max by silhouette")
    ap.add_argument("--k-min", type=int, default=12)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--sil-sample", type=int, default=3000,
                    help="subsample size for silhouette during k scan")
    ap.add_argument("--joint-weights", default=None,
                    help="comma-separated 20 weights; columns scaled by sqrt(w)")
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()

    clips_path = os.path.join(args.out, "clips.csv")
    clips_df = pd.read_csv(clips_path, dtype={"source_file": str})
    if clips_df.empty:
        print(f"no clips in {clips_path}")
        return
    print(f"{len(clips_df)} clips from {clips_df['source_file'].nunique()} recordings")

    X, rows = build_features(clips_df, args.out, args.L, args.repr)

    if args.joint_weights:
        w = np.array([float(x) for x in args.joint_weights.split(",")], dtype=float)
        D = X.shape[1]
        njoint = w.size
        if D % njoint == 0:                       # tile sqrt(w) across time frames
            X = X * np.tile(np.sqrt(w), D // njoint)
        else:
            print(f"WARN joint-weights size {njoint} doesn't divide feature dim {D}; ignored")

    Xz, _mean, _std = features.zscore(X)
    n_comp = min(args.pca, Xz.shape[1], Xz.shape[0] - 1)
    Xp = PCA(n_components=n_comp, random_state=0).fit_transform(Xz)

    labels, _km, best_k, scores = cluster(
        Xp, args.k, args.k_min, args.k_max, args.sil_sample)
    sil = silhouette_score(Xp, labels) if len(set(labels)) > 1 else float("nan")
    if scores:
        print("silhouette by k:",
              {k: round(v, 3) for k, v in sorted(scores.items())})
    print(f"k={best_k}  silhouette={sil:.3f}  "
          f"(repr={args.repr}, L={args.L}, pca={n_comp})")

    # clips_clustered.csv = clips.csv + cluster_id
    clips_df["cluster_id"] = -1
    clips_df.loc[rows, "cluster_id"] = labels.astype(int)
    out_csv = os.path.join(args.out, "clips_clustered.csv")
    clips_df.to_csv(out_csv, index=False)

    # cluster_summary.csv
    summ = []
    for cid in sorted(set(int(x) for x in labels)):
        mask = labels == cid
        members = np.where(mask)[0]
        centroid = Xp[members].mean(axis=0)
        medoid = members[int(np.argmin(((Xp[members] - centroid) ** 2).sum(1)))]
        mr = clips_df.loc[rows[medoid]]
        summ.append({"cluster_id": cid, "size": int(mask.sum()),
                     "medoid_source_file": mr["source_file"],
                     "medoid_clip_id": int(mr["clip_id"]),
                     "medoid_motion_s": round(float(mr["motion_duration_s"]), 3),
                     "label": ""})
    pd.DataFrame(summ).sort_values("size", ascending=False).to_csv(
        os.path.join(args.out, "cluster_summary.csv"), index=False)

    print(f"cluster sizes: "
          f"{sorted((int((labels==c).sum()) for c in set(labels)), reverse=True)}")
    print(f"wrote {out_csv} and cluster_summary.csv")

    if not args.no_gallery:
        cluster_gallery(clips_df, rows, labels, Xp, args.out, args.fs)
        print(f"wrote {os.path.join(args.out, 'clusters_traj', 'index.html')}")


if __name__ == "__main__":
    main()
