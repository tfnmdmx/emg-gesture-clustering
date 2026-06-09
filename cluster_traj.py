from __future__ import annotations

"""TRAJECTORY (time-series) clustering channel -- one of the two coexisting
clustering channels (the other is cluster.py, the static-apex channel).

Reads the clips from ``<out>/index.db`` (store.labeled_view, invalid excluded)
and builds ONE fixed-length joint-angle TRAJECTORY feature per clip over its
motion interval -- the whole movement participates, which is what makes this a
*dynamic*-gesture clusterer (vs cluster.py's single apex-pose frame). Then
z-score -> PCA -> MiniBatchKMeans (scales to the 200k+ clip range). Registers
the run like every channel: writes ``<out>/cluster_runs/{run_id}/`` (params.json
+ clusters.csv + gallery/) and the db (cluster_runs + cluster_assignments), keyed
on (source_file, clip_id) so evaluate.py can score it against the labels.

Feature representations (--repr):
  centered (default) joint-angle trajectory minus its per-clip mean pose --
           keeps the movement PATH + dynamics, drops the absolute pose offset
           (also reduces per-subject bias for later cross-subject runs).
  velocity d(joint-angle)/dt trajectory -- pure dynamics.
  raw      absolute joint-angle trajectory.

Per-joint weighting hook: --joint-weights w0,..,w19 scales each joint's columns
by sqrt(w) so the L2 distance respects which joints matter for the gestures.

sklearn is in both the emg2pose env and base conda; activate your env (or set
PY=) and run:
  python cluster_traj.py --out out
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

from emg_label import features, io_utils, select, store  # noqa: E402
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
    """5-keyframe strip (start->1/3->apex->2/3->end).

    Prefer raw Manus skeleton XYZ; fall back to emg2pose forward-kinematics on
    joint_angles when the npz has no skeleton -- processed data carries only
    ``emg`` + ``joint_angles`` (no ``manus_*_skeleton``), so the skeleton path
    would otherwise leave every cluster as '(no skeleton in source npz)'. The FK
    fallback is the same path the apex gallery / labelling UI use, so the hand
    still renders. Returns False only if neither source is usable."""
    src = str(row["source_path"])
    hand = (str(row.get("hand") or "") or None)
    cs, ce = int(row["start_sample"]), int(row["end_sample"])
    ms, me = int(row["motion_start_sample"]), int(row["motion_end_sample"])
    ap = int(row["apex_sample"])
    md = max(1, me - ms)
    idx = [cs, ms + md // 3, ap, ms + 2 * md // 3, max(cs, ce - 1)]
    titles = ["start", "1/3", "apex", "2/3", "end"]

    skel = io_utils.load_skeleton(src, hand=hand)
    if skel is not None:                                   # raw Manus XYZ
        idx = [min(max(0, i), len(skel) - 1) for i in idx]
        frames = normalize_skeleton(skel[idx])
        draw = draw_skeleton
    else:                                                  # FK from joint angles
        try:
            from emg_label.hand3d import angles_batch_to_landmarks, draw_hand
        except Exception:
            return False
        _, ja = io_utils.load_npz(src, hand=hand)
        if ja is None or len(ja) == 0:
            return False
        idx = [min(max(0, i), len(ja) - 1) for i in idx]
        side = hand or io_utils.parse_file_info(src).hand or "left"
        frames = angles_batch_to_landmarks(np.asarray(ja[idx], dtype=float), side=side)
        draw = draw_hand
    limits = axis_limits(frames)
    fig = plt.figure(figsize=(2.2 * 5, 2.5))
    for i, (fr, t) in enumerate(zip(frames, titles)):
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        draw(ax, fr)
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


def _clip_frames(row, n_frames):
    """(frames[N,J,3], draw_fn) sampled uniformly across the medoid clip
    [start,end) for animation -- skeleton XYZ if present, else emg2pose FK on
    joint_angles. Returns (None, None) if neither source is usable."""
    src = str(row["source_path"])
    hand = (str(row.get("hand") or "") or None)
    cs, ce = int(row["start_sample"]), int(row["end_sample"])
    n = max(2, int(n_frames))
    idx = np.linspace(cs, max(cs + 1, ce - 1), n).round().astype(int)
    skel = io_utils.load_skeleton(src, hand=hand)
    if skel is not None:
        idx = np.clip(idx, 0, len(skel) - 1)
        return normalize_skeleton(skel[idx]), draw_skeleton
    try:
        from emg_label.hand3d import angles_batch_to_landmarks, draw_hand
    except Exception:
        return None, None
    _, ja = io_utils.load_npz(src, hand=hand)
    if ja is None or len(ja) == 0:
        return None, None
    idx = np.clip(idx, 0, len(ja) - 1)
    side = hand or io_utils.parse_file_info(src).hand or "left"
    return angles_batch_to_landmarks(np.asarray(ja[idx], dtype=float), side=side), draw_hand


def render_medoid_gif(row, out_path, fs, n_frames=24, fps=12) -> bool:
    """Looping GIF of the medoid clip's whole motion (clearer than a 5-frame
    strip). Fixed camera + shared axis limits across frames so only the hand
    moves. Same skeleton-or-FK source as the strip. False if no usable source."""
    frames, draw = _clip_frames(row, n_frames)
    if frames is None:
        return False
    from PIL import Image
    limits = axis_limits(frames)
    fig = plt.figure(figsize=(3.2, 3.2))
    ax = fig.add_subplot(111, projection="3d")
    imgs = []
    for fr in frames:                                  # reuse one axes (cla per frame)
        ax.cla()
        draw(ax, fr)
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
        ax.view_init(elev=18, azim=-72)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        imgs.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(1, fps)), loop=0, optimize=True)
    return True


def cluster_gallery(clips_df, rows, labels, medoid_idx, gdir, fs,
                    anim=True, n_frames=24, fps=12):
    """Per-cluster medoid view + index.html, sorted by size, into gdir. anim=True
    -> a looping GIF of the medoid clip's motion (default; clearer than frames);
    anim=False -> a static 5-keyframe strip. medoid_idx maps each global cluster
    id to a positional index into `rows` (precomputed per group, since each
    group is clustered in its own PCA space)."""
    os.makedirs(gdir, exist_ok=True)
    ext = "gif" if anim else "png"
    entries = []
    for cid in sorted(c for c in set(int(x) for x in labels) if c >= 0):
        mask = labels == cid
        row = clips_df.loc[rows[medoid_idx[cid]]]
        fname = f"cluster{cid:02d}_n{int(mask.sum())}.{ext}"
        ok = (render_medoid_gif(row, os.path.join(gdir, fname), fs, n_frames, fps)
              if anim else render_medoid_strip(row, os.path.join(gdir, fname), fs))
        entries.append({"cid": cid, "size": int(mask.sum()),
                        "img": fname if ok else None, "row": row})
    entries.sort(key=lambda e: -e["size"])
    head = (f"medoid-clip motion GIF per cluster ({n_frames} frames @ {fps}fps)"
            if anim else "medoid keyframe strip per cluster "
            "(start &rarr; 1/3 &rarr; apex &rarr; 2/3 &rarr; end)")
    width = 360 if anim else 1000
    parts = ["<!doctype html><meta charset=utf-8><title>Trajectory clusters</title>",
             "<style>body{font-family:sans-serif;margin:16px}"
             "h2{font-size:14px;margin-top:18px} img{border:1px solid #ddd}"
             ".m{font-family:monospace;font-size:11px;color:#666}</style>",
             f"<h1>{len(entries)} trajectory clusters &mdash; {head}</h1>"]
    for e in entries:
        r = e["row"]
        parts.append(f"<h2>cluster {e['cid']} &mdash; {e['size']} clips</h2>")
        parts.append(f"<div class=m>medoid: {html.escape(str(r['source_file']))}"
                     f" clip {int(r['clip_id'])} &middot; motion "
                     f"{float(r['motion_duration_s']):.2f}s</div>")
        if e["img"]:
            parts.append(f"<img src='{e['img']}' width={width}>")
        else:
            parts.append("<div class=m>(no usable pose in source npz)</div>")
    with open(os.path.join(gdir, "index.html"), "w") as f:
        f.write("\n".join(parts))
    return entries


# ---------- grouping (mirrors cluster.py) -----------------------------------

def _remap_group(clips: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Rewrite the 'group' column to the requested granularity. subject-hand
    (default) keeps the per-recording '{subject}-{hand}' group as-is."""
    if group_by == "hand":
        clips["group"] = clips["group"].map(lambda g: str(g).rsplit("-", 1)[-1])
    elif group_by == "all":
        clips["group"] = "all"
    return clips


# ---------- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out", help="output dir (reads index.db)")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--L", type=int, default=32, help="resampled trajectory length")
    ap.add_argument("--repr", default="centered",
                    choices=["centered", "velocity", "raw"])
    ap.add_argument("--group-by", choices=["subject-hand", "hand", "all"],
                    default="subject-hand",
                    help="clustering granularity, same semantics as cluster.py: "
                         "subject-hand = cluster each person independently "
                         "(default); all = one pooled cross-subject clustering.")
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
    ap.add_argument("--gallery-static", action="store_true",
                    help="gallery as a 5-keyframe strip instead of the default "
                         "looping motion GIF")
    ap.add_argument("--gallery-frames", type=int, default=24,
                    help="frames in each cluster's motion GIF")
    ap.add_argument("--gallery-fps", type=int, default=12, help="GIF playback fps")
    select.add_select_args(ap)        # --subjects/--sessions/--dates/... data subset
    args = ap.parse_args()

    conn = store.connect(args.out)
    clips_df = store.labeled_view(conn)        # excludes invalid recordings/clips
    srcp = {r["source_file"]: r["source_path"]
            for r in conn.execute("SELECT source_file, source_path FROM recordings")}
    conn.close()
    if clips_df.empty:
        print(f"no clips in {store.db_path(args.out)}; run segment first")
        return
    clips_df, scope = select.select_clips(clips_df, args)   # subset of the shared db
    if clips_df.empty:
        print("no clips match the data selection "
              "(--subjects/--sessions/--dates/...); nothing to cluster")
        return
    # select_clips keeps the original (non-contiguous) index; build_features /
    # the iloc[rows] assignment below are positional, so re-base the index.
    clips_df = clips_df.reset_index(drop=True)
    clips_df["source_path"] = clips_df["source_file"].map(srcp)
    res = scope["resolved"]
    print(f"{len(clips_df)} clips from {clips_df['source_file'].nunique()} recordings "
          f"[scope: {res['n_subjects']} subj, {len(res['sessions'])} sessions, "
          f"dates {res['date_min']}..{res['date_max']}, tag={select.scope_tag(scope)}]")

    clips_df = _remap_group(clips_df, args.group_by)
    X, rows = build_features(clips_df, args.out, args.L, args.repr)

    if args.joint_weights:
        w = np.array([float(x) for x in args.joint_weights.split(",")], dtype=float)
        D = X.shape[1]
        njoint = w.size
        if D % njoint == 0:                       # tile sqrt(w) across time frames
            X = X * np.tile(np.sqrt(w), D // njoint)
        else:
            print(f"WARN joint-weights size {njoint} doesn't divide feature dim {D}; ignored")

    # Cluster each group independently (default subject-hand), mirroring
    # cluster.py: zscore->PCA->kmeans is run per group so one subject's pose
    # scale never biases another's, and a global cluster-id offset keeps ids
    # unique across groups (pooled clusters.csv has non-colliding ids). Use
    # --group-by all for a single pooled cross-subject clustering.
    row_groups = clips_df.iloc[rows]["group"].to_numpy()
    labels = np.full(len(rows), -1, dtype=int)     # -1 = group too small, dropped
    medoid_idx = {}                                # global cid -> position into rows
    n_comp_seen = []
    next_cid = 0
    for g in pd.unique(row_groups):
        sel = np.where(row_groups == g)[0]
        if len(sel) < 2:
            print(f"group {g}: <2 clips, skipped")
            continue
        Xz, _m, _s = features.zscore(X[sel])
        n_comp = min(args.pca, Xz.shape[1], Xz.shape[0] - 1)
        n_comp_seen.append(n_comp)
        Xp = PCA(n_components=n_comp, random_state=0).fit_transform(Xz)
        lbl, _km, best_k, _scores = cluster(
            Xp, args.k, args.k_min, args.k_max, args.sil_sample)
        local_ids = sorted(set(int(x) for x in lbl))
        remap = {c: next_cid + i for i, c in enumerate(local_ids)}  # -> global
        for c in local_ids:
            m = lbl == c
            members = sel[m]
            centroid = Xp[m].mean(axis=0)
            medoid_idx[remap[c]] = int(
                members[int(np.argmin(((Xp[m] - centroid) ** 2).sum(1)))])
        labels[sel] = np.array([remap[int(x)] for x in lbl])
        next_cid += len(local_ids)
        sil_g = silhouette_score(Xp, lbl) if len(local_ids) > 1 else float("nan")
        print(f"group {g}: {len(sel)} clips -> k={best_k}"
              + (f"  silhouette={sil_g:.3f}" if np.isfinite(sil_g) else ""))

    valid = labels >= 0
    if not valid.any():
        print("no group had >=2 clips; nothing clustered")
        return

    # Register the run: cluster_runs/{run_id}/ (params.json + clusters.csv) + db.
    assign = clips_df.iloc[rows[valid]][["source_file", "clip_id"]].copy()
    assign["cluster_id"] = labels[valid].astype(int)
    params = {"channel": "trajectory", "unit": "clip", "repr": args.repr,
              "L": args.L, "group_by": args.group_by,
              "pca": int(max(n_comp_seen)) if n_comp_seen else 0,
              "k": args.k, "k_min": args.k_min, "k_max": args.k_max,
              "n_clusters": int(next_cid), "scope": scope}
    ktag = str(args.k) if args.k is not None else f"{args.k_min}-{args.k_max}"
    label = f"traj__{select.scope_tag(scope)}__k{ktag}__{args.repr}"
    if args.group_by != "subject-hand":
        label += f"__{args.group_by}"
    run_id, created_at = store.new_run_id("trajectory", params, label=label)
    rundir = store.save_run(args.out, run_id, "trajectory", params, assign, created_at)
    sizes = sorted((int((labels[valid] == c).sum())
                    for c in set(labels[valid])), reverse=True)
    print(f"cluster sizes: {sizes}")
    print(f"trajectory run {run_id}: {len(assign)} clips, {next_cid} clusters -> {rundir}")

    if not args.no_gallery:
        cluster_gallery(clips_df, rows, labels, medoid_idx,
                        os.path.join(rundir, "gallery"), args.fs,
                        anim=not args.gallery_static,
                        n_frames=args.gallery_frames, fps=args.gallery_fps)
        print(f"wrote {os.path.join(rundir, 'gallery', 'index.html')}")


if __name__ == "__main__":
    main()
