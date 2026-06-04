from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from emg_label import clustering, features, io_utils, plotting
from emg_label.config import Config


def main():
    ap = argparse.ArgumentParser(description="Stage 2: cluster segments per group")
    ap.add_argument("input_dir", nargs="?", default=None,
                    help="OPTIONAL legacy fallback dir of .npz. Omit it: npz are "
                         "located via each row's source_path (no work_pool needed).")
    ap.add_argument("--out", default="out")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--k-min", type=int, default=12)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--k", type=int, default=None,
                    help="force a fixed cluster count (overrides --k-min/--k-max)")
    ap.add_argument("--group-by", choices=["subject-hand", "hand", "all"],
                    default="subject-hand",
                    help="clustering granularity: 'subject-hand' (default, one "
                         "KMeans per subject+hand), 'hand' (pool subjects by "
                         "left/right), or 'all' (one global KMeans over "
                         "everything). Note: pooling mixes calibration/mirror "
                         "differences, so KMeans may split by subject/hand "
                         "rather than gesture -- see docs/METHOD.md.")
    ap.add_argument("--subject-norm", choices=["none", "center", "zscore"],
                    default="none",
                    help="per-subject feature normalization for pooled "
                         "(--group-by all/hand) runs, to cut subject "
                         "contamination: 'center' subtracts each subject's mean "
                         "feature; 'zscore' also divides by each subject's std. "
                         "No-op for single-subject groups. Cluster centroids "
                         "still use the raw absolute pose for FK rendering.")
    args = ap.parse_args()

    k_min, k_max = (args.k, args.k) if args.k is not None else (args.k_min, args.k_max)
    cfg = Config(fs=args.fs, out_dir=args.out, k_min=k_min, k_max=k_max)
    seg_df = pd.read_csv(os.path.join(cfg.out_dir, "segments.csv"))
    invalid = io_utils.load_invalid_recordings(cfg.out_dir)
    if invalid:
        n0 = len(seg_df)
        seg_df = seg_df[~seg_df["source_file"].astype(str).isin(invalid)] \
            .reset_index(drop=True)
        print(f"excluded {len(invalid)} invalid recording(s): "
              f"{n0 - len(seg_df)} segments dropped")
    os.makedirs(os.path.join(cfg.out_dir, "clusters"), exist_ok=True)

    # Remap the grouping column. Everything downstream (clustered csv, label
    # template, export) keys off this 'group', so rewriting it here is enough.
    if args.group_by == "hand":
        seg_df["group"] = seg_df["group"].map(lambda g: str(g).rsplit("-", 1)[-1])
    elif args.group_by == "all":
        seg_df["group"] = "all"

    seg_df["cluster_id"] = -1
    template_rows = []
    features_dir = os.path.join(cfg.out_dir, "features")
    cache_hit = cache_miss = 0

    for group, gdf in seg_df.groupby("group"):
        feats, idxs, subs = [], [], []
        # Per-source feature extraction. Prefer the cache that segment.py
        # writes under out/features/{stem}.npz -- apex_pose_feature is a
        # deterministic function of (joint_angles, start, hold_end, rest, fs),
        # so cached and recomputed values are bit-identical.
        for fname, fdf in gdf.groupby("source_file"):
            subj = str(fname).split("__")[0]
            fdf = fdf.sort_values("start_sample")
            stem = str(fname)
            if stem.endswith(".npz"):
                stem = stem[:-4]
            cache_path = os.path.join(features_dir, stem + ".npz")
            if os.path.isfile(cache_path):
                d = np.load(cache_path)
                feat_by_seg = {int(k): v
                               for k, v in zip(d["seg_idx"], d["feature"])}
                missing = [int(r["seg_idx"]) for _, r in fdf.iterrows()
                           if int(r["seg_idx"]) not in feat_by_seg]
                if missing:
                    # Stale cache (e.g. params changed) -> recompute this file.
                    print(f"features stale for {fname} "
                          f"({len(missing)} seg_idx missing); recomputing")
                    feat_by_seg = None
                else:
                    cache_hit += 1
            else:
                feat_by_seg = None

            if feat_by_seg is None:
                sp = (fdf["source_path"].iloc[0]
                      if "source_path" in fdf.columns else None)
                _, ja = io_utils.load_npz(
                    io_utils.resolve_npz_path(fname, sp, args.input_dir))
                rest = np.median(ja, axis=0)
                feat_by_seg = {}
                for _, row in fdf.iterrows():
                    s = int(row["start_sample"])
                    he = int(row["hold_end_sample"])
                    feat_by_seg[int(row["seg_idx"])] = (
                        features.apex_pose_feature(ja, s, he, rest, cfg.fs))
                del ja
                cache_miss += 1

            for ridx, row in fdf.iterrows():
                feats.append(feat_by_seg[int(row["seg_idx"])])
                idxs.append(ridx)
                subs.append(subj)
        X = np.array(feats)
        # Cluster on per-subject normalized features when requested; keep raw X
        # for the centroids below (FK rendering needs absolute joint angles).
        Xc = features.apply_subject_norm(X, np.array(subs), args.subject_norm)
        Xz, _mean, _std = features.zscore(Xc)
        labels, best_k = clustering.select_k_and_cluster(Xz, cfg.k_min, cfg.k_max)
        for ridx, lab in zip(idxs, labels):
            seg_df.at[ridx, "cluster_id"] = int(lab)

        centroids, counts, ids = [], [], []
        for c in sorted(set(int(x) for x in labels)):
            mask = labels == c
            centroids.append(X[mask].mean(axis=0))
            counts.append(int(mask.sum()))
            ids.append(c)
            template_rows.append({"group": group, "cluster_id": c,
                                  "count": int(mask.sum()), "label": ""})
        png = os.path.join(cfg.out_dir, "clusters", f"{group}.png")
        plotting.plot_cluster_preview(centroids, counts, ids, png)
        side = group.rsplit("-", 1)[-1]
        if side not in ("left", "right"):
            side = "left"
        hands_png = os.path.join(cfg.out_dir, "clusters", f"{group}_hands.png")
        plotting.plot_cluster_hands(centroids, counts, ids, hands_png, side=side)
        print(f"group {group}: {len(gdf)} segments -> k={best_k}")

    seg_df.to_csv(os.path.join(cfg.out_dir, "segments_clustered.csv"), index=False)
    pd.DataFrame(template_rows).to_csv(
        os.path.join(cfg.out_dir, "labels_template.csv"), index=False)
    print(f"features cache: {cache_hit} hit, {cache_miss} miss")
    print("Wrote segments_clustered.csv and labels_template.csv")
    print(f"-> Fill the 'label' column in {cfg.out_dir}/labels_template.csv "
          f"and save as {cfg.out_dir}/labels.csv")


if __name__ == "__main__":
    main()
