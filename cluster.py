from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from emg_label import clustering, features, io_utils, plotting
from emg_label.config import Config


def main():
    ap = argparse.ArgumentParser(description="Stage 2: cluster segments per group")
    ap.add_argument("input_dir", help="same folder of .npz used in stage 1")
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
    os.makedirs(os.path.join(cfg.out_dir, "clusters"), exist_ok=True)

    # Remap the grouping column. Everything downstream (clustered csv, label
    # template, export) keys off this 'group', so rewriting it here is enough.
    if args.group_by == "hand":
        seg_df["group"] = seg_df["group"].map(lambda g: str(g).rsplit("-", 1)[-1])
    elif args.group_by == "all":
        seg_df["group"] = "all"

    seg_df["cluster_id"] = -1
    template_rows = []

    for group, gdf in seg_df.groupby("group"):
        feats, idxs, subs = [], [], []
        # Load each source file once, extract all its segments' features,
        # then release it -- keeps memory flat across large file sets.
        for fname, fdf in gdf.groupby("source_file"):
            _, ja = io_utils.load_npz(os.path.join(args.input_dir, fname))
            rest = np.median(ja, axis=0)
            subj = str(fname).split("__")[0]
            # The distinctive held pose is the apex of joint deviation in the
            # hold window after the burst, up to the next segment's start.
            fdf = fdf.sort_values("start_sample")
            starts = [int(v) for v in fdf["start_sample"].tolist()]
            ends = [int(v) for v in fdf["end_sample"].tolist()]
            order = list(fdf.index)
            for j in range(len(order)):
                s = starts[j]
                win_end = (starts[j + 1] if j + 1 < len(starts)
                           else min(len(ja), ends[j] + cfg.fs))
                feats.append(features.apex_pose_feature(
                    ja, s, win_end, rest, cfg.fs))
                idxs.append(order[j])
                subs.append(subj)
            del ja
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
    print("Wrote segments_clustered.csv and labels_template.csv")
    print(f"-> Fill the 'label' column in {cfg.out_dir}/labels_template.csv "
          f"and save as {cfg.out_dir}/labels.csv")


if __name__ == "__main__":
    main()
