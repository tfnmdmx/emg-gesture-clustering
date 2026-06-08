from __future__ import annotations

"""Stage 2 -- APEX clustering channel.

Clusters the **clip** unit (source_file, clip_id) on its static formed-pose
feature (the apex within the clip's hold). One run lands under
``OUT/cluster_runs/{run_id}/`` (params.json + clusters.csv + gallery/) and is
registered in ``OUT/index.db`` (cluster_runs + cluster_assignments), so it shares
the clip key with annotations -> label-driven evaluate.py is a single JOIN. The
trajectory (time-series) channel is cluster_traj.py, which writes the SAME
contract; both are first-class.
"""

import argparse
import os

import numpy as np
import pandas as pd

from emg_label import clustering, features, select, store
from emg_label.config import Config


def _remap_group(clips: pd.DataFrame, group_by: str) -> pd.DataFrame:
    if group_by == "hand":
        clips["group"] = clips["group"].map(lambda g: str(g).rsplit("-", 1)[-1])
    elif group_by == "all":
        clips["group"] = "all"
    return clips


def main():
    ap = argparse.ArgumentParser(description="Stage 2: apex clustering (per clip)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--k-min", type=int, default=12)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--k", type=int, default=None,
                    help="force a fixed cluster count (overrides --k-min/--k-max)")
    ap.add_argument("--group-by", choices=["subject-hand", "hand", "all"],
                    default="subject-hand",
                    help="clustering granularity (see docs/METHOD.md).")
    ap.add_argument("--subject-norm", choices=["none", "center", "zscore"],
                    default="none",
                    help="per-subject feature normalization for pooled runs.")
    ap.add_argument("--no-gallery", action="store_true",
                    help="skip the per-cluster 3D-hand gallery (needs torch FK).")
    select.add_select_args(ap)        # --subjects/--sessions/--dates/... data subset
    args = ap.parse_args()

    k_min, k_max = (args.k, args.k) if args.k is not None else (args.k_min, args.k_max)
    cfg = Config(fs=args.fs, out_dir=args.out, k_min=k_min, k_max=k_max)
    features_dir = os.path.join(cfg.out_dir, "features")

    conn = store.connect(cfg.out_dir)
    # labeled_view already drops invalid recordings + invalid clips; gesture_label
    # is ignored here (clustering is unsupervised).
    clips = store.labeled_view(conn)
    src_path = {r["source_file"]: r["source_path"]
                for r in conn.execute("SELECT source_file, source_path FROM recordings")}
    conn.close()
    if clips.empty:
        raise SystemExit(f"no clips in {store.db_path(cfg.out_dir)}; run segment first")
    clips, scope = select.select_clips(clips, args)   # subset of the shared db
    if clips.empty:
        raise SystemExit("no clips match the data selection "
                         "(--subjects/--sessions/--dates/...); nothing to cluster")
    res = scope["resolved"]
    print(f"data scope: {res['n_clips']} clips / {res['n_recordings']} recordings / "
          f"{res['n_subjects']} subject(s); sessions={len(res['sessions'])}; "
          f"dates {res['date_min']}..{res['date_max']}  [{select.scope_tag(scope)}]")
    clips = _remap_group(clips, args.group_by)

    rows = []            # per-clip: source_file, clip_id, cluster_id (global)
    gallery = []         # (group, centroids, counts, ids, side)
    cache_hit = cache_miss = 0
    next_cid = 0         # global cluster-id offset so ids are unique across groups

    for group, gdf in clips.groupby("group"):
        feats, keys, subs = [], [], []
        for sf, cdf in gdf.groupby("source_file"):
            cdf = cdf.sort_values("start_sample")
            fb, hit = features.clip_features_for(
                sf, cdf, cfg.fs, features_dir, source_path=src_path.get(sf))
            cache_hit += int(hit)
            cache_miss += int(not hit)
            subj = str(sf).split("__")[0]
            for _, r in cdf.iterrows():
                feats.append(fb[int(r["clip_id"])])
                keys.append((sf, int(r["clip_id"])))
                subs.append(subj)
        X = np.asarray(feats, dtype=float)
        finite = np.all(np.isfinite(X), axis=1)
        if not finite.all():
            print(f"group {group}: dropping {int((~finite).sum())} non-finite "
                  f"feature row(s)")
            X, keys, subs = X[finite], [k for k, f in zip(keys, finite) if f], \
                list(np.asarray(subs)[finite])
        if len(X) < 2:
            print(f"group {group}: <2 clips, skipped")
            continue
        Xc = features.apply_subject_norm(X, np.asarray(subs), args.subject_norm)
        Xz, _m, _s = features.zscore(Xc)
        labels, best_k = clustering.select_k_and_cluster(Xz, cfg.k_min, cfg.k_max)

        local_ids = sorted(set(int(x) for x in labels))
        remap = {c: next_cid + i for i, c in enumerate(local_ids)}  # -> global
        for (sf, cid), lab in zip(keys, labels):
            rows.append((sf, cid, remap[int(lab)]))
        centroids, counts, ids = [], [], []
        for c in local_ids:
            mask = labels == c
            centroids.append(X[mask].mean(axis=0))   # raw absolute pose
            counts.append(int(mask.sum()))
            ids.append(remap[c])
        side = group.rsplit("-", 1)[-1]
        gallery.append((group, centroids, counts, ids,
                        side if side in ("left", "right") else "left"))
        next_cid += len(local_ids)
        print(f"group {group}: {len(X)} clips -> k={best_k}")

    clusters_df = pd.DataFrame(rows, columns=["source_file", "clip_id", "cluster_id"])
    params = {"channel": "apex", "unit": "clip", "group_by": args.group_by,
              "subject_norm": args.subject_norm, "k": args.k,
              "k_min": k_min, "k_max": k_max, "n_clusters": int(next_cid),
              "scope": scope}
    ktag = str(args.k) if args.k is not None else f"{k_min}-{k_max}"
    label = f"apex__{select.scope_tag(scope)}__k{ktag}"
    if args.group_by != "subject-hand":
        label += f"__{args.group_by}"
    run_id, created_at = store.new_run_id("apex", params, label=label)
    rundir = store.save_run(cfg.out_dir, run_id, "apex", params, clusters_df, created_at)
    print(f"features cache: {cache_hit} hit, {cache_miss} miss")
    print(f"apex run {run_id}: {len(clusters_df)} clips, {next_cid} clusters -> {rundir}")

    if not args.no_gallery and gallery:
        _render_gallery(rundir, gallery)


def _render_gallery(rundir, gallery):
    """Per-cluster 3D-hand montage into rundir/gallery/ (needs torch FK)."""
    try:
        from emg_label import plotting
    except Exception as ex:                       # pragma: no cover
        print(f"gallery skipped ({ex})")
        return
    gdir = os.path.join(rundir, "gallery")
    os.makedirs(gdir, exist_ok=True)
    for group, centroids, counts, ids, side in gallery:
        try:
            plotting.plot_cluster_hands(
                centroids, counts, ids,
                os.path.join(gdir, f"{group}_hands.png"), side=side)
        except Exception as ex:                   # pragma: no cover
            print(f"gallery {group} skipped ({ex})")
            return


if __name__ == "__main__":
    main()
