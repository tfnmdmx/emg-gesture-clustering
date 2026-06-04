from __future__ import annotations

"""Stage 2.5: evaluate a pooled multi-subject clustering for subject-invariance.

The end goal is subject-INDEPENDENT gesture recognition: clusters should be
gestures, not people. Plain silhouette can't tell those apart (one person's
data is naturally tight, so optimizing silhouette can even reward clustering
by person). This script turns "does it generalize across people" into numbers,
using only the data we already have -- no gesture ground-truth required.

For each ``group`` in ``<out>/segments_clustered.csv`` it reports:

  1. Cluster quality   -- overall + weakest per-cluster silhouette.
  2. Subject contamination -- dominant-subject fraction per cluster and how many
     clusters are dominated by one person. Lower = better (closer to balanced).
  3. Subject LEAKAGE   -- accuracy of a classifier predicting *which subject*
     a segment came from, using the pose features (stratified 5-fold CV).
     Near the random baseline = features carry little identity = good.
     Well above baseline = features encode the person, not just the gesture.
  4. LOSO transfer     -- leave-one-subject-out: cluster on N-1 subjects, drop
     the held-out subject onto the nearest centroid, and measure how tightly
     the held-out poses fit the learned gesture structure (transfer ratio).
     Ratio near 1.0 = the held-out person lands as cleanly as training people.

Headline numbers to drive down over iterations: subject-leakage accuracy and
mean dominant-subject fraction, both toward the random baseline, while keeping
silhouette up and the LOSO transfer ratio near 1.0.

Usage:
    python evaluate.py <input_dir> [--out out_4users] [--fs 2000] [--force]
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from emg_label import features, io_utils


def _subject(source_file: str) -> str:
    return str(source_file).split("__")[0]


def _group_features(input_dir, gdf, fs):
    """Recompute (X, cluster_ids, subjects) for one group, aligned row-for-row.

    Mirrors cluster.py / plot_cluster_features.py exactly: reads each segment's
    precomputed hold-window end and takes the apex pose within it.
    """
    feats, cids, subs = [], [], []
    for fname, fdf in gdf.groupby("source_file"):
        sp = fdf["source_path"].iloc[0] if "source_path" in fdf.columns else None
        _, ja = io_utils.load_npz(
            io_utils.resolve_npz_path(fname, sp, input_dir))
        rest = np.median(ja, axis=0)
        fdf = fdf.sort_values("start_sample")
        subj = _subject(fname)
        for _, row in fdf.iterrows():
            s = int(row["start_sample"])
            he = int(row["hold_end_sample"])
            feats.append(features.apex_pose_feature(ja, s, he, rest, fs))
            cids.append(int(row["cluster_id"]))
            subs.append(subj)
        del ja
    return np.asarray(feats), np.asarray(cids), np.asarray(subs)


def _cluster_quality(Xz, cids):
    uniq = sorted(set(int(c) for c in cids))
    sil = silhouette_score(Xz, cids)
    ss = silhouette_samples(Xz, cids)
    per = {c: float(ss[cids == c].mean()) for c in uniq}
    weak = sorted(per.items(), key=lambda kv: kv[1])[:3]
    return sil, per, weak


def _contamination(cids, subs):
    """Per-cluster dominant-subject fraction (top subject's share of the cluster)."""
    uniq = sorted(set(int(c) for c in cids))
    n_subj = len(set(subs))
    rows, fracs = [], []
    for c in uniq:
        sc = subs[cids == c]
        vc = pd.Series(sc).value_counts()
        frac = float(vc.iloc[0] / vc.sum())
        fracs.append(frac)
        rows.append((c, int(vc.sum()), vc.index[0], frac))
    return rows, np.array(fracs), n_subj


def _subject_leakage(Xz, subs):
    """How well a classifier predicts subject from features. Lower = better."""
    n_subj = len(set(subs))
    if n_subj < 2:
        return None
    vc = pd.Series(subs).value_counts()
    majority = float(vc.iloc[0] / vc.sum())
    random_base = 1.0 / n_subj
    clf = LogisticRegression(max_iter=2000)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc = cross_val_score(clf, Xz, subs, cv=skf, scoring="accuracy").mean()
    return float(acc), random_base, majority


def _loso_transfer(X, cids, subs, random_state=0):
    """Leave-one-subject-out: train clusters on N-1, fit held-out subject.

    transfer ratio = (held-out mean dist to its assigned centroid) /
                     (train mean dist to assigned centroid), in train-z space.
    1.0 means the held-out person lands as tightly as the training people did;
    >>1 means their poses don't match the learned gesture clusters.
    """
    subjects = sorted(set(subs))
    if len(subjects) < 2:
        return None
    k = len(set(int(c) for c in cids))
    out = []
    for s in subjects:
        tr = subs != s
        te = subs == s
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Xtr)
        d_tr = np.linalg.norm(Xtr - km.cluster_centers_[km.labels_], axis=1)
        te_lab = km.predict(Xte)
        d_te = np.linalg.norm(Xte - km.cluster_centers_[te_lab], axis=1)
        ratio = float(d_te.mean() / (d_tr.mean() + 1e-12))
        out.append((s, int(te.sum()), ratio))
    return out


def evaluate_group(group, X, cids, subs, subject_norm="none"):
    # Measure the SAME representation that was clustered: optionally normalize
    # each subject in feature space before z-scoring (and before LOSO, where it
    # acts as test-time per-person normalization).
    Xc = features.apply_subject_norm(X, subs, subject_norm)
    Xz, _, _ = features.zscore(Xc)
    uniq = sorted(set(int(c) for c in cids))
    print(f"\n{'='*72}\nGROUP: {group}   N={len(X)}  k={len(uniq)}  "
          f"subjects={sorted(set(subs))}")

    sil, per, weak = _cluster_quality(Xz, cids)
    print(f"\n[1] CLUSTER QUALITY")
    print(f"    overall silhouette : {sil:.3f}   (>0.30 = reasonable structure)")
    print(f"    weakest clusters   : " +
          ", ".join(f"c{c}={v:.3f}" for c, v in weak))

    if len(set(subs)) < 2:
        print("\n    (single-subject group -- subject metrics N/A)")
        return {"group": group, "silhouette": sil, "n_subjects": 1}

    rows, fracs, n_subj = _contamination(cids, subs)
    heavy50 = int((fracs > 0.50).sum())
    heavy77 = int((fracs > 0.77).sum())
    print(f"\n[2] SUBJECT CONTAMINATION   (ideal dominant frac ~{1/n_subj:.2f})")
    print(f"    mean dominant-subject fraction : {fracs.mean():.3f}")
    print(f"    clusters >50% one subject      : {heavy50}/{len(uniq)}")
    print(f"    clusters >77% one subject      : {heavy77}/{len(uniq)}")
    worst = sorted(rows, key=lambda r: -r[3])[:5]
    print(f"    most contaminated: " +
          ", ".join(f"c{c}({_subject(sj).split('-')[0]} {fr:.0%})"
                    for c, _, sj, fr in worst))

    leak = _subject_leakage(Xz, subs)
    acc, rb, maj = leak
    print(f"\n[3] SUBJECT LEAKAGE   (lower = more subject-invariant)")
    print(f"    classifier predicts subject : {acc:.3f}")
    print(f"    random baseline             : {rb:.3f}")
    print(f"    majority-class baseline     : {maj:.3f}")
    print(f"    -> leakage above random     : +{acc-rb:.3f}  "
          f"({'HIGH' if acc-rb > 0.2 else 'moderate' if acc-rb > 0.1 else 'low'})")

    loso = _loso_transfer(Xc, cids, subs)
    ratios = [r[2] for r in loso]
    print(f"\n[4] LOSO TRANSFER   (ratio~1.0 = held-out person fits as tightly as train)")
    for s, n, ratio in loso:
        print(f"    hold out {s.split('-')[0]:>4} (n={n:>5}): transfer ratio={ratio:.2f}")
    print(f"    mean transfer ratio={np.mean(ratios):.2f}")

    return {
        "group": group, "silhouette": sil, "n_subjects": n_subj,
        "mean_dominant_frac": float(fracs.mean()),
        "heavy50": heavy50, "heavy77": heavy77,
        "subject_leak_acc": acc, "random_baseline": rb,
        "loso_transfer_ratio": float(np.mean(ratios)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate pooled clustering for subject-invariance")
    ap.add_argument("input_dir", nargs="?", default=None,
                    help="OPTIONAL legacy fallback dir of .npz. Omit it: npz are "
                         "located via each row's source_path (no work_pool needed).")
    ap.add_argument("--out", default="out_4users")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--force", action="store_true",
                    help="recompute features even if eval cache exists")
    ap.add_argument("--subject-norm", choices=["none", "center", "zscore"],
                    default="none",
                    help="evaluate the per-subject normalized representation "
                         "(match cluster.py --subject-norm)")
    args = ap.parse_args()

    clu = pd.read_csv(os.path.join(args.out, "segments_clustered.csv"))
    invalid = io_utils.load_invalid_recordings(args.out)
    if invalid:
        clu = clu[~clu["source_file"].astype(str).isin(invalid)] \
            .reset_index(drop=True)
    cache_dir = os.path.join(args.out, "eval_cache")
    os.makedirs(cache_dir, exist_ok=True)

    summary = []
    for group, gdf in clu.groupby("group"):
        cache = os.path.join(cache_dir, f"{group}.npz")
        if os.path.exists(cache) and not args.force:
            z = np.load(cache, allow_pickle=True)
            X, cids, subs = z["X"], z["cids"], z["subs"]
            print(f"[{group}] loaded cached features {X.shape}")
        else:
            print(f"[{group}] computing features ...")
            X, cids, subs = _group_features(args.input_dir, gdf, args.fs)
            np.savez(cache, X=X, cids=cids, subs=subs)
        summary.append(evaluate_group(group, X, cids, subs,
                                       subject_norm=args.subject_norm))

    print(f"\n{'='*72}\nSUMMARY")
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))
    out_csv = os.path.join(args.out, "eval_metrics.csv")
    sdf.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
