from __future__ import annotations

"""Stage 2.5: evaluate ONE clustering run -- label-driven + subject-invariance.

Evaluates a run registered in ``OUT/index.db`` (apex or trajectory channel)
against the human clip labels in ``annotations`` -- the real "用打标评估聚类".
Because the clustering unit and the labelling unit are both the clip
``(source_file, clip_id)``, this is a single JOIN (store.clusters_with_labels).

Reports (written to ``cluster_runs/{run_id}/metrics.json``):

  A. LABEL-DRIVEN (on the labelled subset; needs >=2 labelled clips & >=2
     gesture classes): Adjusted Rand Index, Normalized Mutual Info, cluster
     purity, and per-cluster dominant-gesture fraction. NaN/None when too few
     labels (does NOT crash).
  B. SUBJECT CONTAMINATION -- dominant-subject fraction per cluster (no labels
     needed): clusters should be gestures, not people.
  C. POSE-SPACE QUALITY -- silhouette + subject-leakage + LOSO transfer computed
     on the apex pose representation (a generic pose-space probe; for the
     trajectory channel this is an auxiliary view, not its own feature space).

Usage:
    python evaluate.py --out out [--run <run_id>] [--fs 2000] [--subject-norm none]
    (--run defaults to the latest run in the db)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from emg_label import features, store


def _pose_features(out_dir, cw, fs):
    """Apex pose-feature matrix for a run's clips, aligned row-for-row with `cw`
    (a pose-space probe for silhouette/leakage/LOSO). Reads the OUT/features
    cache (clip-keyed); recomputes from the npz on miss. Torch-free."""
    features_dir = os.path.join(out_dir, "features")
    conn = store.connect(out_dir)
    srcp = {r["source_file"]: r["source_path"]
            for r in conn.execute("SELECT source_file, source_path FROM recordings")}
    feat = {}
    for sf, g in cw.groupby("source_file"):
        cdf = pd.DataFrame(store.clips_for_recording(conn, sf))  # has apex_sample
        if cdf.empty:
            continue
        fb, _ = features.clip_features_for(sf, cdf, fs, features_dir,
                                           source_path=srcp.get(sf))
        for cid in g["clip_id"]:
            if int(cid) in fb:
                feat[(sf, int(cid))] = fb[int(cid)]
    conn.close()
    return np.asarray([feat.get((r["source_file"], int(r["clip_id"])))
                       for _, r in cw.iterrows()
                       if (r["source_file"], int(r["clip_id"])) in feat], dtype=float)


def _label_metrics(cw) -> dict:
    """Label-driven cluster scores on the labelled subset of `cw`
    (gesture_label non-empty). None when too few labels/classes."""
    lab = cw[cw["gesture_label"].notna()
             & (cw["gesture_label"].astype(str).str.len() > 0)]
    n_lab, n_cls = len(lab), lab["gesture_label"].nunique()
    if n_lab < 2 or n_cls < 2:
        return {"n_labeled": int(n_lab), "n_classes": int(n_cls),
                "ari": None, "nmi": None, "purity": None}
    y = lab["gesture_label"].to_numpy()
    c = lab["cluster_id"].to_numpy()
    purity = sum(int((y[c == cid] == m).sum())
                 for cid in set(c)
                 for m in [pd.Series(y[c == cid]).value_counts().index[0]]) / n_lab
    return {"n_labeled": int(n_lab), "n_classes": int(n_cls),
            "ari": round(float(adjusted_rand_score(y, c)), 4),
            "nmi": round(float(normalized_mutual_info_score(y, c)), 4),
            "purity": round(float(purity), 4)}


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
    # StratifiedKFold needs every class populated >= n_splits; cap to the
    # smallest subject count and skip entirely when that is < 2.
    n_splits = min(5, int(vc.min()))
    if n_splits < 2:
        return None
    clf = LogisticRegression(max_iter=2000)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
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


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate one clustering run: label-driven + subject-invariance")
    ap.add_argument("--out", default="out")
    ap.add_argument("--run", default=None,
                    help="run_id under cluster_runs/ (default: latest in the db)")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--subject-norm", choices=["none", "center", "zscore"],
                    default="none",
                    help="per-subject feature norm for the pose-space probe "
                         "(match cluster.py --subject-norm)")
    args = ap.parse_args()

    conn = store.connect(args.out)
    run = args.run or store.latest_run(conn)
    if not run:
        raise SystemExit(f"no cluster runs in {store.db_path(args.out)}; "
                         f"run cluster.py / cluster_traj.py first")
    meta = conn.execute("SELECT channel, params_json FROM cluster_runs WHERE run_id=?",
                        (run,)).fetchone()
    channel = meta["channel"] if meta else "?"
    cw = store.clusters_with_labels(conn, run)
    conn.close()
    if cw.empty:
        raise SystemExit(f"run {run} has no assignments")
    cw["subject"] = cw["subject"].fillna("").astype(str)
    cids = cw["cluster_id"].to_numpy()
    subs = cw["subject"].to_numpy()
    uniq = sorted(set(int(c) for c in cids))
    print(f"{'='*72}\nRUN {run}  channel={channel}  N={len(cw)}  k={len(uniq)}")

    # --- A. label-driven (the point: 用打标评估聚类) -----------------------
    lab = _label_metrics(cw)
    print(f"\n[A] LABEL-DRIVEN  (labelled clips: {lab['n_labeled']}, "
          f"gestures: {lab['n_classes']})")
    if lab["ari"] is None:
        print("    too few labelled clips / classes -- ARI/NMI/purity = N/A")
    else:
        print(f"    ARI={lab['ari']}  NMI={lab['nmi']}  purity={lab['purity']}")

    # --- B. subject contamination (label-free) ----------------------------
    n_subj = len(set(subs) - {""})
    contam = {}
    if n_subj >= 2:
        rows, fracs, ns = _contamination(cids, subs)
        contam = {"mean_dominant_subject_frac": round(float(fracs.mean()), 4),
                  "heavy50": int((fracs > 0.50).sum()),
                  "heavy77": int((fracs > 0.77).sum()), "n_subjects": ns}
        print(f"\n[B] SUBJECT CONTAMINATION (ideal dominant frac ~{1/ns:.2f})")
        print(f"    mean dominant-subject fraction: {fracs.mean():.3f}  "
              f"heavy>50%: {contam['heavy50']}/{len(uniq)}")

    # --- C. pose-space quality (silhouette/leakage/LOSO; auxiliary) --------
    pose = {}
    X = _pose_features(args.out, cw, args.fs)
    if len(X) == len(cw) and len(uniq) >= 2 and len(X) > len(uniq):
        Xc = features.apply_subject_norm(X, subs, args.subject_norm)
        Xz, _, _ = features.zscore(Xc)
        # subsample the O(n^2) silhouette on large pooled runs (200k+ clips)
        sil_kw = ({} if len(Xz) <= 10000
                  else {"sample_size": 10000, "random_state": 0})
        sil = float(silhouette_score(Xz, cids, **sil_kw))
        pose["pose_silhouette"] = round(sil, 4)
        print(f"\n[C] POSE-SPACE (apex probe)  silhouette={sil:.3f}")
        if n_subj >= 2:
            leak = _subject_leakage(Xz, subs)
            if leak:
                pose["subject_leak_acc"] = round(leak[0], 4)
                pose["random_baseline"] = round(leak[1], 4)
                print(f"    subject-leakage acc={leak[0]:.3f} "
                      f"(random {leak[1]:.3f})")
            loso = _loso_transfer(Xc, cids, subs)
            if loso:
                pose["loso_transfer_ratio"] = round(float(np.mean([r[2] for r in loso])), 3)
                print(f"    LOSO transfer ratio={pose['loso_transfer_ratio']}")
    else:
        print("\n[C] POSE-SPACE: skipped (insufficient/aligned features)")

    metrics = {"run_id": run, "channel": channel, "n_clips": int(len(cw)),
               "k": len(uniq), "label_driven": lab,
               "subject_contamination": contam, "pose_space": pose}
    run_dir = os.path.join(args.out, "cluster_runs", run)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nWrote {os.path.join(run_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
