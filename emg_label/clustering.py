from __future__ import annotations

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# silhouette_score builds an O(n^2) pairwise-distance matrix; on pooled runs
# (GROUP_BY=all) n can be 200k+, which would OOM/hang. Subsample the metric above
# this size (the labels still come from KMeans over ALL points; only the quality
# score is estimated on a random subsample, which is unbiased for this purpose).
_SIL_MAX = 10000


def select_k_and_cluster(X, k_min: int, k_max: int, random_state: int = 0):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int), 1
    sil_kw = ({} if n <= _SIL_MAX
              else {"sample_size": _SIL_MAX, "random_state": random_state})
    hi = min(k_max, n - 1)
    lo = max(2, min(k_min, hi))
    # A fixed-k request (k_min == k_max) is silently clamped to n-1 for small
    # groups; surface it so 'k=18' that became k=7 is not mistaken for the ask.
    if k_min == k_max and lo != k_min:
        warnings.warn(
            f"requested k={k_min} infeasible for n={n} sample(s); using k={lo}",
            stacklevel=2)
    best_k = lo
    best_score = -np.inf
    best_labels = None
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, **sil_kw)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    if best_labels is None:
        km = KMeans(n_clusters=lo, n_init=10, random_state=random_state)
        best_labels = km.fit_predict(X)
        best_k = lo
    return best_labels, best_k
