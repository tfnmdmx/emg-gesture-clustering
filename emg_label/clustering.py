from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def select_k_and_cluster(X, k_min: int, k_max: int, random_state: int = 0):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int), 1
    hi = min(k_max, n - 1)
    lo = max(2, min(k_min, hi))
    best_k = lo
    best_score = -np.inf
    best_labels = None
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    if best_labels is None:
        km = KMeans(n_clusters=lo, n_init=10, random_state=random_state)
        best_labels = km.fit_predict(X)
        best_k = lo
    return best_labels, best_k
