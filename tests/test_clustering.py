import warnings

import numpy as np

from emg_label.clustering import select_k_and_cluster


def _three_blobs(per=15):
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 10], [0, 10]], dtype=float)
    X = np.vstack([c + rng.normal(0, 0.3, size=(per, 2)) for c in centers])
    return X


def test_selects_correct_k_for_three_blobs():
    X = _three_blobs()
    labels, best_k = select_k_and_cluster(X, k_min=2, k_max=8)
    assert best_k == 3
    # each true blob maps to a single cluster
    for i in range(3):
        block = labels[i * 15:(i + 1) * 15]
        assert len(set(block)) == 1


def test_handles_single_sample():
    X = np.array([[1.0, 2.0]])
    labels, best_k = select_k_and_cluster(X, k_min=2, k_max=8)
    assert best_k == 1
    assert labels.tolist() == [0]


def test_clamps_k_to_sample_count():
    X = _three_blobs(per=2)  # 6 samples, k_max=30 must clamp
    labels, best_k = select_k_and_cluster(X, k_min=12, k_max=30)
    assert best_k <= 5
    assert len(labels) == 6


def test_fixed_k_clamp_warns():
    # A fixed k (k_min == k_max) that can't fit the sample count is clamped;
    # that must surface as a warning, not silently produce a different k.
    X = _three_blobs(per=2)  # 6 samples -> max feasible k is 5
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _labels, best_k = select_k_and_cluster(X, k_min=18, k_max=18)
    assert best_k < 18
    assert any("infeasible" in str(w.message) for w in caught)


def test_fixed_k_no_warn_when_feasible():
    X = _three_blobs(per=15)  # 45 samples, k=3 is feasible
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        select_k_and_cluster(X, k_min=3, k_max=3)
    assert not any("infeasible" in str(w.message) for w in caught)
