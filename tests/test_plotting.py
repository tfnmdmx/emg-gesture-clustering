import os

import numpy as np

from emg_label.plotting import plot_overview, plot_cluster_preview


def test_plot_overview_creates_png(tmp_path):
    n = 2000
    emg = np.random.default_rng(0).normal(0, 1, size=(n, 16))
    activity = np.abs(np.random.default_rng(1).normal(0, 0.1, size=n))
    segs = [(400, 600), (1000, 1300)]
    out = tmp_path / "ov.png"
    plot_overview(emg, activity, segs, fs=1000, out_path=str(out),
                  enter_thr=0.5, exit_thr=0.3, labels=["a", "b"])
    assert os.path.getsize(out) > 0


def test_plot_cluster_preview_creates_png(tmp_path):
    centroids = [np.random.default_rng(i).normal(0, 1, size=20) for i in range(5)]
    counts = [3, 5, 2, 8, 4]
    ids = [0, 1, 2, 3, 4]
    out = tmp_path / "cl.png"
    plot_cluster_preview(centroids, counts, ids, str(out))
    assert os.path.getsize(out) > 0


def test_plot_cluster_hands_creates_png(tmp_path):
    from emg_label.plotting import plot_cluster_hands
    rng = np.random.default_rng(0)
    centroids = [rng.normal(0.3, 0.3, size=20) for _ in range(5)]
    counts = [10, 20, 5, 8, 3]
    ids = [0, 1, 2, 3, 4]
    out = tmp_path / "hands.png"
    plot_cluster_hands(centroids, counts, ids, str(out), side="left")
    assert os.path.getsize(out) > 0
