import numpy as np

from emg_label.features import apex_pose_feature, zscore


def test_apex_pose_feature_picks_held_apex_not_movement():
    # rest ~ 0; early movement has small deviation, the late HOLD has the
    # large, distinctive deviation -> feature must capture the hold pose.
    rest = np.zeros(3)
    X = np.zeros((1000, 3))
    X[100:300] = np.array([0.3, 0.1, 0.0])    # early movement (small dev)
    X[400:800] = np.array([1.0, -1.0, 0.5])   # held apex (large dev)
    feat = apex_pose_feature(X, 0, 1000, rest, fs=1000, win_ms=20, smooth_ms=10)
    assert np.allclose(feat, [1.0, -1.0, 0.5], atol=0.05)


def test_apex_pose_feature_respects_window_end():
    # a bigger deviation exists OUTSIDE the window; it must be ignored.
    rest = np.zeros(2)
    X = np.zeros((1000, 2))
    X[100:200] = np.array([0.5, 0.5])         # inside window [0,300)
    X[600:700] = np.array([2.0, 2.0])         # outside window
    feat = apex_pose_feature(X, 0, 300, rest, fs=1000, win_ms=10, smooth_ms=10)
    assert np.allclose(feat, [0.5, 0.5], atol=0.05)


def test_apex_pose_feature_clips_window_to_data_length():
    rest = np.zeros(2)
    X = np.zeros((100, 2))
    X[80:] = np.array([1.0, 1.0])
    feat = apex_pose_feature(X, 0, 10_000, rest, fs=1000, win_ms=10, smooth_ms=10)
    assert np.all(np.isfinite(feat))


def test_zscore_unit_variance():
    X = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    Xz, mean, std = zscore(X)
    assert np.allclose(Xz.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xz.std(axis=0), 1.0, atol=1e-9)


def test_zscore_constant_column_safe():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    Xz, mean, std = zscore(X)
    assert np.all(np.isfinite(Xz))
    assert np.allclose(Xz[:, 0], 0.0)
