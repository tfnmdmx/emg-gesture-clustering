import numpy as np

from emg_label.features import (
    apex_index, apex_pose_feature, per_subject_center, per_subject_zscore,
    zscore)


def test_apex_pose_feature_picks_held_apex_not_movement():
    # rest ~ 0; early movement has small deviation, the late HOLD has the
    # large, distinctive deviation -> feature must capture the hold pose.
    rest = np.zeros(3)
    X = np.zeros((1000, 3))
    X[100:300] = np.array([0.3, 0.1, 0.0])    # early movement (small dev)
    X[400:800] = np.array([1.0, -1.0, 0.5])   # held apex (large dev)
    feat = apex_pose_feature(X, 0, 1000, rest, fs=1000, win_ms=20, smooth_ms=10)
    assert np.allclose(feat, [1.0, -1.0, 0.5], atol=0.05)


def test_apex_index_points_inside_held_apex():
    rest = np.zeros(2)
    X = np.zeros((1000, 2))
    X[400:800] = np.array([1.0, 1.0])          # held apex region
    ai = apex_index(X, 0, 1000, rest, fs=1000, smooth_ms=10)
    assert 400 <= ai < 800


def test_per_subject_center_removes_subject_offset():
    # two subjects, same gesture shape but different additive offset.
    base = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    A = base + np.array([10.0, 10.0])          # subject A offset
    B = base + np.array([-5.0, 3.0])           # subject B offset
    X = np.vstack([A, B])
    subs = np.array(["A", "A", "A", "B", "B", "B"])
    Xc = per_subject_center(X, subs)
    # after centering, both subjects collapse onto the same centered cloud
    assert np.allclose(Xc[:3], Xc[3:], atol=1e-9)


def test_per_subject_center_single_subject_is_global_demean():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    subs = np.array(["A", "A", "A"])
    Xc = per_subject_center(X, subs)
    assert np.allclose(Xc, X - X.mean(axis=0))


def test_per_subject_zscore_unit_variance_per_subject():
    rng = np.random.default_rng(0)
    A = rng.normal(5.0, 3.0, size=(50, 2))     # subject A: offset + wide scale
    B = rng.normal(-2.0, 0.5, size=(50, 2))    # subject B: offset + tight scale
    X = np.vstack([A, B])
    subs = np.array(["A"] * 50 + ["B"] * 50)
    Xz = per_subject_zscore(X, subs)
    # each subject becomes zero-mean, unit-variance on every axis
    assert np.allclose(Xz[:50].mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xz[50:].mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xz[:50].std(axis=0), 1.0, atol=1e-9)
    assert np.allclose(Xz[50:].std(axis=0), 1.0, atol=1e-9)


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


def test_apex_index_never_returns_nan_frame():
    # A leftover NaN frame (long/edge gap that survived interpolation) must never
    # be selected as the apex (nan_to_num(-inf) guard). The high-deviation hold
    # sits early, clear of the NaN's forward smoothing smear; plain np.argmax
    # would instead return the first NaN-deviation frame (~851).
    rest = np.zeros(3)
    X = np.zeros((1000, 3))
    X[100:140] = 5.0          # the real, finite apex region (early)
    X[900] = np.nan           # dropped frame late -> smear poisons only the tail
    a = apex_index(X, 0, len(X), rest, fs=2000)
    assert np.all(np.isfinite(X[a]))
    assert 60 <= a <= 160     # the finite hold, not the NaN-smear tail


def test_apex_pose_feature_finite_despite_nan():
    # nanmedian over the +/-win_ms window must skip a NaN frame so the feature
    # stays finite (plain np.median would yield NaN and poison zscore/KMeans).
    rest = np.zeros(3)
    X = np.tile([1.0, 2.0, 3.0], (40, 1))
    X[2] = np.nan
    feat = apex_pose_feature(X, 0, len(X), rest, fs=2000)
    assert np.all(np.isfinite(feat))
    assert np.allclose(feat, [1.0, 2.0, 3.0])
