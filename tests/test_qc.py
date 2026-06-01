import numpy as np

from emg_label.qc import estimate_emg_pose_lag, lag_status


def _signal_with_lag(fs, duration, lag_s, seed=0):
    """Build (env, pose_speed) where env leads pose_speed by exactly lag_s.

    Both signals are smoothed pulse trains so the peak of their cross-corr
    falls cleanly at the true lag.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    pulses = np.zeros(n)
    peaks = rng.integers(int(fs * 0.5), n - int(fs * 0.5), size=12)
    pulses[peaks] = 1.0
    # Convolve to make wide pulses.
    kernel = np.exp(-0.5 * (np.arange(-fs // 4, fs // 4) / (fs * 0.05)) ** 2)
    env = np.convolve(pulses, kernel, mode="same")
    # Pose speed = same pulses shifted later by lag_s.
    shift = int(round(lag_s * fs))
    pose = np.zeros(n)
    if shift > 0:
        pose[shift:] = env[:-shift]
    elif shift < 0:
        pose[:shift] = env[-shift:]
    else:
        pose = env.copy()
    # Add small noise to avoid trivial perfect correlation.
    env += rng.normal(0, 0.02, n)
    pose += rng.normal(0, 0.02, n)
    return env, pose


def test_lag_recovers_positive_emg_lead():
    # EMG leads pose by +120 ms (typical physiological value).
    env, spd = _signal_with_lag(fs=1000, duration=20.0, lag_s=0.120)
    lag, corr = estimate_emg_pose_lag(env, spd, fs=1000)
    assert abs(lag - 0.120) < 0.025  # within 25ms on a 10ms grid
    assert corr > 0.5


def test_lag_recovers_zero():
    env, spd = _signal_with_lag(fs=1000, duration=20.0, lag_s=0.0)
    lag, corr = estimate_emg_pose_lag(env, spd, fs=1000)
    assert abs(lag) < 0.025
    assert corr > 0.5


def test_lag_recovers_negative_lag():
    # Pose moves BEFORE EMG fires -- non-physiological. Detector still finds it.
    env, spd = _signal_with_lag(fs=1000, duration=20.0, lag_s=-0.150)
    lag, corr = estimate_emg_pose_lag(env, spd, fs=1000)
    assert abs(lag - (-0.150)) < 0.025
    assert lag_status(lag) == "early"


def test_lag_returns_nan_for_all_nan_input():
    # Real raw recordings sometimes have full-NaN slices (Manus occlusion).
    # The estimator must not return (0.0, -inf) and get mis-flagged "ok".
    n = 5000
    env = np.full(n, np.nan)
    spd = np.full(n, np.nan)
    lag, corr = estimate_emg_pose_lag(env, spd, fs=1000)
    assert np.isnan(lag)
    assert lag_status(lag) == "nan"


def test_lag_returns_nan_for_no_correlation():
    # Pure independent noise -> best cross-corr is near-zero, below the
    # min_corr threshold for any reasonable threshold relative to signal
    # length. Tightening min_corr to 0.2 here keeps the test stable across
    # signal lengths (real recordings are ~120s and would yield NaN at
    # the default min_corr=0.05 too).
    rng = np.random.default_rng(123)
    env = rng.normal(0, 1, 5000)
    spd = rng.normal(0, 1, 5000)
    lag, corr = estimate_emg_pose_lag(env, spd, fs=1000, min_corr=0.2)
    assert np.isnan(lag)
    assert abs(corr) < 0.2


def test_lag_robust_to_partial_nan():
    # 20% NaN holes scattered through both signals -- the true lag should
    # still be recovered (the finite portion carries the signal).
    env, spd = _signal_with_lag(fs=1000, duration=20.0, lag_s=0.100)
    rng = np.random.default_rng(7)
    mask = rng.random(len(env)) < 0.20
    env_nan = env.copy()
    spd_nan = spd.copy()
    env_nan[mask] = np.nan
    spd_nan[mask] = np.nan
    lag, corr = estimate_emg_pose_lag(env_nan, spd_nan, fs=1000)
    assert abs(lag - 0.100) < 0.05


def test_lag_status_buckets():
    assert lag_status(0.10) == "ok"      # mid-physiological
    assert lag_status(-0.05) == "early"  # EMG fires after motion
    assert lag_status(0.6) == "late"     # implausibly large lag
    assert lag_status(float("nan")) == "nan"
