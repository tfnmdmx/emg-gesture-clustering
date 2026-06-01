from __future__ import annotations

import warnings

import numpy as np


def estimate_emg_pose_lag(env, pose_speed, fs: int,
                          max_lag_s: float = 1.0,
                          grid_hz: int = 100,
                          min_corr: float = 0.05) -> tuple[float, float]:
    """Cross-correlate EMG envelope vs pose speed -> lag in seconds.

    Both signals are downsampled to ``grid_hz``, z-scored (NaN-aware) and
    scanned over ``[-max_lag_s, +max_lag_s]`` for the lag of maximum
    correlation.

    Returns ``(lag_s, corr)``. Positive ``lag_s`` means **EMG leads pose
    speed** -- physiologically expected to fall in roughly ``[0, 0.4]`` s.

    Returns ``(NaN, corr)`` (where ``corr`` is the best value found, possibly
    0) when the input has too few finite samples or the best correlation
    falls below ``min_corr`` -- treat such recordings as "lag unknown"
    rather than mis-flag them. ``lag_status(NaN) == "nan"``.

    ``max_lag_s`` defaults to ``1.0`` (was ``3.0`` in the reference): tighter
    window keeps the QC's signal-to-noise high in the range we actually care
    about. Raise if you suspect a particular task has unusually long lags.
    """
    env = np.asarray(env, dtype=np.float64)
    spd = np.asarray(pose_speed, dtype=np.float64)
    if env.shape != spd.shape:
        raise ValueError(
            f"env / pose_speed length mismatch: {env.shape} vs {spd.shape}")
    if env.ndim != 1 or env.size < 10:
        return float("nan"), 0.0

    step = max(1, int(round(fs / grid_hz)))
    env_d = env[::step].astype(np.float64)
    spd_d = spd[::step].astype(np.float64)
    actual_hz = fs / step

    # All numeric work below assumes some NaNs may slip in (raw Manus loses
    # frames during occlusion). Replace NaNs with the per-signal mean AFTER
    # z-scoring so NaN samples contribute zero to the inner product without
    # warping the scale; warnings get muted because they fire benignly when
    # an all-NaN slice slips through (handled below as no-signal).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        env_m, env_s = float(np.nanmean(env_d)), float(np.nanstd(env_d))
        spd_m, spd_s = float(np.nanmean(spd_d)), float(np.nanstd(spd_d))
    if not (np.isfinite(env_m) and np.isfinite(spd_m)
            and env_s > 1e-12 and spd_s > 1e-12):
        return float("nan"), 0.0

    env_z = (env_d - env_m) / env_s
    spd_z = (spd_d - spd_m) / spd_s
    env_z = np.where(np.isfinite(env_z), env_z, 0.0)
    spd_z = np.where(np.isfinite(spd_z), spd_z, 0.0)

    max_lag_n = int(round(max_lag_s * actual_hz))
    best_lag_n, best_corr = 0, -np.inf
    # Convention: lag_n > 0 means EMG LEADS pose-speed by lag_n samples.
    for lag_n in range(-max_lag_n, max_lag_n + 1):
        if lag_n > 0:
            a, b = env_z[:-lag_n], spd_z[lag_n:]
        elif lag_n < 0:
            a, b = env_z[-lag_n:], spd_z[:lag_n]
        else:
            a, b = env_z, spd_z
        n = min(len(a), len(b))
        if n < 10:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c = float(np.mean(a[:n] * b[:n]))
        if np.isfinite(c) and c > best_corr:
            best_corr = c
            best_lag_n = lag_n

    if not np.isfinite(best_corr) or best_corr < min_corr:
        return float("nan"), float(max(best_corr, 0.0)
                                   if np.isfinite(best_corr) else 0.0)
    return float(best_lag_n / actual_hz), float(best_corr)


def lag_status(lag_s: float, ok_range: tuple[float, float] = (0.0, 0.4)) -> str:
    """Categorise a lag estimate for quick filtering.

    Returns one of:
      - ``"ok"``    -- lag in ``ok_range`` (default 0..0.4s, physiological)
      - ``"early"`` -- lag below ok_range (EMG fires AFTER motion; sync bug)
      - ``"late"``  -- lag above ok_range (large delay; sync drift or noise)
      - ``"nan"``   -- lag unreliable (too few finite samples or low corr)
    """
    if not np.isfinite(lag_s):
        return "nan"
    lo, hi = ok_range
    if lag_s < lo:
        return "early"
    if lag_s > hi:
        return "late"
    return "ok"
