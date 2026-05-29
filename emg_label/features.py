from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def apex_index(joint_angles, start, window_end, rest, fs, smooth_ms=50.0):
    """Absolute sample index of the gesture apex within ``[start, window_end)``.

    The apex is the frame of maximum (smoothed) joint deviation from ``rest`` --
    the instant the held gesture pose is most formed. Shared by
    ``apex_pose_feature`` (the clustering feature) and the stage-1 overview plot
    so both refer to the exact same frame.
    """
    X = np.asarray(joint_angles, dtype=float)
    rest = np.asarray(rest, dtype=float)
    win_end = min(int(window_end), len(X))
    seg = X[int(start):win_end]
    if len(seg) == 0:
        raise ValueError("empty window")
    sm = max(1, int(round(smooth_ms / 1000.0 * fs)))
    dev = uniform_filter1d(np.linalg.norm(seg - rest, axis=1), size=sm,
                           mode="nearest")
    return int(start) + int(np.argmax(dev))


def apex_pose_feature(joint_angles, start, window_end, rest, fs,
                      win_ms=50.0, smooth_ms=50.0):
    """Pose at the gesture apex within a gesture's hold window.

    With EMG-based segmentation a segment is the muscle *burst* (the hand
    moving into position); the distinctive, settled gesture pose occurs during
    the low-EMG *hold* after the burst. This finds the apex (max joint deviation
    from ``rest`` within ``[start, window_end)``, the hold window up to the next
    gesture's burst) and returns the median joint angles over a small +/-
    ``win_ms`` window around it.
    """
    X = np.asarray(joint_angles, dtype=float)
    rest = np.asarray(rest, dtype=float)
    win_end = min(int(window_end), len(X))
    seg = X[int(start):win_end]
    if len(seg) == 0:
        raise ValueError("empty window")
    apex = apex_index(X, start, window_end, rest, fs, smooth_ms) - int(start)
    half = max(1, int(round(win_ms / 1000.0 * fs)))
    a0 = max(0, apex - half)
    a1 = min(len(seg), apex + half + 1)
    return np.median(seg[a0:a1], axis=0)


def per_subject_center(X, subjects):
    """Subtract each subject's mean feature vector (feature-space de-mean).

    Removes the per-subject additive offset (1st moment) so pooled KMeans
    clusters by gesture rather than by person. No-op when all rows share one
    subject (it then equals the global mean that ``zscore`` removes anyway).
    Apply to the clustering features only -- keep the raw (absolute) pose for
    FK rendering.
    """
    X = np.asarray(X, dtype=float)
    subjects = np.asarray(subjects)
    out = X.copy()
    for s in np.unique(subjects):
        m = subjects == s
        out[m] = out[m] - out[m].mean(axis=0)
    return out


def per_subject_zscore(X, subjects):
    """Per-subject standardization: de-mean AND divide by each subject's std.

    Beyond ``per_subject_center``, this also removes each person's per-axis
    *scale* (diagonal 2nd moment) -- the gesture range/amplitude differences
    that survive de-meaning and let KMeans still split by person. No-op when a
    subject has a single sample on an axis (std guarded to 1).
    """
    X = np.asarray(X, dtype=float)
    subjects = np.asarray(subjects)
    out = X.copy()
    for s in np.unique(subjects):
        m = subjects == s
        mean = out[m].mean(axis=0)
        std = out[m].std(axis=0)
        std_safe = np.where(std < 1e-12, 1.0, std)
        out[m] = (out[m] - mean) / std_safe
    return out


def apply_subject_norm(X, subjects, mode):
    """Dispatch per-subject normalization: 'none' | 'center' | 'zscore'."""
    if mode == "center":
        return per_subject_center(X, subjects)
    if mode == "zscore":
        return per_subject_zscore(X, subjects)
    return np.asarray(X, dtype=float)


def zscore(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std_safe, mean, std
