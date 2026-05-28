from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def apex_pose_feature(joint_angles, start, window_end, rest, fs,
                      win_ms=50.0, smooth_ms=50.0):
    """Pose at the gesture apex within a gesture's hold window.

    With EMG-based segmentation a segment is the muscle *burst* (the hand
    moving into position); the distinctive, settled gesture pose occurs during
    the low-EMG *hold* after the burst. This finds the frame of maximum joint
    deviation from ``rest`` within ``[start, window_end)`` (the hold window, up
    to the next gesture's burst) and returns the median joint angles over a
    small +/- ``win_ms`` window around it.
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
    apex = int(np.argmax(dev))
    half = max(1, int(round(win_ms / 1000.0 * fs)))
    a0 = max(0, apex - half)
    a1 = min(len(seg), apex + half + 1)
    return np.median(seg[a0:a1], axis=0)


def zscore(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std_safe, mean, std
