from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import io_utils


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
    # NaN frames (long/edge Manus dropouts left unfilled) must never win argmax.
    return int(start) + int(np.argmax(np.nan_to_num(dev, nan=-np.inf)))


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
    # nan-aware: a few dropped frames in the +/-win_ms window must not poison the
    # whole feature vector (would propagate NaN into zscore/KMeans).
    return np.nanmedian(seg[a0:a1], axis=0)


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


def _read_feature_cache(features_dir, source_file, fdf):
    """{seg_idx: feature} from <features_dir>/{stem}.npz, or None if the cache is
    absent or missing any seg_idx in `fdf` (stale -> recompute)."""
    if not features_dir:
        return None
    stem = str(source_file)
    if stem.endswith(".npz"):
        stem = stem[:-4]
    cache_path = os.path.join(features_dir, stem + ".npz")
    if not os.path.isfile(cache_path):
        return None
    d = np.load(cache_path)
    feat_by_seg = {int(k): v for k, v in zip(d["seg_idx"], d["feature"])}
    missing = [int(r["seg_idx"]) for _, r in fdf.iterrows()
               if int(r["seg_idx"]) not in feat_by_seg]
    if missing:
        print(f"features stale for {source_file} "
              f"({len(missing)} seg_idx missing); recomputing")
        return None
    return feat_by_seg


def feature_by_seg(source_file, fdf, fs, features_dir=None, input_dir=None):
    """``{seg_idx: apex_pose_feature}`` for one recording's segment rows.

    The single shared apex-feature extractor for cluster.py,
    plot_cluster_features.py and evaluate.py. Prefers the per-recording cache
    segment.py writes to ``<features_dir>/{stem}.npz`` (a deterministic function
    of the inputs, so cached and recomputed values are bit-identical); otherwise
    loads the npz and recomputes with the SAME nan-aware
    ``rest = nanmedian(joint_angles)`` the cache writer uses. Routing every
    consumer through here keeps rest/cache identical so each scores exactly the
    features that were clustered. `fdf` needs columns seg_idx / start_sample /
    hold_end_sample (+ optional source_path). Returns ``(feat_by_seg, cache_hit)``.
    """
    feat_by_seg = _read_feature_cache(features_dir, source_file, fdf)
    if feat_by_seg is not None:
        return feat_by_seg, True
    sp = fdf["source_path"].iloc[0] if "source_path" in fdf.columns else None
    _, ja = io_utils.load_npz(io_utils.resolve_npz_path(source_file, sp, input_dir))
    rest = np.nanmedian(ja, axis=0)
    feat_by_seg = {}
    for _, row in fdf.iterrows():
        feat_by_seg[int(row["seg_idx"])] = apex_pose_feature(
            ja, int(row["start_sample"]), int(row["hold_end_sample"]), rest, fs)
    return feat_by_seg, False
