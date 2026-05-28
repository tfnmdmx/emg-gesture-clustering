from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def emg_envelope(emg, fs: int, smooth_ms: float = 150.0):
    """Smoothed rectified EMG envelope (mean |emg| across channels).

    This is the action/rest signal: it rises during a gesture's muscle
    activity and falls back to baseline when the muscle relaxes between
    gestures -- even when the hand pose has not yet returned to neutral.
    That makes it separate consecutive non-neutral gestures cleanly, which
    a joint-angle distance-from-rest signal cannot.
    """
    e = np.asarray(emg, dtype=float)
    rect = np.abs(e).mean(axis=1)
    win = max(1, int(round(smooth_ms / 1000.0 * fs)))
    return uniform_filter1d(rect, size=win, mode="nearest")


def _otsu_threshold(x, nbins: int = 256) -> float:
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return hi
    hist, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    p = hist.astype(float) / hist.sum()
    omega = np.cumsum(p)
    centers = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    return float(centers[int(np.argmax(sigma_b2))])


def auto_thresholds(activity):
    """Bimodal (Otsu) split between rest and action activity levels.

    Robust to how much of the recording is action: enter = Otsu valley,
    exit = halfway between the rest-mode center and the valley (so the
    detector exits cleanly back into rest without flickering).
    """
    a = np.asarray(activity, dtype=float)
    t = _otsu_threshold(a)
    rest = a[a <= t]
    rest_center = float(np.median(rest)) if rest.size else float(a.min())
    enter = float(t)
    exit_thr = float(rest_center + 0.5 * (t - rest_center))
    return enter, exit_thr


def hysteresis_segments(activity, enter_thr: float, exit_thr: float):
    a = np.asarray(activity, dtype=float)
    segments: list[tuple[int, int]] = []
    in_action = False
    start = 0
    for i, v in enumerate(a):
        if not in_action and v >= enter_thr:
            in_action = True
            start = i
        elif in_action and v < exit_thr:
            segments.append((start, i))  # end exclusive
            in_action = False
    if in_action:
        segments.append((start, len(a)))
    return segments


def filter_segments(segments, fs: int, min_action_s: float = 0.4,
                    min_rest_gap_s: float = 0.2):
    if not segments:
        return []
    min_action = int(round(min_action_s * fs))
    min_gap = int(round(min_rest_gap_s * fs))
    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if (e - s) >= min_action]


def segment_emg(emg, config):
    """Detect action segments from the EMG envelope.

    Returns (segments, activity, enter_thr, exit_thr) where ``activity`` is
    the smoothed EMG envelope used for detection.
    """
    activity = emg_envelope(emg, config.fs, config.smooth_ms)
    if config.enter_thresh is not None and config.exit_thresh is not None:
        enter, exit_thr = config.enter_thresh, config.exit_thresh
    else:
        enter, exit_thr = auto_thresholds(activity)
    if enter <= exit_thr:
        # Degenerate (e.g. constant/silent signal): no usable rest/action split.
        return [], activity, enter, exit_thr
    raw = hysteresis_segments(activity, enter, exit_thr)
    segs = filter_segments(
        raw, config.fs, config.min_action_s, config.min_rest_gap_s
    )
    return segs, activity, enter, exit_thr
