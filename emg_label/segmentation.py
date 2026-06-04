from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def emg_envelope(emg, fs: int, smooth_ms: float = 150.0):
    """Baseline-centered RMS EMG envelope (reference's robust formulation).

    Per-channel baseline (median across time) is subtracted first so the
    envelope is immune to channel DC drift; then RMS across channels measures
    instantaneous muscle activation variance (the physiologically meaningful
    quantity). Smoothed by a uniform window for segmentation use.

    Rises during a gesture's muscle activity and falls back to baseline when
    the muscle relaxes between gestures -- even when the hand pose has not
    yet returned to neutral, which is what lets it separate consecutive
    non-neutral gestures cleanly (a joint-angle distance-from-rest cannot).
    """
    e = np.asarray(emg, dtype=np.float64)
    baseline = np.nanmedian(e, axis=0, keepdims=True)
    centered = e - baseline
    rms = np.sqrt(np.nanmean(np.square(centered), axis=1))
    win = max(1, int(round(smooth_ms / 1000.0 * fs)))
    return uniform_filter1d(rms, size=win, mode="nearest")


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


def auto_thresholds(activity, enter_k: float = 0.8, exit_k: float = 0.4):
    """Bimodal (Otsu) split between rest and action activity levels.

    The Otsu valley ``t`` marks the gap between the rest mode and the action
    mode. ``enter`` / ``exit`` are placed on the rest_center -> valley span by
    ``enter_k`` / ``exit_k`` (0 = at rest_center, 1.0 = at the valley):

        enter = rest_center + enter_k * (t - rest_center)
        exit  = rest_center + exit_k  * (t - rest_center)

    ``enter_k < 1.0`` lowers the enter threshold below the Otsu valley to catch
    lower-amplitude bursts the valley alone would miss (recordings where a few
    tall bursts drag the valley up); ``enter_k=1.0, exit_k=0.5`` reproduces the
    original "enter at the valley, exit halfway" behaviour. The hysteresis (enter
    high, exit low) keeps the detector from flickering near threshold.

    Falls back to ``median + 1.2 * 1.4826 * MAD`` when Otsu collapses (e.g.
    near-uniform signal) so the segmenter never silently returns zero segments
    on a usable recording.
    """
    a = np.asarray(activity, dtype=float)
    t = _otsu_threshold(a)
    rest = a[a <= t]
    rest_center = float(np.median(rest)) if rest.size else float(a.min())
    span = t - rest_center
    enter = float(rest_center + enter_k * span)
    exit_thr = float(rest_center + exit_k * span)
    if enter <= exit_thr or span <= 0:
        # Otsu degenerate: rest_center sits at/above the valley. Use a robust
        # outlier-style threshold instead -- single-mode safe.
        median = float(np.median(a))
        mad_sigma = 1.4826 * float(np.median(np.abs(a - median)))
        if mad_sigma > 0:
            enter = median + 1.2 * mad_sigma
            exit_thr = median + 0.5 * mad_sigma
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


def hold_windows(segments, n_samples: int, tail_samples: int):
    """Hold-window end (exclusive) per burst segment.

    The burst is the muscle activation; the *hold* is the low-EMG interval
    after it, when the gesture pose is actually formed and held. Hold runs
    from a burst's end up to the next burst's start. For the last segment,
    it extends ``tail_samples`` past the end (or to the signal end).
    """
    n = len(segments)
    ends = []
    for j, (_, e) in enumerate(segments):
        if j + 1 < n:
            ends.append(int(segments[j + 1][0]))
        else:
            ends.append(int(min(n_samples, e + tail_samples)))
    return ends


def segment_emg(emg, config):
    """Detect action segments from the EMG envelope.

    Returns (segments, activity, enter_thr, exit_thr) where ``activity`` is
    the smoothed EMG envelope used for detection.
    """
    activity = emg_envelope(emg, config.fs, config.smooth_ms)
    if config.enter_thresh is not None and config.exit_thresh is not None:
        enter, exit_thr = config.enter_thresh, config.exit_thresh
    else:
        enter, exit_thr = auto_thresholds(
            activity, config.emg_enter_k, config.emg_exit_k)
    if enter <= exit_thr:
        # Degenerate (e.g. constant/silent signal): no usable rest/action split.
        return [], activity, enter, exit_thr
    raw = hysteresis_segments(activity, enter, exit_thr)
    segs = filter_segments(
        raw, config.fs, config.min_action_s, config.min_rest_gap_s
    )
    return segs, activity, enter, exit_thr
