from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def pose_speed(joint_angles, fs: int, smooth_ms: float = 250.0):
    """Smoothed joint-angle speed: ``‖d ja / dt‖`` along the joint axis.

    Independent of EMG: rises when the hand is moving, drops to near-zero when
    the gesture is held -- catches slow/passive transitions that the EMG burst
    detector misses, and pins down where the pose is settled.

    NaN samples (raw Manus occlusion drops) propagate through ``np.diff`` and
    ``uniform_filter1d`` -- those regions become NaN speed and fail the
    ``speed <= threshold`` test, i.e. they're treated as motion (which is the
    safe assumption: an unknown segment shouldn't be promoted to a static
    hold). ``np.errstate`` mutes the benign invalid-subtract warning.
    """
    ja = np.asarray(joint_angles, dtype=float)
    if len(ja) < 2:
        return np.zeros(len(ja), dtype=float)
    with np.errstate(invalid="ignore"):
        delta = np.linalg.norm(np.diff(ja, axis=0), axis=1) * fs
    speed = np.concatenate([[delta[0]], delta])
    win = max(1, int(round(smooth_ms / 1000.0 * fs)))
    return uniform_filter1d(speed, size=win, mode="nearest")


def robust_threshold(x, percentile: float = 35.0, mad_scale: float = 1.5) -> float:
    """``max(P{percentile}, median + mad_scale * 1.4826 * MAD)``.

    Used as the static/motion split on the pose-speed signal. Unlike Otsu it
    doesn't require a bimodal histogram, so it survives near-uniform recordings
    where most of the timeline is either static or motion.
    """
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad_sigma = 1.4826 * float(np.median(np.abs(finite - median)))
    pct = float(np.percentile(finite, percentile))
    return max(pct, median + mad_scale * mad_sigma)


def mask_to_intervals(mask, fs: int, min_sec: float, merge_gap_sec: float):
    """Boolean mask -> list of ``(start, end)`` sample-index intervals (end exclusive).

    Drops intervals shorter than ``min_sec``; then merges intervals separated
    by gaps shorter than ``merge_gap_sec``.
    """
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    padded = np.concatenate([[False], m, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    min_samples = int(round(min_sec * fs))
    raw = []
    for s, e in edges.reshape(-1, 2):
        if e - s >= min_samples:
            raw.append((int(s), int(e)))
    if not raw:
        return []
    merge_gap = int(round(merge_gap_sec * fs))
    merged = [list(raw[0])]
    for s, e in raw[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def velocity_peak_segments(speed, fs: int, detect_thr: float,
                           bound_frac: float = 0.5, prom_k: float = 2.0,
                           min_gesture_s: float = 0.25, merge_gap_s: float = 0.12):
    """One dynamic-gesture segment per pose-speed peak; split mega-runs at valleys.

    A dynamic gesture is a *movement* = one bell in the pose-speed signal. Unlike
    ``static_motion_intervals`` (which needs a static hold to bracket gestures and
    so collapses continuous back-to-back gesturing into one mega-segment), this
    cuts on the movements themselves:

    1. detect peaks above ``detect_thr`` (the robust motion floor), spaced at
       least ``min_gesture_s`` apart, each standing out by >= ``prom_k * MAD``
       (so a wobble inside one movement is not promoted to its own gesture);
    2. bound each movement at the lower ``bound_frac * detect_thr`` so the rising
       and falling flanks stay in the segment (hysteresis: detect high, bound low);
    3. when one supra-``bound`` run holds several peaks (continuous gesturing that
       never settles) split it at the velocity *valley* (argmin) between
       consecutive peaks.

    ``apex`` for each segment is the peak-velocity frame -- for a dynamic gesture
    the most intense instant of the movement, far more meaningful than a settled
    end-pose. NaN samples (Manus occlusion) are treated as no-motion (0) for
    detection, the safe default. Returns ``(segments, apex_samples)``.
    """
    s = np.nan_to_num(np.asarray(speed, float), nan=0.0)
    finite = s[np.isfinite(s)]
    med = float(np.median(finite)) if finite.size else 0.0
    mad = 1.4826 * float(np.median(np.abs(finite - med))) if finite.size else 0.0
    prom = max(prom_k * mad, 1e-9)
    min_dist = max(1, int(round(min_gesture_s * fs)))
    peaks, _ = find_peaks(s, height=detect_thr, distance=min_dist, prominence=prom)

    bound = bound_frac * detect_thr
    runs = mask_to_intervals(s > bound, fs, 0.0, merge_gap_s)
    segs, apexes = [], []
    for rs, re in runs:
        pk = peaks[(peaks >= rs) & (peaks < re)]
        if len(pk) == 0:
            continue
        bnds = [rs]
        for j in range(len(pk) - 1):
            bnds.append(pk[j] + int(np.argmin(s[pk[j]:pk[j + 1] + 1])))
        bnds.append(re)
        for j in range(len(pk)):
            a, b = bnds[j], bnds[j + 1]
            if b - a >= min_dist:
                segs.append((int(a), int(b)))
                apexes.append(int(pk[j]))
    return segs, apexes


def static_motion_intervals(speed, fs: int, threshold: float,
                            min_static_s: float = 0.35,
                            min_motion_s: float = 0.20,
                            merge_gap_s: float = 0.20):
    """Split a pose-speed signal into static-hold vs. transition-motion intervals.

    Static = ``speed <= threshold`` for at least ``min_static_s``. Motion =
    everything in between (plus head/tail) that is at least ``min_motion_s``.
    """
    n = len(speed)
    static = mask_to_intervals(np.asarray(speed) <= threshold, fs,
                               min_static_s, merge_gap_s)
    min_motion = int(round(min_motion_s * fs))
    motion = []
    cursor = 0
    for s, e in static:
        if s - cursor >= min_motion:
            motion.append((cursor, s))
        cursor = max(cursor, e)
    if n - cursor >= min_motion:
        motion.append((cursor, n))
    return static, motion


def build_smc_clips(static_intervals, motion_intervals, fs: int, n_samples: int,
                    pre_static_s: float = 0.5, post_static_s: float = 0.5,
                    pad_s: float = 0.05, min_motion_s: float = 0.20):
    """Compose static -> motion -> static triples into one clip per gesture.

    ``pre_static_s`` / ``post_static_s`` cap each static side to that much
    context (long holds don't blow up the clip). ``pad_s`` extends both
    endpoints by a small buffer to give annotators breathing room. Returns
    a list of dicts in time order.
    """
    events = sorted(
        [(s, e, "static") for s, e in static_intervals]
        + [(s, e, "motion") for s, e in motion_intervals],
        key=lambda x: x[0],
    )
    pad = int(round(pad_s * fs))
    pre = int(round(pre_static_s * fs))
    post = int(round(post_static_s * fs))
    min_motion = int(round(min_motion_s * fs))

    clips = []
    cid = 0
    for i in range(1, len(events) - 1):
        ps, pe, pt = events[i - 1]
        ms, me, mt = events[i]
        ns, ne, nt = events[i + 1]
        if pt != "static" or mt != "motion" or nt != "static":
            continue
        if me - ms < min_motion:
            continue
        clip_start = max(0, max(ps, ms - pre) - pad)
        clip_end = min(n_samples, min(ne, me + post) + pad)
        clips.append({
            "clip_id": cid,
            "clip_start": int(clip_start),
            "clip_end": int(clip_end),
            "static_in_start": int(ps),
            "static_in_end": int(pe),
            "motion_start": int(ms),
            "motion_end": int(me),
            "static_out_start": int(ns),
            "static_out_end": int(ne),
        })
        cid += 1
    return clips


def match_bursts_to_clips(burst_segments, clips):
    """For each burst, the ``clip_id`` of the clip with maximum sample overlap, or -1.

    Symmetric companion to ``match_clips_to_bursts``. Used to populate the
    ``matched_clip_id`` cross-reference in segments.csv so disagreements
    between the two segmenters are visible at a glance.
    """
    out = []
    for bs, be in burst_segments:
        best_id, best_ov = -1, 0
        for c in clips:
            ov = max(0, min(be, c["clip_end"]) - max(bs, c["clip_start"]))
            if ov > best_ov:
                best_ov = ov
                best_id = c["clip_id"]
        out.append(best_id)
    return out


def match_clips_to_bursts(clips, burst_segments):
    """For each clip, the ``seg_idx`` of the burst with maximum overlap, or -1."""
    out = []
    for c in clips:
        best_idx, best_ov = -1, 0
        for j, (bs, be) in enumerate(burst_segments):
            ov = max(0, min(be, c["clip_end"]) - max(bs, c["clip_start"]))
            if ov > best_ov:
                best_ov = ov
                best_idx = j
        out.append(best_idx)
    return out
