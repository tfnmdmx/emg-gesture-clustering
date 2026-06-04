from __future__ import annotations

"""Unified action segmentation for the regular 静止-动作-静止-动作 protocol.

Design spec: docs/检测切分统一设计.md (pose-spine + EMG-split, asymmetric fusion).

The protocol is move -> hold -> move -> hold with NO return to a neutral pose:
each 静止/hold is the just-formed gesture pose itself, and consecutive holds are
DIFFERENT non-neutral poses. So a hold cannot be detected as "near neutral"; it is
detected as low pose-speed AND low EMG (joints stopped AND muscle relaxed).

Two independent detectors, asymmetric fusion:
  * pose-speed hysteresis state machine = the SPINE -- it draws every motion onset
    and where the hand settles (it catches the gentle/slow gestures EMG misses).
    A short velocity dip inside one natural gesture does not settle a run
    (dip-merge), which kills the velocity-peak 2x over-cut.
  * EMG envelope = the ARBITER for splitting -- between two consecutive non-neutral
    holds the hand may barely move (it only changes which muscle is engaged), so
    pose-speed has no valley; but the muscle DOES relax between the two efforts.
    EMG re-splits an over-long merged run (R2) and, crucially, vetoes the
    back-to-back-hold under-cut (R7): a normal-length pose-hold that contains a
    clean EMG burst actually spans two poses -> force a cut at the burst.

One action = a motion run + its following REAL stable hold, right boundary at the
next onset (segments tile seamlessly, no artificial pad, no hold cap).

This module is standalone (imported by diag_seg.py for validation) and does NOT
touch the production segment.py path until the diagnostic confirms the metrics.
"""

import numpy as np

from . import features, segmentation
from .pose_segmentation import pose_speed, robust_threshold


# ---------- small geometry helpers ------------------------------------------

def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _complement(runs, n):
    """Intervals of [0, n) not covered by the (sorted, disjoint) `runs`."""
    holds, cursor = [], 0
    for s, e in runs:
        if s > cursor:
            holds.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < n:
        holds.append((cursor, n))
    return holds


# ---------- NaN handling + absolute gate ------------------------------------

def interpolate_short_nan_gaps(ja, fs, max_gap_s=0.20, min_finite_frac=0.5):
    """Per-channel linear-fill of internal NaN runs shorter than `max_gap_s`.

    Manus drops single/short frames during occlusion; if not filled BEFORE
    pose_speed, uniform_filter1d smears one NaN frame into a ~250 ms NaN-speed
    hole that reads as motion and can split one hold in two. Head/tail gaps (no
    anchor on one side) and gaps longer than `max_gap_s` are left NaN -> they
    propagate to NaN speed and are treated as motion (the safe default).
    """
    ja = np.array(ja, dtype=float, copy=True)
    n, d = ja.shape
    max_gap = int(round(max_gap_s * fs))
    idx = np.arange(n)
    for c in range(d):
        col = ja[:, c]
        nan = ~np.isfinite(col)
        if not nan.any():
            continue
        padded = np.concatenate([[False], nan, [False]])
        edges = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
        for s, e in edges:
            if s == 0 or e == n:          # head/tail: no two-sided anchor
                continue
            if (e - s) <= max_gap:
                col[s:e] = np.interp(idx[s:e], [s - 1, e],
                                     [col[s - 1], col[e]])
        ja[:, c] = col
    return ja


def seg_range_ok(ja, a, b, pose_min_range, min_finite_frac=0.5):
    """Absolute joint-excursion gate with a NaN-fraction guard.

    A real gesture moves the joints an absolute distance. `nanmax/nanmin` over an
    all-NaN window returns NaN and `NaN < thr` is False, which would wrongly let a
    NaN-saturated window PASS -- so a window with too few finite rows fails first.
    """
    if b <= a:
        return False
    w = ja[a:b]
    finite_rows = np.all(np.isfinite(w), axis=1)
    if finite_rows.mean() < min_finite_frac:
        return False
    with np.errstate(invalid="ignore"):
        rng = float(np.linalg.norm(np.nanmax(w, axis=0) - np.nanmin(w, axis=0)))
    return np.isfinite(rng) and rng >= pose_min_range


# ---------- Detector B: pose static/motion hysteresis -----------------------

def pose_hysteresis(spd, fs, move_enter, move_exit, min_static_s=0.35,
                    min_motion_s=0.20, settle_frac=0.25):
    """STATIC<->MOVING hysteresis on pose-speed -> (motion_runs, holds).

    STATIC->MOVING when spd >= move_enter. MOVING->STATIC when spd settles below
    max(move_exit, settle_frac * peak_of_this_run) for at least min_static_s --
    the relative-settle term (clamped below by move_exit) closes a tense micro-
    jitter hold sitting in the move_exit..move_enter dead-band, which a single
    absolute threshold would leave forever open (under-cut). A dip too short to
    satisfy the dwell never settles the run -> dip-merge -> no velocity-peak
    over-cut. NaN speed never triggers motion-onset nor counts as settled.
    """
    spd = np.asarray(spd, dtype=float)
    n = len(spd)
    min_static = int(round(min_static_s * fs))
    min_motion = int(round(min_motion_s * fs))
    runs = []
    state, start, quiet, peak_run = "STATIC", 0, 0, 0.0
    for t in range(n):
        v = spd[t]
        if state == "STATIC":
            if np.isfinite(v) and v >= move_enter:
                state, start, peak_run, quiet = "MOVING", t, v, 0
        else:
            if np.isfinite(v) and v > peak_run:
                peak_run = v
            # relative-settle closes a tense hold above move_exit, but must stay
            # below move_enter or it would re-trigger motion on the next sample.
            settle = min(0.9 * move_enter, max(move_exit, settle_frac * peak_run))
            if np.isfinite(v) and v <= settle:
                quiet += 1
            else:
                quiet = 0
            if quiet >= min_static:
                end = t - quiet + 1
                if end - start >= min_motion:
                    runs.append((int(start), int(end)))
                state, quiet = "STATIC", 0
    if state == "MOVING" and (n - start) >= min_motion:
        runs.append((int(start), n))
    return runs, _complement(runs, n)


# ---------- EMG-rest helpers (env < exit_e, NOT the burst complement) --------

def _rest_run_near(env, exit_e, idx, span, side):
    """Is there EMG-rest (env < exit_e) within `span` samples before/after idx?"""
    if side == "before":
        a, b = max(0, idx - span), idx
    else:
        a, b = idx, min(len(env), idx + span)
    return b > a and bool(np.any(env[a:b] < exit_e))


def clean_rest_gaps(env, inner_bursts, exit_e, min_rest_gap_s, fs):
    """Clean EMG-rest gaps (env<exit_e, length>=min_rest_gap) between inner bursts."""
    if len(inner_bursts) < 2:
        return []
    span = int(round(min_rest_gap_s * fs))
    gaps = []
    bs = sorted(inner_bursts)
    for (a0, a1), (b0, b1) in zip(bs[:-1], bs[1:]):
        gs, ge = a1, b0
        if ge - gs < span:
            continue
        if np.all(env[gs:ge] < exit_e):
            gaps.append((int(gs), int(ge)))
    return gaps


def _env_argmin(env, gap):
    gs, ge = gap
    return gs + int(np.argmin(env[gs:ge]))


def snap_to_speed_min(spd, anchor, snap_window_s, fs):
    """Nearest pose-speed minimum within +/- snap_window of anchor (bounded)."""
    w = int(round(snap_window_s * fs))
    a, b = max(0, anchor - w), min(len(spd), anchor + w + 1)
    if b <= a:
        return int(anchor)
    seg = np.nan_to_num(spd[a:b], nan=np.inf)
    return a + int(np.argmin(seg))


def snap_to_motion_foot(spd, s, fs, move_exit, snap_window_s):
    """Pull a motion onset back to the foot of its rising flank (nearest
    spd<=move_exit on the left, within snap_window). Clamps to the window left
    (or 0) when the recording opens mid-motion."""
    w = int(round(snap_window_s * fs))
    a = max(0, s - w)
    seg = spd[a:s + 1]
    below = np.flatnonzero(np.isfinite(seg) & (seg <= move_exit))
    return int(a + below[-1]) if below.size else int(a)


def _valley_split(spd, ms, me, fs, move_enter):
    """Fallback for a saturated never-resting run (no EMG-rest gap): cut at pose-
    speed valleys between supra-move_enter peaks. Absolute prominence floor (not
    the run's own tiny MAD) so a saturated run is not itself over-cut."""
    from scipy.signal import find_peaks
    seg = np.nan_to_num(spd[ms:me], nan=0.0)
    min_dist = int(round(0.25 * fs))
    prom = 0.5 * move_enter
    peaks, _ = find_peaks(seg, height=move_enter, distance=min_dist, prominence=prom)
    cuts = []
    for j in range(len(peaks) - 1):
        v = peaks[j] + int(np.argmin(seg[peaks[j]:peaks[j + 1] + 1]))
        cuts.append(int(ms + v))
    return cuts


# ---------- threshold resolution --------------------------------------------

def resolve_move_thresholds(spd, env, exit_e, cfg):
    """(move_enter, move_exit) in rad/s (joint angles are radians post-load).

    Pose-speed scale varies ~4x across recordings (jm-0503 detect~2.8 vs hzy~10.4
    rad/s) and EMG-rest is NOT a clean proxy for pose-hold here (during the 50%
    low-EMG time pose-speed p95 is still ~17 rad/s -- gentle gestures fire at
    rest-level EMG), so neither a fixed absolute rad/s nor an EMG-rest-gated
    percentile works.
    The motion onset must be the per-recording static/motion split, so we reuse the
    project's adaptive ``robust_threshold`` (bimodal-ish valley): move_enter at the
    valley, move_exit a fraction below it for hysteresis. Near-static recordings
    (where robust_threshold descends to the noise floor) are caught downstream by
    the absolute R0 excursion gate, not here. cfg.auto_move_thresh=False uses the
    fixed cfg.move_enter/move_exit (for ablation)."""
    if not cfg.auto_move_thresh:
        return cfg.move_enter, cfg.move_exit
    detect = robust_threshold(spd, cfg.pose_pct, cfg.pose_mad)
    if not np.isfinite(detect) or detect <= 0:
        return cfg.move_enter, cfg.move_exit
    return float(detect), float(cfg.move_exit_frac * detect)


# ---------- R7: back-to-back hold veto ---------------------------------------

def _r7_splits(holds, bursts, env, enter, exit_e, spd, cfg, fs):
    """Normal-length pose-hold containing a clean EMG burst (>=min_action, with
    EMG-rest on both sides) actually spans two poses -> inject a cut at the
    muscle-relaxation minimum. R2 handles over-long runs; this catches the
    back-to-back under-cut inside normal-length runs that R2 never triggers on."""
    min_action = int(round(cfg.min_action_s * fs))
    span = int(round(cfg.min_rest_gap_s * fs))
    extra = []
    for hs, he in holds:
        if (he - hs) / fs > cfg.pose_long_seg_s:
            continue
        for bs, be in bursts:
            c = (bs + be) // 2
            if not (hs <= c < he) or (be - bs) < min_action:
                continue
            if (_rest_run_near(env, exit_e, bs, span, "before")
                    and _rest_run_near(env, exit_e, be, span, "after")):
                extra.append(snap_to_speed_min(spd, _env_argmin(env, (bs, be)),
                                               cfg.snap_window_s, fs))
    return extra


def _motion_end_in(motion_runs, s, e, min_motion):
    """End of the leading motion within [s, e): first pose-settle after s, clamped.
    A pure-hold segment (R7 cut inside a hold) gets a nominal short motion."""
    cands = [me for ms, me in motion_runs if me > s and ms < e]
    if cands:
        return int(min(max(s + 1, min(cands)), e - 1))
    return int(min(max(s + 1, s + min_motion), max(s + 1, e - 1)))


# ---------- main entry point -------------------------------------------------

def segment_recording(emg, ja, cfg):
    """Detect + segment one recording. Returns (actions, dbg).

    `actions` is a list of dicts (start/end/motion_start/motion_end/hold_start/
    hold_end/apex/fusion_type/review_flag), tiling [first_onset, n) seamlessly.
    `dbg` carries the signals + intermediate detector outputs + counters for the
    diagnostic plot.
    """
    fs = cfg.fs
    ja = interpolate_short_nan_gaps(ja, fs, cfg.nan_max_gap_s, cfg.min_finite_frac)
    n = len(emg)
    spd = pose_speed(ja, fs, cfg.pose_smooth_ms)
    rest = np.nanmedian(ja, axis=0)
    bursts, env, enter, exit_e = segmentation.segment_emg(emg, cfg)
    move_enter, move_exit = resolve_move_thresholds(spd, env, exit_e, cfg)

    with np.errstate(invalid="ignore"):
        rec_pose_range = float(np.linalg.norm(
            np.nanmax(ja, axis=0) - np.nanmin(ja, axis=0)))
    dbg = {"spd": spd, "env": env, "enter": enter, "exit_e": exit_e,
           "bursts": bursts, "move_enter": move_enter, "move_exit": move_exit,
           "motion_runs": [], "holds": [], "n_r2": 0, "n_r7": 0,
           "pose_static": 0, "rec_pose_range": rec_pose_range}

    if not seg_range_ok(ja, 0, n, cfg.pose_min_range, cfg.min_finite_frac):
        dbg["pose_static"] = 1
        return [], dbg

    motion_runs, holds = pose_hysteresis(
        spd, fs, move_enter, move_exit, cfg.min_static_s, cfg.min_motion_s,
        cfg.settle_frac)
    dbg["motion_runs"], dbg["holds"] = motion_runs, holds

    # R1 onsets (+ R2 split of over-long never-settling runs)
    onsets = []
    for ms, me in motion_runs:
        if not seg_range_ok(ja, ms, me, cfg.pose_min_range, cfg.min_finite_frac):
            continue
        onsets.append(ms)
        if (me - ms) / fs <= cfg.pose_long_seg_s:
            continue
        if not cfg.enable_r2:
            continue
        inner = [(bs, be) for bs, be in bursts if ms <= (bs + be) // 2 < me]
        gaps = clean_rest_gaps(env, inner, exit_e, cfg.min_rest_gap_s, fs)
        if len(inner) >= 2 and gaps:
            for g in gaps:
                cut = snap_to_speed_min(spd, _env_argmin(env, g),
                                        cfg.snap_window_s, fs)
                if seg_range_ok(ja, cut, me, cfg.pose_min_range,
                                cfg.min_finite_frac):
                    onsets.append(cut)
                    dbg["n_r2"] += 1
        else:
            vcuts = _valley_split(spd, ms, me, fs, move_enter)
            onsets.extend(vcuts)
            dbg["n_r2"] += len(vcuts)

    if not onsets:
        return [], dbg

    # R7 back-to-back hold veto (off by default -- see Config.enable_r7)
    if cfg.enable_r7:
        extra = _r7_splits(holds, bursts, env, enter, exit_e, spd, cfg, fs)
        dbg["n_r7"] = len(extra)
        onsets += extra

    # snap all onsets, dedup, sort -> shared boundaries keep tiling seamless
    min_motion = int(round(cfg.min_motion_s * fs))
    onsets = sorted({snap_to_motion_foot(spd, o, fs, move_exit, cfg.snap_window_s)
                     for o in onsets})

    actions = []
    for i, s in enumerate(onsets):
        e = onsets[i + 1] if i + 1 < len(onsets) else n
        me = _motion_end_in(motion_runs, s, e, min_motion)
        hs, he = me, e
        # apex = the formed-pose frame, taken WITHIN the hold (the held pose IS the
        # gesture). With no neutral pose, max-deviation-from-median over the whole
        # [start,end) lands in the transition, not the hold -- so restrict to the
        # hold window. Degenerate hold (continuous run, he~=hs) falls back.
        if he - hs >= 2:
            apex = features.apex_index(ja, hs, he, rest, fs)
        elif he - s >= 2:
            apex = features.apex_index(ja, s, he, rest, fs)
        else:
            apex = s
        motion_dur = (me - s) / fs
        hold_dur = (he - hs) / fs
        emg_in_motion = any(_overlap(bs, be, s, me) > 0 for bs, be in bursts)
        ftype = "both" if emg_in_motion else "pose_only"
        if motion_dur > cfg.pose_long_seg_s:
            flag = "long"            # never-settling run (continuous gesturing)
        elif hold_dur < cfg.min_static_s:
            flag = "nohold"          # movement with no real following hold
        elif not emg_in_motion:
            flag = "slow"            # gentle gesture, rest-level EMG (kept)
        elif hold_dur > cfg.max_hold_s:
            flag = "long_static"     # over-long rest
        else:
            flag = ""
        actions.append({
            "start": int(s), "end": int(e),
            "motion_start": int(s), "motion_end": int(me),
            "hold_start": int(hs), "hold_end": int(he),
            "apex": int(apex), "fusion_type": ftype, "review_flag": flag,
        })
    return actions, dbg
