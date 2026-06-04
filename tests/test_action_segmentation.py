import numpy as np

from emg_label.action_segmentation import (
    interpolate_short_nan_gaps, pose_hysteresis, seg_range_ok,
    segment_recording,
)
from emg_label.config import Config


# ---------- pose_hysteresis: dip-merge + run detection -----------------------

def test_pose_hysteresis_detects_one_run_per_gesture():
    # rest, move (high speed), rest -> exactly one motion run.
    fs = 1000
    spd = np.zeros(3 * fs)
    spd[fs:2 * fs] = 200.0
    runs, holds = pose_hysteresis(spd, fs, move_enter=100, move_exit=50)
    assert len(runs) == 1
    s, e = runs[0]
    assert abs(s - fs) <= 2 and abs(e - 2 * fs) <= int(0.35 * fs) + 2


def test_pose_hysteresis_dip_merge_keeps_one_run():
    # one gesture with a SHORT (0.1s) velocity dip in the middle -- the dip is
    # shorter than min_static_s so the run must NOT split (dip-merge). This is
    # the mechanism that kills the velocity-peak 2x over-cut.
    fs = 1000
    spd = np.full(3 * fs, 0.0)
    spd[fs:2 * fs] = 200.0
    spd[1450:1550] = 10.0          # 0.1s dip < min_static_s (0.35s)
    runs, _ = pose_hysteresis(spd, fs, move_enter=100, move_exit=50,
                              min_static_s=0.35)
    assert len(runs) == 1


def test_pose_hysteresis_splits_on_real_hold():
    # two gestures separated by a real (>=0.35s) low-speed hold -> two runs.
    fs = 1000
    spd = np.zeros(5 * fs)
    spd[fs:2 * fs] = 200.0          # gesture 1
    spd[3 * fs:4 * fs] = 200.0      # gesture 2 (1s hold between)
    runs, _ = pose_hysteresis(spd, fs, move_enter=100, move_exit=50)
    assert len(runs) == 2


# ---------- NaN handling -----------------------------------------------------

def test_interpolate_short_nan_gaps_fills_short_keeps_long():
    fs = 1000
    ja = np.ones((1000, 3))
    ja[100:120, 0] = np.nan        # 20ms gap -> filled (short)
    ja[500:800, 1] = np.nan        # 300ms gap -> kept NaN (> max_gap_s=0.2)
    out = interpolate_short_nan_gaps(ja, fs, max_gap_s=0.2)
    assert np.all(np.isfinite(out[100:120, 0]))      # short gap filled
    assert np.any(~np.isfinite(out[500:800, 1]))     # long gap left NaN


def test_interpolate_short_nan_gaps_per_channel_independent():
    fs = 1000
    ja = np.ones((1000, 3))
    ja[100:110, 0] = np.nan        # only channel 0 occluded
    out = interpolate_short_nan_gaps(ja, fs, max_gap_s=0.2)
    assert np.all(np.isfinite(out[100:110, 0]))
    assert np.all(out[:, 1] == 1.0) and np.all(out[:, 2] == 1.0)


def test_interpolate_short_nan_gaps_leaves_head_tail():
    # head/tail NaN have no two-sided anchor -> not extrapolated.
    fs = 1000
    ja = np.ones((1000, 1))
    ja[:50, 0] = np.nan
    out = interpolate_short_nan_gaps(ja, fs, max_gap_s=0.2)
    assert np.all(~np.isfinite(out[:50, 0]))


# ---------- absolute excursion gate -----------------------------------------

def test_seg_range_ok_nan_fraction_guard():
    # a window that is mostly NaN must FAIL the gate (NaN < thr would be False
    # and wrongly pass without the guard).
    ja = np.full((100, 3), np.nan)
    ja[:10] = np.arange(10)[:, None]   # 10% finite, big range
    assert seg_range_ok(ja, 0, 100, pose_min_range=1.0, min_finite_frac=0.5) is False


def test_seg_range_ok_passes_real_motion():
    ja = np.zeros((100, 3))
    ja[:, 0] = np.linspace(0, 50, 100)
    assert seg_range_ok(ja, 0, 100, pose_min_range=15.0) is True


# ---------- segment_recording: end-to-end invariants -------------------------

def _synthetic_recording(fs=2000, n_gestures=4):
    """rest-action-rest-action... EMG bursts aligned with pose movements."""
    rng = np.random.default_rng(0)
    period = fs              # 1s per gesture (0.4s move + 0.6s hold)
    n = n_gestures * period + fs
    emg = rng.normal(0, 0.02, (n, 16)).astype(np.float32)
    ja = np.zeros((n, 20), dtype=np.float32)
    pose = np.zeros(20, dtype=np.float32)
    for g in range(n_gestures):
        ms = g * period + fs // 2
        me = ms + int(0.4 * fs)
        target = pose.copy()
        target[g % 20] += 60.0            # a new formed pose each gesture
        ja[ms:me] = np.linspace(pose, target, me - ms)
        ja[me:] = target
        pose = target
        emg[ms:me] += rng.normal(0, 0.3, (me - ms, 16))   # burst during move
    return emg, ja


def test_segment_recording_seamless_tiling():
    cfg = Config()
    emg, ja = _synthetic_recording()
    actions, dbg = segment_recording(emg, ja, cfg)
    assert len(actions) >= 3
    for a, b in zip(actions[:-1], actions[1:]):
        assert a["end"] == b["start"]            # no gap, no overlap


def test_segment_recording_apex_in_hold():
    cfg = Config()
    emg, ja = _synthetic_recording()
    actions, _ = segment_recording(emg, ja, cfg)
    real = [a for a in actions if a["hold_end"] - a["hold_start"] >= 2]
    assert real
    assert all(a["hold_start"] <= a["apex"] < a["hold_end"] for a in real)


def test_segment_recording_static_recording_yields_no_clips():
    cfg = Config()
    rng = np.random.default_rng(1)
    n = 6 * cfg.fs
    emg = rng.normal(0, 0.02, (n, 16)).astype(np.float32)
    ja = np.zeros((n, 20), dtype=np.float32)     # never moves
    actions, dbg = segment_recording(emg, ja, cfg)
    assert actions == []
    assert dbg["pose_static"] == 1
