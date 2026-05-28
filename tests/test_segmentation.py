import numpy as np

from emg_label.segmentation import emg_envelope


def test_emg_envelope_high_during_action():
    # 16-channel EMG: quiet baseline, a burst of activity in the middle
    rng = np.random.default_rng(0)
    emg = rng.normal(0, 1.0, size=(1000, 16))
    emg[400:600] += rng.normal(0, 20.0, size=(200, 16))  # muscle burst
    env = emg_envelope(emg, fs=1000, smooth_ms=20.0)
    assert env.shape == (1000,)
    assert env[500] > env[100]          # burst > baseline
    assert env.min() >= 0.0             # rectified envelope is non-negative


from emg_label.segmentation import auto_thresholds


def test_auto_thresholds_enter_above_exit():
    rng = np.random.default_rng(1)
    act = np.abs(rng.normal(0, 0.1, size=5000))
    act[1000:1200] += 2.0
    enter, exit_thr = auto_thresholds(act)
    assert enter > exit_thr
    assert exit_thr > np.median(act)


def test_auto_thresholds_robust_to_high_duty_cycle():
    # 50% of samples are "action" at ~1.5: median sits between the two modes,
    # so a median+MAD heuristic fails. Otsu must still land between them.
    act = np.concatenate([np.full(2500, 0.02), np.full(2500, 1.5)])
    enter, exit_thr = auto_thresholds(act)
    assert 0.02 < exit_thr < enter < 1.5


from emg_label.segmentation import hysteresis_segments, filter_segments


def test_hysteresis_basic():
    a = np.array([0, 0, 5, 5, 5, 0, 0, 5, 5, 0], dtype=float)
    segs = hysteresis_segments(a, enter_thr=3.0, exit_thr=1.0)
    assert segs == [(2, 5), (7, 9)]


def test_hysteresis_open_segment_at_end():
    a = np.array([0, 5, 5], dtype=float)
    segs = hysteresis_segments(a, enter_thr=3.0, exit_thr=1.0)
    assert segs == [(1, 3)]


def test_filter_drops_short_segments():
    # fs=1000 -> min_action 0.4s = 400 samples
    segs = [(0, 100), (1000, 1600)]
    out = filter_segments(segs, fs=1000, min_action_s=0.4, min_rest_gap_s=0.2)
    assert out == [(1000, 1600)]


def test_filter_merges_close_segments():
    # gap of 50 samples < min_rest_gap 0.2s(=200) -> merge
    segs = [(0, 500), (550, 1000)]
    out = filter_segments(segs, fs=1000, min_action_s=0.1, min_rest_gap_s=0.2)
    assert out == [(0, 1000)]


from emg_label.config import Config
from emg_label.segmentation import segment_emg


def _synthetic_emg_action_rest(fs=1000):
    # rest-dominant: 12s total, three 1s muscle bursts at 2s, 5s, 8s.
    # Between bursts the muscle relaxes (low EMG), so the envelope returns
    # to baseline and the bursts are detected as three separate segments.
    rng = np.random.default_rng(2)
    emg = rng.normal(0, 1.0, size=(12 * fs, 16))
    for c in (2, 5, 8):
        emg[c * fs:(c * fs + fs)] += rng.normal(0, 25.0, size=(fs, 16))
    return emg


def test_segment_emg_finds_three_actions():
    emg = _synthetic_emg_action_rest(fs=1000)
    cfg = Config(fs=1000, min_action_s=0.4, min_rest_gap_s=0.2, smooth_ms=50.0)
    segs, act, enter, exit_thr = segment_emg(emg, cfg)
    assert len(segs) == 3
    # each segment roughly within its 1s burst window (±200ms tolerance)
    centers = [(s + e) / 2 / 1000 for s, e in segs]
    assert np.allclose(sorted(centers), [2.5, 5.5, 8.5], atol=0.2)
    assert enter > exit_thr


def test_segment_emg_constant_signal_returns_no_segments():
    # a perfectly flat (silent/broken) recording yields degenerate thresholds
    # (enter == exit); the guard must return no segments rather than one giant one
    emg = np.full((5000, 16), 3.0)
    cfg = Config(fs=1000)
    segs, act, enter, exit_thr = segment_emg(emg, cfg)
    assert segs == []
    assert enter <= exit_thr
