import numpy as np

from emg_label.pose_segmentation import (
    build_smc_clips, mask_to_intervals, match_bursts_to_clips,
    match_clips_to_bursts, pose_speed, robust_threshold,
    static_motion_intervals,
)


def test_pose_speed_rises_during_motion():
    # 1s rest, 1s moving (steady velocity), 1s rest -- pose-speed should
    # spike in the middle and decay back to ~0 at the ends.
    fs = 1000
    ja = np.zeros((3 * fs, 5))
    ja[fs:2 * fs] = (np.arange(fs)[:, None] / fs) * np.ones(5)
    spd = pose_speed(ja, fs, smooth_ms=20.0)
    assert spd.shape == (3 * fs,)
    assert spd[1500] > 10 * spd[100]
    assert spd[2500] < spd[1500]


def test_robust_threshold_above_median_when_outliers_present():
    rng = np.random.default_rng(0)
    x = np.concatenate([np.abs(rng.normal(0, 0.1, 1000)),
                        np.full(50, 5.0)])
    t = robust_threshold(x, percentile=35.0, mad_scale=1.5)
    # Threshold must lie above the rest cloud (~0.1) but below the outlier mass.
    assert 0.1 < t < 5.0


def test_mask_to_intervals_filters_short_and_merges_close():
    # Order matches reference: filter-by-length FIRST, then merge close survivors.
    # Short intervals are dropped before they can be merged in.
    fs = 1000
    mask = np.zeros(3000, dtype=bool)
    mask[100:200] = True       # 0.1s -> too short, dropped at filter stage
    mask[400:900] = True       # 0.5s -> kept
    mask[1000:1700] = True     # 0.7s -> kept; gap of 100ms to prev -> merged
    mask[2500:2600] = True     # 0.1s -> too short, dropped
    out = mask_to_intervals(mask, fs, min_sec=0.3, merge_gap_sec=0.2)
    assert out == [(400, 1700)]


def test_static_motion_intervals_alternation():
    # speed: static-motion-static-motion-static
    fs = 1000
    spd = np.zeros(5000)
    spd[1000:2000] = 5.0
    spd[3000:4000] = 5.0
    static, motion = static_motion_intervals(
        spd, fs, threshold=1.0,
        min_static_s=0.2, min_motion_s=0.2, merge_gap_s=0.05)
    # Three static blocks of length 1000 samples each.
    assert len(static) == 3
    # Two motion blocks (1000:2000) and (3000:4000).
    assert motion == [(1000, 2000), (3000, 4000)]


def test_build_smc_clips_one_per_gesture():
    fs = 1000
    static = [(0, 1000), (2000, 3000), (4000, 5000)]
    motion = [(1000, 2000), (3000, 4000)]
    clips = build_smc_clips(static, motion, fs, n_samples=5000,
                            pre_static_s=0.2, post_static_s=0.2,
                            pad_s=0.0, min_motion_s=0.2)
    # Two static->motion->static triples => two clips.
    assert len(clips) == 2
    c0 = clips[0]
    # First clip's motion is between samples 1000 and 2000.
    assert c0["motion_start"] == 1000 and c0["motion_end"] == 2000
    # pre_static caps left side to last 0.2s of static_in (i.e. sample 800).
    assert c0["clip_start"] == 800
    # post_static caps right side to first 0.2s of static_out (i.e. sample 2200).
    assert c0["clip_end"] == 2200


def test_build_smc_clips_drops_short_motion():
    fs = 1000
    static = [(0, 1000), (1050, 2000)]
    motion = [(1000, 1050)]  # 50ms -- below min_motion_s=0.2s
    clips = build_smc_clips(static, motion, fs, n_samples=2000,
                            pre_static_s=0.2, post_static_s=0.2, pad_s=0.0,
                            min_motion_s=0.2)
    assert clips == []


def test_match_bursts_to_clips_overlap_wins():
    burst = [(100, 200), (1500, 1600)]
    clips = [{"clip_id": 0, "clip_start": 0, "clip_end": 1000},
             {"clip_id": 1, "clip_start": 1200, "clip_end": 1700}]
    assert match_bursts_to_clips(burst, clips) == [0, 1]


def test_match_bursts_to_clips_unmatched_returns_minus_one():
    burst = [(5000, 5100)]
    clips = [{"clip_id": 0, "clip_start": 0, "clip_end": 1000}]
    assert match_bursts_to_clips(burst, clips) == [-1]


def test_match_clips_to_bursts_symmetric():
    burst = [(100, 200), (1500, 1600)]
    clips = [{"clip_id": 0, "clip_start": 0, "clip_end": 1000},
             {"clip_id": 1, "clip_start": 1200, "clip_end": 1700}]
    assert match_clips_to_bursts(clips, burst) == [0, 1]
