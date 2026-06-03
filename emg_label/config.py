from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    fs: int = 2000
    smooth_ms: float = 150.0
    min_action_s: float = 0.4
    min_rest_gap_s: float = 0.2
    k_min: int = 12
    k_max: int = 30
    out_dir: str = "out"
    enter_thresh: float | None = None
    exit_thresh: float | None = None
    # Pose-speed (joint-angle velocity) segmentation -- independent of EMG.
    # Used to carve static_hold / transition_motion intervals and compose them
    # into static-motion-static gesture clips for manual labelling.
    pose_smooth_ms: float = 250.0
    pose_pct: float = 35.0
    pose_mad: float = 1.5
    min_static_s: float = 0.35
    min_motion_s: float = 0.20
    merge_gap_s: float = 0.20
    pre_static_s: float = 0.5
    post_static_s: float = 0.5
    pad_s: float = 0.05
    # Velocity-peak segmentation (the dynamic-gesture segmenter that replaces
    # static->motion->static clips: one segment per pose-speed peak). See
    # pose_segmentation.velocity_peak_segments.
    pose_prom_k: float = 2.0          # peak prominence = k * MAD(speed)
    pose_min_gesture_s: float = 0.25  # min peak spacing AND min segment length
    pose_bound_frac: float = 0.5      # segment-bound level = frac * detect_thr
    pose_peak_merge_gap_s: float = 0.12
    # Absolute motion gate -- robust_threshold is per-recording relative, so on a
    # static recording it descends to the noise floor and manufactures phantom
    # segments. A real dynamic gesture must move the joints an absolute distance.
    pose_min_range: float = 15.0      # min joint excursion (pose_range) per segment
    pose_long_seg_s: float = 2.5      # segments longer than this -> review flag
