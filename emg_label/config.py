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
