from __future__ import annotations

import math
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
    # Auto-threshold placement between the rest-mode center (0.0) and the Otsu
    # valley (1.0). enter_k < 1.0 lowers the enter threshold BELOW the Otsu
    # valley to catch lower-amplitude bursts the valley would miss; exit_k sets
    # the hysteresis exit and must stay < enter_k. (enter_k=1.0, exit_k=0.5
    # reproduces the original "enter at the Otsu valley" behaviour.) Per-recording
    # absolute overrides still come from enter_thresh / exit_thresh.
    emg_enter_k: float = 0.8
    emg_exit_k: float = 0.4
    # Pose-speed (joint-angle velocity) smoothing/threshold knobs. The live
    # detector (action_segmentation.segment_recording) uses pose_smooth_ms for the
    # speed signal and, when auto_move_thresh=True, derives move_enter/move_exit
    # from robust_threshold(speed, pose_pct, pose_mad). min_static_s/min_motion_s
    # gate the pose-hysteresis dwell tests.
    pose_smooth_ms: float = 250.0
    pose_pct: float = 35.0
    pose_mad: float = 1.5
    min_static_s: float = 0.35
    min_motion_s: float = 0.20
    # --- legacy / diagnostic-only knobs ---------------------------------------
    # The live pipeline (segment.py -> segment_recording) does NOT read these.
    # They parameterize the superseded segmenters in pose_segmentation.py
    # (static_motion_intervals, velocity_peak_segments) which now run only inside
    # diag_seg.py for the before/after over-cut comparison. Kept so that
    # diagnostic stays runnable; not exposed as segment.py CLI flags.
    merge_gap_s: float = 0.20         # static_motion_intervals merge gap
    pose_prom_k: float = 2.0          # velocity-peak prominence = k * MAD(speed)
    pose_min_gesture_s: float = 0.25  # velocity-peak min spacing AND min length
    pose_bound_frac: float = 0.5      # velocity-peak segment-bound = frac*detect_thr
    pose_peak_merge_gap_s: float = 0.12
    # Absolute motion gate -- robust_threshold is per-recording relative, so on a
    # static recording it descends to the noise floor and manufactures phantom
    # segments. A real dynamic gesture must move the joints an absolute distance.
    # RADIANS: joint angles are normalized to radians at load (io_utils.load_npz),
    # so pose_range = ‖max-min‖ is a rad-space norm. ~15 deg-equivalent.
    pose_min_range: float = math.radians(15.0)   # min joint excursion per segment
    pose_long_seg_s: float = 2.5      # segments longer than this -> review flag
    # --- Unified pose-hysteresis + EMG-split detector (action_segmentation.py).
    # Designed for the regular 静止-动作-静止 protocol with NO neutral return: a
    # hold is the formed pose itself, so rest = low pose-speed AND low EMG (never
    # 'near neutral'). pose-speed is the spine; EMG splits back-to-back holds.
    # See docs/检测切分统一设计.md.
    # rad/s (joint angles are radians post-load). Only used when
    # auto_move_thresh=False (ablation) or as the near-static fallback; the
    # math.radians() keeps the original deg-tuned provenance visible.
    move_enter: float = math.radians(60.0)   # rad/s (≈60 deg/s): STATIC->MOVING onset
    move_exit: float = math.radians(25.0)    # rad/s (≈25 deg/s): MOVING->STATIC settle
    settle_frac: float = 0.25         # relative-settle: spd <= max(move_exit, frac*peak_run)
    auto_move_thresh: bool = True     # derive move_enter/exit per-rec from robust_threshold
    move_exit_frac: float = 0.5       # move_exit = frac * move_enter when auto
    enable_r2: bool = True            # split over-long never-settling runs at EMG-rest gaps
    enable_r7: bool = False           # back-to-back hold veto: OFF by default -- on regular
                                      # data the ~125ms pose-smoothing group delay drops EMG
                                      # bursts into the preceding hold, making R7 over-split.
                                      # Only enable if a recording genuinely under-cuts.
    max_hold_s: float = 4.0           # hold longer than this -> review_flag='long_static'
    snap_window_s: float = 0.15       # half-window for onset/cut snapping to speed foot/min
    nan_max_gap_s: float = 0.20       # Manus dropouts shorter than this are linearly filled
    min_finite_frac: float = 0.5      # window with fewer finite rows than this fails the gate
