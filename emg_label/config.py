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
    # Auto-threshold placement between the rest-mode center (0.0) and the Otsu
    # valley (1.0). enter_k < 1.0 lowers the enter threshold BELOW the Otsu
    # valley to catch lower-amplitude bursts the valley would miss; exit_k sets
    # the hysteresis exit and must stay < enter_k. (enter_k=1.0, exit_k=0.5
    # reproduces the original "enter at the Otsu valley" behaviour.) Per-recording
    # absolute overrides still come from enter_thresh / exit_thresh.
    emg_enter_k: float = 0.8
    emg_exit_k: float = 0.4
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
    # --- Unified pose-hysteresis + EMG-split detector (action_segmentation.py).
    # Designed for the regular 静止-动作-静止 protocol with NO neutral return: a
    # hold is the formed pose itself, so rest = low pose-speed AND low EMG (never
    # 'near neutral'). pose-speed is the spine; EMG splits back-to-back holds.
    # See docs/检测切分统一设计.md.
    move_enter: float = 60.0          # deg/s: STATIC->MOVING (pose motion onset);
                                      # only used when auto_move_thresh=False
    move_exit: float = 25.0           # deg/s: MOVING->STATIC settle (must be < move_enter)
    settle_frac: float = 0.25         # relative-settle: spd <= max(move_exit, frac*peak_run)
    auto_move_thresh: bool = True     # derive move_enter/exit per-rec from robust_threshold
    move_exit_frac: float = 0.5       # move_exit = frac * move_enter when auto
    enable_r2: bool = True            # split over-long never-settling runs at EMG-rest gaps
    enable_r7: bool = False           # back-to-back hold veto: OFF by default -- on regular
                                      # data the ~125ms pose-smoothing group delay drops EMG
                                      # bursts into the preceding hold, making R7 over-split.
                                      # Only enable if a recording genuinely under-cuts.
    hold_confirm_s: float = 0.30      # min pose-quiet ∩ EMG-rest overlap to confirm a hold (R3)
    max_hold_s: float = 4.0           # hold longer than this -> review_flag='long_static'
    snap_window_s: float = 0.15       # half-window for onset/cut snapping to speed foot/min
    nan_max_gap_s: float = 0.20       # Manus dropouts shorter than this are linearly filled
    min_finite_frac: float = 0.5      # window with fewer finite rows than this fails the gate
