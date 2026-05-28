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
