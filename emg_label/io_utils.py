from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class FileInfo:
    path: str
    stem: str
    subject: str | None
    hand: str | None
    group: str
    parsed: bool


def parse_file_info(path: str) -> FileInfo:
    """Parse ``{subject}__{date}-{hand}[-{run}|_{note}...]__{timestamp}``.

    The middle segment is ``date-hand`` with an optional run number (joined
    by ``-``) and/or a note suffix (joined by ``_``):
    ``20260502-left``, ``20260502-left-3``, ``20260504-left-2_v71``,
    ``20260504-left_v71``. ``hand`` is normalized to its leading
    ``left``/``right`` prefix, so notes like ``_v71`` or ``_v33flat`` don't
    spawn separate groups.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("__")
    mid = parts[1].split("-") if len(parts) >= 3 else []
    if len(mid) >= 2:
        subject = parts[0]
        hand_raw = mid[1]
        hand_lower = hand_raw.lower()
        if hand_lower.startswith("left"):
            hand = "left"
        elif hand_lower.startswith("right"):
            hand = "right"
        else:
            hand = hand_raw
        return FileInfo(path, stem, subject, hand, f"{subject}-{hand}", True)
    warnings.warn(f"Cannot parse filename, treating as own group: {stem}")
    return FileInfo(path, stem, None, None, stem, False)


def group_files(paths: list[str]) -> dict[str, list[FileInfo]]:
    groups: dict[str, list[FileInfo]] = {}
    for p in paths:
        info = parse_file_info(p)
        groups.setdefault(info.group, []).append(info)
    return groups


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    emg = np.asarray(d["emg"], dtype=np.float32)
    ja = np.asarray(d["joint_angles"], dtype=np.float32)
    if emg.ndim != 2 or ja.ndim != 2:
        raise ValueError(f"Expected 2D emg/joint_angles in {path}")
    if emg.shape[0] != ja.shape[0]:
        raise ValueError(
            f"emg/joint_angles length mismatch in {path}: "
            f"{emg.shape[0]} vs {ja.shape[0]}"
        )
    return emg, ja
