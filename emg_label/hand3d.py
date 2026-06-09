from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

LANDMARK_NAMES = [
    "THUMB_TIP", "INDEX_TIP", "MIDDLE_TIP", "RING_TIP", "PINKY_TIP",
    "WRIST",
    "THUMB_INT", "THUMB_DIST",
    "INDEX_PROX", "INDEX_INT", "INDEX_DIST",
    "MIDDLE_PROX", "MIDDLE_INT", "MIDDLE_DIST",
    "RING_PROX", "RING_INT", "RING_DIST",
    "PINKY_PROX", "PINKY_INT", "PINKY_DIST",
    "PALM",
]

BONE_CONNECTIONS = [
    [5, 6, 7, 0],
    [5, 8, 9, 10, 1],
    [5, 11, 12, 13, 2],
    [5, 14, 15, 16, 3],
    [5, 17, 18, 19, 4],
]

PALM_CONNECTIONS = [
    [20, 5], [20, 8], [20, 11], [20, 14], [20, 17], [20, 7],
]

_CALC_CACHE: dict[str, object] = {}


def _resolve_vis_dir() -> str:
    """Directory containing emg2pose's calculate_hand_error.py -- NO hardcoded
    personal path. Resolution order:
      1. $EMG2POSE_VIS_DIR (explicit override, for a custom/newer version);
      2. the copy vendored in this repo (emg_label/_emg2pose_vis) -- the
         self-contained default, so no external checkout is needed;
      3. emg2pose's own scripts/visualize_3d, relative to the importable package.
    The script still needs the emg2pose package + torch importable (it does
    `import emg2pose.kinematics/...`); this only removes the loose-path lookup.
    Raises ImportError with guidance if none yields the file."""
    cands = []
    env = os.environ.get("EMG2POSE_VIS_DIR")
    if env:
        cands.append(Path(env))
    cands.append(Path(__file__).resolve().parent / "_emg2pose_vis")
    try:
        import emg2pose
        pkg = Path(emg2pose.__file__).resolve().parent
        cands += [pkg / "scripts" / "visualize_3d",
                  pkg.parent / "scripts" / "visualize_3d"]
    except Exception:
        pass
    for c in cands:
        if (c / "calculate_hand_error.py").is_file():
            return str(c.resolve())
    raise ImportError(
        "emg2pose visualize_3d not found (calculate_hand_error.py). Set "
        "$EMG2POSE_VIS_DIR to the directory that contains it "
        "(e.g. <emg2pose-checkout>/scripts/visualize_3d).")


def _get_calculator(side: str):
    # Note: emg2pose's right-hand init currently has a shape-mismatch bug in
    # mirrored_hand_model (22-element joint mask applied to 21-element landmark
    # tensor). We avoid it by always using a left-hand calculator and mirroring
    # the X coordinate after FK for right-hand inputs (a hand is bilaterally
    # symmetric up to X reflection in this kinematic model).
    side = (side or "left").lower()
    if side not in ("left", "right"):
        side = "left"
    if "left" in _CALC_CACHE:
        return _CALC_CACHE["left"]
    vis_dir = _resolve_vis_dir()
    if vis_dir not in sys.path:
        sys.path.insert(0, vis_dir)
    from calculate_hand_error import HandPositionErrorCalculator  # type: ignore
    calc = HandPositionErrorCalculator(device="cpu", side="left")
    _CALC_CACHE["left"] = calc
    return calc


def _to_radians(a):
    """Manus ``*_ergonomics`` joint angles are in DEGREES (range ~±60); emg2pose
    FK expects RADIANS. Feeding degrees twists the fingers into a blob. Convert
    when the values clearly look like degrees (|angle| > 2π somewhere); leave
    genuine radian inputs (e.g. emg2pose's own joint_angles) untouched."""
    finite = a[np.isfinite(a)]
    if finite.size and float(np.max(np.abs(finite))) > 2 * np.pi:
        return a * (np.pi / 180.0)
    return a


def angles_to_landmarks(angles20, side="left"):
    """20 joint angles -> (21, 3) landmark positions in mm.

    Uses emg2pose forward kinematics. The landmark order matches
    LANDMARK_NAMES / BONE_CONNECTIONS (copied from the user's
    visualize_hand_3d.py and consistent with emg2pose's angles_to_positions
    output). For ``side='right'`` we mirror X after FK (see _get_calculator).
    Degree inputs (Manus ergonomics) are auto-converted to radians.
    """
    calc = _get_calculator(side)
    a = _to_radians(np.asarray(angles20, dtype=np.float32).reshape(1, 20))
    pos = np.asarray(calc.angles_to_positions(a))[0]
    if (side or "left").lower() == "right":
        pos = pos.copy()
        pos[:, 0] *= -1.0
    return pos


def angles_batch_to_landmarks(angles_batch, side="left"):
    """Batch version: (N, 20) -> (N, 21, 3) in mm. Single FK call.
    Degree inputs (Manus ergonomics) are auto-converted to radians."""
    calc = _get_calculator(side)
    a = _to_radians(np.asarray(angles_batch, dtype=np.float32).reshape(-1, 20))
    pos = np.asarray(calc.angles_to_positions(a))
    if (side or "left").lower() == "right":
        pos = pos.copy()
        pos[..., 0] *= -1.0
    return pos


def draw_hand(ax, landmarks, finger_color="steelblue", thumb_color="crimson"):
    """Draw a hand skeleton on a 3D axis.

    The thumb (first bone chain) is drawn in ``thumb_color`` and the four
    fingers in ``finger_color`` so the thumb is easy to pick out; palm lines
    are dotted gray.
    """
    L = np.asarray(landmarks)
    for bi, bone in enumerate(BONE_CONNECTIONS):
        col = thumb_color if bi == 0 else finger_color
        ax.plot(L[bone, 0], L[bone, 1], L[bone, 2], c=col, lw=2.5)
        ax.scatter(L[bone, 0], L[bone, 1], L[bone, 2], c=col, s=10)
    for conn in PALM_CONNECTIONS:
        ax.plot(L[conn, 0], L[conn, 1], L[conn, 2], c="gray", lw=0.8, ls=":")
