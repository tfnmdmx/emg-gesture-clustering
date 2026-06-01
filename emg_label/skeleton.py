from __future__ import annotations

import numpy as np


# Manus glove joint layout: 25 joints arranged as 5 chains of 5 (thumb..pinky).
# First joint of each chain (indices 0,5,10,15,20) is the wrist/palm anchor;
# last joint (4,9,14,19,24) is the fingertip.
SKELETON_CONNECTIONS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24],
]
SKELETON_PALM_CONNECTIONS = [[0, 5, 10, 15, 20]]
SKELETON_ROOT_INDICES = [0, 5, 10, 15, 20]
SKELETON_FINGER_COLORS = ["#CC6677", "#4477AA", "#228833", "#EE6677", "#AA3377"]


def normalize_skeleton(xyz):
    """Centre an ``(..., 25, 3)`` skeleton on the mean root-joint position.

    The first joint of each finger chain (indices 0,5,10,15,20) anchors at the
    palm; subtracting their per-frame mean keeps the rendered hand visually
    pinned at the same point so a viewer can compare gesture shape across
    frames instead of getting distracted by absolute translation.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    roots = xyz[..., SKELETON_ROOT_INDICES, :]
    return xyz - np.nanmean(roots, axis=-2, keepdims=True)


def axis_limits(xyz, margin_ratio: float = 0.14):
    """Cubic ``[(xlo,xhi), (ylo,yhi), (zlo,zhi)]`` limits for a point cloud."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    mid = (mins + maxs) / 2.0
    span = float(np.nanmax(maxs - mins))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    half = span * (0.5 + margin_ratio)
    return [(float(mid[i] - half), float(mid[i] + half)) for i in range(3)]


def draw_skeleton(ax, xyz_25x3, palm_color: str = "#888888") -> None:
    """Plot one 25-joint skeleton frame on a 3D axes (matplotlib).

    Independent of torch / emg2pose FK -- works directly on the XYZ joint
    positions stored in raw Manus npz under ``manus_<hand>_skeleton``.
    """
    pts = np.asarray(xyz_25x3, dtype=np.float64)
    for conn, color in zip(SKELETON_CONNECTIONS, SKELETON_FINGER_COLORS):
        chain = pts[conn]
        ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], color=color, lw=2.5)
        ax.scatter(chain[:, 0], chain[:, 1], chain[:, 2], color=color, s=18)
    for conn in SKELETON_PALM_CONNECTIONS:
        chain = pts[conn]
        ax.plot(chain[:, 0], chain[:, 1], chain[:, 2],
                color=palm_color, lw=1.2, ls="--", alpha=0.65)
