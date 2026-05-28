from __future__ import annotations

"""Animated 3D hand pose for one exported gesture segment.

Renders the hand skeleton (emg2pose forward kinematics) frame by frame over
the segment's duration as an interactive Plotly HTML (rotate / zoom / play),
so you can watch the gesture form.

Output path (organized by label + filename):
    <out_root>/<subdir>/<label>/<npz stem>.html

By default <out_root> is auto-derived: if the npz sits at
``<root>/segments/<label>/file.npz`` then <out_root>=<root>; otherwise the
npz's own directory is used. <subdir> defaults to ``hand_anim``.

Usage:
    python animate_segment.py <segment.npz> [--out-root out_fgw]
        [--subdir hand_anim] [--side left|right] [--max-frames 80] [--fps 30]
"""

import argparse
import os

import numpy as np
import plotly.graph_objects as go

from emg_label.hand3d import (
    BONE_CONNECTIONS,
    PALM_CONNECTIONS,
    LANDMARK_NAMES,
    angles_batch_to_landmarks,
)

FINGER_COLORS = {
    "thumb": "#e41a1c", "index": "#ff7f00", "middle": "#4daf4a",
    "ring": "#377eb8", "pinky": "#984ea3", "palm": "#7f7f7f",
}
# order matches BONE_CONNECTIONS (thumb, index, middle, ring, pinky)
BONE_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
LANDMARK_FINGER = {
    0: "thumb", 6: "thumb", 7: "thumb",
    1: "index", 8: "index", 9: "index", 10: "index",
    2: "middle", 11: "middle", 12: "middle", 13: "middle",
    3: "ring", 14: "ring", 15: "ring", 16: "ring",
    4: "pinky", 17: "pinky", 18: "pinky", 19: "pinky",
    5: "palm", 20: "palm",
}


def _side_from_meta(d, fallback="left") -> str:
    for key in ("group", "source_file", "label"):
        if key in d.files:
            s = str(d[key]).lower()
            if "right" in s:
                return "right"
            if "left" in s:
                return "left"
    return fallback


def _skeleton_traces(pos, show_legend=False):
    """Scatter (joints) + line (bones + palm) traces for one frame."""
    traces = []
    marker_colors = [FINGER_COLORS[LANDMARK_FINGER[i]] for i in range(21)]
    hover = [f"{LANDMARK_NAMES[i]}<br>({pos[i,0]:.1f}, {pos[i,1]:.1f}, {pos[i,2]:.1f})"
             for i in range(21)]
    traces.append(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers",
        marker=dict(size=5, color=marker_colors, opacity=0.95),
        hovertext=hover, hoverinfo="text", showlegend=False, name="joints",
    ))
    for bone, finger in zip(BONE_CONNECTIONS, BONE_FINGERS):
        traces.append(go.Scatter3d(
            x=pos[bone, 0], y=pos[bone, 1], z=pos[bone, 2], mode="lines",
            line=dict(color=FINGER_COLORS[finger], width=6),
            showlegend=False, hoverinfo="skip",
        ))
    for conn in PALM_CONNECTIONS:
        c = np.asarray(conn)
        traces.append(go.Scatter3d(
            x=pos[c, 0], y=pos[c, 1], z=pos[c, 2], mode="lines",
            line=dict(color=FINGER_COLORS["palm"], width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
    return traces


def animate(npz_path, out_path, side=None, max_frames=80, fps=30):
    d = np.load(npz_path, allow_pickle=True)
    ja = np.asarray(d["joint_angles"], dtype=float)        # (T, 20)
    fs = int(d["fs"]) if "fs" in d.files else 2000
    label = str(d["label"]) if "label" in d.files else "?"
    cid = str(d["cluster_id"]) if "cluster_id" in d.files else "?"
    src = str(d["source_file"]) if "source_file" in d.files else ""
    if side is None:
        side = _side_from_meta(d)

    T = ja.shape[0]
    # Downsample to <= max_frames evenly spaced frames.
    step = max(1, T // max_frames)
    idxs = list(range(0, T, step))[:max_frames]
    pos_all = angles_batch_to_landmarks(ja[idxs], side=side)  # (F, 21, 3) mm

    # Fixed axis ranges across all frames (cube, true aspect).
    pts = pos_all.reshape(-1, 3)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    mid = (mn + mx) / 2.0
    half = (mx - mn).max() / 2.0 * 1.1
    lo, hi = mid - half, mid + half

    # Initial frame + animation frames.
    init = _skeleton_traces(pos_all[0])
    frames = []
    for fi, src_idx in enumerate(idxs):
        frames.append(go.Frame(
            data=_skeleton_traces(pos_all[fi]),
            name=str(fi),
            layout=go.Layout(title=f"{label}  |  t={src_idx / fs:.3f}s  "
                                   f"(frame {src_idx}/{T})"),
        ))
    fig = go.Figure(data=init, frames=frames)

    fig.update_layout(
        title=f"{label}  |  t=0.000s (frame 0/{T})",
        height=720, width=820,
        scene=dict(
            xaxis=dict(range=[lo[0], hi[0]], title="X (mm)"),
            yaxis=dict(range=[lo[1], hi[1]], title="Y (mm)"),
            zaxis=dict(range=[lo[2], hi[2]], title="Z (mm)"),
            aspectmode="cube",
            camera=dict(eye=dict(x=-0.25, y=-1.6, z=0.6)),
        ),
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.1, "y": 0,
            "pad": {"r": 10, "t": 70},
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": int(1000 / fps),
                                           "redraw": True},
                                 "fromcurrent": True,
                                 "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0, "x": 0.1, "y": 0, "len": 0.9,
            "pad": {"b": 10, "t": 60},
            "currentvalue": {"prefix": "frame ", "visible": True},
            "steps": [{"label": str(fi), "method": "animate",
                       "args": [[str(fi)], {"frame": {"duration": 0,
                                                      "redraw": True},
                                            "mode": "immediate"}]}
                      for fi in range(len(idxs))],
        }],
        annotations=[dict(
            x=0, y=1, xref="paper", yref="paper", showarrow=False, align="left",
            text=f"side={side}  cluster_id={cid}<br>{src}",
            font=dict(size=11), bgcolor="rgba(255,255,255,0.6)")],
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, auto_open=False)
    print(f"Wrote {out_path}")
    print(f"  T={T} ({T / fs:.2f}s)  frames={len(idxs)} (step={step})  side={side}")


def _derive_out_root(npz_path: str) -> str:
    """If npz is <root>/segments/<label>/file.npz -> <root>, else its dir."""
    label_dir = os.path.dirname(os.path.abspath(npz_path))
    segments_dir = os.path.dirname(label_dir)
    if os.path.basename(segments_dir) == "segments":
        return os.path.dirname(segments_dir)
    return label_dir


def main():
    ap = argparse.ArgumentParser(
        description="Animated 3D hand pose (Plotly HTML) for a segment npz")
    ap.add_argument("npz", help="path to a segment .npz")
    ap.add_argument("--out-root", default=None,
                    help="output root (default: auto-derive from npz path)")
    ap.add_argument("--subdir", default="hand_anim",
                    help="subdir under out-root (default: hand_anim)")
    ap.add_argument("--side", choices=["left", "right"], default=None,
                    help="hand side (default: inferred from metadata)")
    ap.add_argument("--max-frames", type=int, default=80,
                    help="max animation frames (downsampled, default 80)")
    ap.add_argument("--fps", type=int, default=30,
                    help="playback frames per second (default 30)")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    label = str(d["label"]) if "label" in d.files else "unlabeled"
    stem = os.path.splitext(os.path.basename(args.npz))[0]
    out_root = args.out_root or _derive_out_root(args.npz)
    # Output: <out_root>/<subdir>/<label>/<stem>.html
    out_path = os.path.join(out_root, args.subdir, label, stem + ".html")

    animate(args.npz, out_path, side=args.side,
            max_frames=args.max_frames, fps=args.fps)


if __name__ == "__main__":
    main()
