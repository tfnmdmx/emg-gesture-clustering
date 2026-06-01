from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_ROOT = Path("/data/dyh_data/raw_data_subgroup")
SESSION_RE = re.compile(r"^(?P<date>\d{8})-(?P<hand>left|right)(?:-(?P<batch>\d+))?$")
SKELETON_CONNECTIONS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24],
]
SKELETON_PALM_CONNECTIONS = [[0, 5, 10, 15, 20]]
SKELETON_FINGER_COLORS = ["#CC6677", "#4477AA", "#228833", "#EE6677", "#AA3377"]
SKELETON_ROOT_INDICES = [0, 5, 10, 15, 20]


@dataclass(frozen=True)
class SegmentationConfig:
    smooth_sec: float = 0.25
    threshold_percentile: float = 35.0
    threshold_mad_scale: float = 1.5
    min_static_sec: float = 0.35
    min_motion_sec: float = 0.20
    merge_gap_sec: float = 0.20
    pad_sec: float = 0.05


def discover_npz_files(root: Path = DATA_ROOT) -> pd.DataFrame:
    rows = []
    for c_dir in sorted(root.glob("C*"), key=lambda p: int(p.name[1:]) if re.fullmatch(r"C\d+", p.name) else 999):
        if not c_dir.is_dir() or not re.fullmatch(r"C\d+", c_dir.name):
            continue
        for person_dir in sorted(p for p in c_dir.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in person_dir.iterdir() if p.is_dir()):
                match = SESSION_RE.match(session_dir.name)
                if not match:
                    continue
                for npz_path in sorted(session_dir.rglob("*.npz")):
                    rows.append(
                        {
                            "C": c_dir.name,
                            "person": person_dir.name,
                            "date": match.group("date"),
                            "hand": match.group("hand"),
                            "batch": int(match.group("batch") or 1),
                            "session": session_dir.name,
                            "path": npz_path,
                        }
                    )
    return pd.DataFrame(rows)


def _first_existing_key(keys: Iterable[str], candidates: Iterable[str]) -> str | None:
    key_set = set(keys)
    for key in candidates:
        if key in key_set:
            return key
    return None


def find_pose_key(npz: np.lib.npyio.NpzFile, hand: str | None = None) -> str:
    keys = npz.files
    if hand:
        preferred = f"manus_{hand}_ergonomics"
        if preferred in keys:
            return preferred
    candidates = [k for k in keys if k.startswith("manus_") and k.endswith("_ergonomics")]
    if candidates:
        return sorted(candidates)[0]
    if "joint_angles" in keys:
        return "joint_angles"
    raise KeyError("No pose key found. Expected manus_*_ergonomics or joint_angles.")


def find_skeleton_key(npz: np.lib.npyio.NpzFile, hand: str | None = None) -> str | None:
    candidates = [k for k in npz.files if k.startswith("manus_") and k.endswith("_skeleton")]
    if hand is not None:
        preferred = f"manus_{hand}_skeleton"
        if preferred in candidates:
            return preferred
    return sorted(candidates)[0] if candidates else None


def load_skeleton_sequence(path: Path, hand: str | None = None) -> tuple[np.ndarray, str]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as npz:
        skeleton_key = find_skeleton_key(npz, hand=hand)
        if skeleton_key is None:
            raise KeyError(f"{path} has no manus_*_skeleton array.")
        skeleton = np.asarray(npz[skeleton_key], dtype=np.float32)
    if skeleton.ndim != 3 or skeleton.shape[1:] != (25, 7):
        raise ValueError(f"Expected skeleton shape (T, 25, 7), got {skeleton.shape} in {path}.")
    return skeleton, skeleton_key


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.divide(q, norm, out=np.zeros_like(q, dtype=np.float32), where=norm > 0)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    ).astype(np.float32)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def skeleton_to_world_xyz(skeleton: np.ndarray) -> np.ndarray:
    """Convert local chain translations/quaternions to world joint xyz.

    Keep this available for skeleton arrays that are stored as local joint
    offsets. The Manus skeleton files in this dataset already store usable
    per-joint xyz positions in the first three channels, so rendering should
    normally use those positions directly instead of applying this conversion.
    """
    skeleton = np.asarray(skeleton, dtype=np.float32)
    local_t = skeleton[..., :3]
    local_q = _quat_normalize(skeleton[..., 3:7])
    world_t = np.full_like(local_t, np.nan, dtype=np.float32)
    world_q = np.full_like(local_q, np.nan, dtype=np.float32)

    for chain in SKELETON_CONNECTIONS:
        root = chain[0]
        world_t[:, root] = local_t[:, root]
        world_q[:, root] = local_q[:, root]
        for parent, child in zip(chain[:-1], chain[1:]):
            rotated_offset = _quat_rotate(world_q[:, parent], local_t[:, child])
            world_t[:, child] = world_t[:, parent] + rotated_offset
            world_q[:, child] = _quat_multiply(world_q[:, parent], local_q[:, child])
    return world_t


def _linspace_time(length: int, duration: float | None = None) -> np.ndarray:
    if duration is None or not np.isfinite(duration) or duration <= 0:
        duration = max((length - 1) / 2000.0, 0.0)
    return np.linspace(0.0, float(duration), int(length), dtype=np.float64)


def _timestamp_preserves_record_offset(key: str) -> bool:
    return key.endswith("_raw") or key.endswith("_rel") or "_timestamps_raw" in key or "_ts_rel" in key


def _time_from_npz(
    npz: np.lib.npyio.NpzFile,
    length: int,
    preferred_keys: Iterable[str],
    fallback_duration: float | None,
) -> np.ndarray:
    for key in preferred_keys:
        if not key or key not in npz.files:
            continue
        values = np.asarray(npz[key], dtype=np.float64)
        if values.ndim != 1 or len(values) < 2:
            continue
        if not np.all(np.isfinite(values)) or np.nanmax(values) <= np.nanmin(values):
            continue
        if len(values) != length:
            src_x = np.linspace(0.0, 1.0, len(values))
            dst_x = np.linspace(0.0, 1.0, length)
            values = np.interp(dst_x, src_x, values)
        if not _timestamp_preserves_record_offset(key):
            values = values - values[0]
        if np.all(np.isfinite(values)) and np.nanmax(values) > np.nanmin(values):
            return values
    return _linspace_time(length, fallback_duration)


def load_record(path: Path, hand: str | None = None) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=True) as npz:
        if "emg" not in npz.files:
            raise KeyError(f"{path} has no 'emg' array.")
        emg = np.asarray(npz["emg"], dtype=np.float32)
        pose_key = find_pose_key(npz, hand)
        pose = np.asarray(npz[pose_key], dtype=np.float32)

        duration = None
        if "duration" in npz.files:
            duration = float(np.asarray(npz["duration"]))
        elif "record_t0" in npz.files and "record_t1" in npz.files:
            duration = float(np.asarray(npz["record_t1"]) - np.asarray(npz["record_t0"]))

        emg_t = _time_from_npz(npz, len(emg), ["emg_ts_rel", "emg_timestamps_raw", "emg_timestamps"], duration)
        pose_t = _time_from_npz(
            npz,
            len(pose),
            [
                "manus_ts_rel",
                "manus_timestamps_raw",
                "manus_timestamps",
                f"manus_{hand}_timestamps_raw" if hand else "",
                f"manus_{hand}_ts_rel" if hand else "",
                f"manus_{hand}_timestamps_rel" if hand else "",
                f"manus_{hand}_timestamps" if hand else "",
            ],
            duration if duration is not None else (emg_t[-1] if len(emg_t) else None),
        )

        if len(pose_t) and len(emg_t):
            if np.isfinite(pose_t[-1]) and np.isfinite(emg_t[-1]) and pose_t[-1] > 0 and emg_t[-1] > 0:
                pose_t = pose_t * (emg_t[-1] / pose_t[-1])

        force = np.asarray(npz["force"], dtype=np.float32) if "force" in npz.files else None

    return {
        "path": path,
        "emg": emg,
        "emg_t": emg_t,
        "pose": pose,
        "pose_t": pose_t,
        "pose_key": pose_key,
        "force": force,
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = max(int(window), 1)
    if window <= 1 or len(values) < 3:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def motion_speed(pose: np.ndarray, pose_t: np.ndarray, smooth_sec: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose, dtype=np.float64)
    pose_t = np.asarray(pose_t, dtype=np.float64)
    if len(pose) != len(pose_t):
        raise ValueError("pose and pose_t must have the same length.")
    if len(pose) < 2:
        return pose_t, np.zeros(len(pose), dtype=np.float64)

    pose = np.nan_to_num(pose, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    dt = np.diff(pose_t)
    good_dt = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(good_dt)) if len(good_dt) else 1.0 / 120.0
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, median_dt)
    delta = np.linalg.norm(np.diff(pose, axis=0), axis=1) / dt
    delta = np.nan_to_num(delta, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.concatenate([[delta[0]], delta])
    smooth_window = max(1, round(smooth_sec / median_dt))
    return pose_t, moving_average(speed, smooth_window)


def robust_speed_threshold(speed: np.ndarray, config: SegmentationConfig) -> float:
    speed = np.asarray(speed, dtype=np.float64)
    finite = speed[np.isfinite(speed)]
    if len(finite) == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    mad_sigma = 1.4826 * mad
    pct = float(np.percentile(finite, config.threshold_percentile))
    return max(pct, median + config.threshold_mad_scale * mad_sigma)


def _mask_to_intervals(mask: np.ndarray, t: np.ndarray, min_sec: float, merge_gap_sec: float) -> list[tuple[float, float]]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    intervals: list[tuple[float, float]] = []
    for start_idx, end_idx in changes.reshape(-1, 2):
        start = float(t[start_idx])
        end = float(t[min(end_idx - 1, len(t) - 1)])
        if end - start >= min_sec:
            intervals.append((start, end))

    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap_sec:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _time_to_index(t: np.ndarray, start: float, end: float) -> tuple[int, int]:
    left = int(np.searchsorted(t, start, side="left"))
    right = int(np.searchsorted(t, end, side="right"))
    return max(left, 0), min(max(right, left + 1), len(t))


def segment_record(path: Path, metadata: dict | None = None, config: SegmentationConfig | None = None) -> tuple[pd.DataFrame, dict]:
    config = config or SegmentationConfig()
    metadata = metadata or {}
    record = load_record(path, metadata.get("hand"))
    pose_t, speed = motion_speed(record["pose"], record["pose_t"], config.smooth_sec)
    threshold = robust_speed_threshold(speed, config)

    static_intervals = _mask_to_intervals(
        speed <= threshold,
        pose_t,
        min_sec=config.min_static_sec,
        merge_gap_sec=config.merge_gap_sec,
    )

    total_start = float(pose_t[0]) if len(pose_t) else 0.0
    total_end = float(pose_t[-1]) if len(pose_t) else 0.0
    motion_intervals: list[tuple[float, float]] = []
    cursor = total_start
    for start, end in static_intervals:
        if start - cursor >= config.min_motion_sec:
            motion_intervals.append((cursor, start))
        cursor = max(cursor, end)
    if total_end - cursor >= config.min_motion_sec:
        motion_intervals.append((cursor, total_end))

    rows = []
    for segment_type, intervals in [("static_hold", static_intervals), ("transition_motion", motion_intervals)]:
        for idx, (start, end) in enumerate(intervals, start=1):
            padded_start = max(total_start, start - config.pad_sec)
            padded_end = min(total_end, end + config.pad_sec)
            pose_start, pose_end = _time_to_index(record["pose_t"], padded_start, padded_end)
            emg_start, emg_end = _time_to_index(record["emg_t"], padded_start, padded_end)
            pose_slice = record["pose"][pose_start:pose_end]
            emg_slice = record["emg"][emg_start:emg_end]
            rows.append(
                {
                    **metadata,
                    "path": str(path),
                    "pose_key": record["pose_key"],
                    "segment_type": segment_type,
                    "segment_index": idx,
                    "gesture_label": "",
                    "start_s": padded_start,
                    "end_s": padded_end,
                    "duration_s": padded_end - padded_start,
                    "pose_start": pose_start,
                    "pose_end": pose_end,
                    "emg_start": emg_start,
                    "emg_end": emg_end,
                    "pose_frames": max(pose_end - pose_start, 0),
                    "emg_frames": max(emg_end - emg_start, 0),
                    "mean_speed": float(np.mean(speed[pose_start:pose_end])) if pose_end > pose_start else np.nan,
                    "max_speed": float(np.max(speed[pose_start:pose_end])) if pose_end > pose_start else np.nan,
                    "emg_rms": float(np.sqrt(np.mean(np.square(emg_slice)))) if emg_slice.size else np.nan,
                    "pose_mean": np.mean(pose_slice, axis=0) if len(pose_slice) else np.full(record["pose"].shape[1], np.nan),
                }
            )

    segments = pd.DataFrame(rows).sort_values(["start_s", "segment_type"]).reset_index(drop=True)
    debug = {
        **record,
        "speed_t": pose_t,
        "speed": speed,
        "threshold": threshold,
        "static_intervals": static_intervals,
        "motion_intervals": motion_intervals,
        "config": config,
    }
    return segments, debug


def batch_segment(files_df: pd.DataFrame, config: SegmentationConfig | None = None, limit: int | None = None) -> pd.DataFrame:
    rows = []
    iterator = files_df.head(limit).iterrows() if limit else files_df.iterrows()
    for _, row in iterator:
        metadata = {k: row[k] for k in ["C", "person", "date", "hand", "batch", "session"] if k in row}
        try:
            segments, _ = segment_record(Path(row["path"]), metadata=metadata, config=config)
        except Exception as exc:
            rows.append({**metadata, "path": str(row["path"]), "error": str(exc)})
            continue
        rows.append(segments.drop(columns=["pose_mean"], errors="ignore"))
    if not rows:
        return pd.DataFrame()
    frames = [r if isinstance(r, pd.DataFrame) else pd.DataFrame([r]) for r in rows]
    return pd.concat(frames, ignore_index=True)


def build_static_motion_static_clips(
    segments: pd.DataFrame,
    pre_static_sec: float | None = 0.5,
    post_static_sec: float | None = 0.5,
    min_motion_sec: float = 0.2,
) -> pd.DataFrame:
    """Combine adjacent static/motion/static intervals into action clips.

    The returned rows keep the same column contract as segment rows, so they can
    be passed to export/visualization helpers. ``start_s`` and ``end_s`` cover
    the full static-motion-static window, while ``motion_start_s`` and
    ``motion_end_s`` mark the active movement inside the clip.
    """
    if segments.empty:
        return segments.copy()

    ordered = segments.sort_values("start_s").reset_index(drop=True)
    rows = []
    clip_index = 1
    for i in range(1, len(ordered) - 1):
        prev_row = ordered.iloc[i - 1]
        motion_row = ordered.iloc[i]
        next_row = ordered.iloc[i + 1]
        if (
            prev_row.get("segment_type") != "static_hold"
            or motion_row.get("segment_type") != "transition_motion"
            or next_row.get("segment_type") != "static_hold"
        ):
            continue
        if float(motion_row["duration_s"]) < min_motion_sec:
            continue

        start_s = float(prev_row["start_s"])
        end_s = float(next_row["end_s"])
        if pre_static_sec is not None:
            start_s = max(start_s, float(prev_row["end_s"]) - float(pre_static_sec))
        if post_static_sec is not None:
            end_s = min(end_s, float(next_row["start_s"]) + float(post_static_sec))

        record = load_record(Path(motion_row["path"]), motion_row.get("hand"))
        pose_start, pose_end = _time_to_index(record["pose_t"], start_s, end_s)
        emg_start, emg_end = _time_to_index(record["emg_t"], start_s, end_s)
        pose_slice = record["pose"][pose_start:pose_end]
        emg_slice = record["emg"][emg_start:emg_end]

        row = motion_row.to_dict()
        row.update(
            {
                "segment_type": "static_motion_static",
                "segment_index": clip_index,
                "gesture_label": row.get("gesture_label", ""),
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "pose_start": pose_start,
                "pose_end": pose_end,
                "emg_start": emg_start,
                "emg_end": emg_end,
                "pose_frames": max(pose_end - pose_start, 0),
                "emg_frames": max(emg_end - emg_start, 0),
                "pre_static_index": int(prev_row["segment_index"]),
                "motion_index": int(motion_row["segment_index"]),
                "post_static_index": int(next_row["segment_index"]),
                "pre_static_start_s": float(prev_row["start_s"]),
                "pre_static_end_s": float(prev_row["end_s"]),
                "motion_start_s": float(motion_row["start_s"]),
                "motion_end_s": float(motion_row["end_s"]),
                "post_static_start_s": float(next_row["start_s"]),
                "post_static_end_s": float(next_row["end_s"]),
                "emg_rms": float(np.sqrt(np.mean(np.square(emg_slice)))) if emg_slice.size else np.nan,
                "pose_mean": np.mean(pose_slice, axis=0) if len(pose_slice) else row.get("pose_mean"),
            }
        )
        rows.append(row)
        clip_index += 1

    return pd.DataFrame(rows)


def export_segment_npz(segment_row: pd.Series, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = load_record(Path(segment_row["path"]), segment_row.get("hand"))
    emg = record["emg"][int(segment_row["emg_start"]): int(segment_row["emg_end"])]
    pose = record["pose"][int(segment_row["pose_start"]): int(segment_row["pose_end"])]
    stem = Path(segment_row["path"]).stem
    label = segment_row.get("gesture_label") or segment_row["segment_type"]
    out = output_dir / f"{segment_row.get('person', 'unknown')}__{stem}__{label}_{int(segment_row['segment_index']):03d}.npz"
    np.savez_compressed(
        out,
        emg=emg,
        pose=pose,
        emg_start=int(segment_row["emg_start"]),
        emg_end=int(segment_row["emg_end"]),
        pose_start=int(segment_row["pose_start"]),
        pose_end=int(segment_row["pose_end"]),
        start_s=float(segment_row["start_s"]),
        end_s=float(segment_row["end_s"]),
        segment_type=str(segment_row["segment_type"]),
        gesture_label=str(segment_row.get("gesture_label", "")),
        source_path=str(segment_row["path"]),
    )
    return out


def skeleton_segment_xyz(
    segment_row: pd.Series,
    center_on_roots: bool = True,
    scale: float = 1000.0,
    max_frames: int | None = 240,
    use_forward_kinematics: bool = False,
) -> tuple[np.ndarray, str]:
    skeleton, skeleton_key = load_skeleton_sequence(Path(segment_row["path"]), segment_row.get("hand"))
    start = int(segment_row["pose_start"])
    end = int(segment_row["pose_end"])
    segment = skeleton[start:end]
    xyz = skeleton_to_world_xyz(segment) if use_forward_kinematics else np.asarray(segment[:, :, :3], dtype=np.float32)

    finite = np.all(np.isfinite(xyz), axis=(1, 2))
    xyz = xyz[finite]
    if len(xyz) == 0:
        raise ValueError("No finite skeleton frames in this segment.")

    if center_on_roots:
        roots = xyz[:, SKELETON_ROOT_INDICES, :]
        root_center = np.nanmean(roots, axis=1, keepdims=True)
        xyz = xyz - root_center

    xyz = xyz * scale
    if max_frames is not None and max_frames > 0 and len(xyz) > max_frames:
        sample_idx = np.linspace(0, len(xyz) - 1, max_frames).round().astype(int)
        xyz = xyz[sample_idx]
    return xyz, skeleton_key


def skeleton_axis_limits(xyz: np.ndarray, margin_ratio: float = 0.14) -> list[tuple[float, float]]:
    xyz = np.asarray(xyz, dtype=np.float32)
    mins = np.nanmin(xyz, axis=(0, 1))
    maxs = np.nanmax(xyz, axis=(0, 1))
    center = (mins + maxs) / 2.0
    span = float(np.nanmax(maxs - mins))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    half = span * (0.5 + margin_ratio)
    return [(float(center[i] - half), float(center[i] + half)) for i in range(3)]


def draw_skeleton_frame(ax, frame_xyz: np.ndarray, limits: list[tuple[float, float]] | None = None) -> None:
    for conn, color in zip(SKELETON_CONNECTIONS, SKELETON_FINGER_COLORS):
        pts = frame_xyz[conn]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=3)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=22)

    for conn in SKELETON_PALM_CONNECTIONS:
        pts = frame_xyz[conn]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#666666", linewidth=1.4, linestyle="--", alpha=0.65)

    if limits is not None:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-72)
    ax.set_axis_off()


def skeleton_motion_score(segment_row: pd.Series) -> float:
    xyz, _ = skeleton_segment_xyz(segment_row, max_frames=None)
    if len(xyz) < 2:
        return 0.0
    start_to_end = np.linalg.norm(xyz[-1] - xyz[0], axis=1)
    per_frame = np.linalg.norm(np.diff(xyz, axis=0), axis=2)
    return float(np.nanmean(start_to_end) + np.nanpercentile(per_frame, 95))


def plot_skeleton_keyframes(
    segment_row: pd.Series,
    n_frames: int = 6,
    center_on_roots: bool = True,
    scale: float = 1000.0,
):
    import matplotlib.pyplot as plt

    xyz, skeleton_key = skeleton_segment_xyz(
        segment_row,
        center_on_roots=center_on_roots,
        scale=scale,
        max_frames=max(n_frames, 1),
    )
    if len(xyz) < n_frames:
        frame_idx = np.arange(len(xyz))
    else:
        frame_idx = np.linspace(0, len(xyz) - 1, n_frames).round().astype(int)
    limits = skeleton_axis_limits(xyz)

    fig = plt.figure(figsize=(3.0 * len(frame_idx), 3.2))
    for plot_i, idx in enumerate(frame_idx, start=1):
        ax = fig.add_subplot(1, len(frame_idx), plot_i, projection="3d")
        draw_skeleton_frame(ax, xyz[idx], limits)
        ax.set_title(f"frame {idx}")
    fig.suptitle(
        f"{segment_row.get('person', '')} {segment_row.get('segment_type', '')} "
        f"#{segment_row.get('segment_index', '')} ({skeleton_key})"
    )
    fig.tight_layout()
    return fig


def make_skeleton_animation(
    segment_row: pd.Series,
    fps: int = 30,
    max_frames: int = 180,
    center_on_roots: bool = True,
    scale: float = 1000.0,
    show_tip_trails: bool = True,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    xyz, skeleton_key = skeleton_segment_xyz(
        segment_row,
        center_on_roots=center_on_roots,
        scale=scale,
        max_frames=max_frames,
    )
    limits = skeleton_axis_limits(xyz)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()
        if show_tip_trails and i > 0:
            tips = xyz[: i + 1, [4, 9, 14, 19, 24], :]
            for tip_i, color in enumerate(SKELETON_FINGER_COLORS):
                trail = tips[:, tip_i]
                ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=color, linewidth=1.2, alpha=0.35)
        draw_skeleton_frame(ax, xyz[i], limits)
        ax.set_title(f"{skeleton_key} | frame {i + 1}/{len(xyz)}")
        return []

    anim = FuncAnimation(fig, update, frames=len(xyz), interval=1000 / fps, blit=False)
    plt.close(fig)
    return anim


def render_skeleton_segment_frames(
    segment_row: pd.Series,
    fps: int = 30,
    max_frames: int = 180,
    width: int = 720,
    height: int = 560,
    center_on_roots: bool = True,
    scale: float = 1000.0,
    show_tip_trails: bool = True,
) -> tuple[np.ndarray, str]:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    xyz, skeleton_key = skeleton_segment_xyz(
        segment_row,
        center_on_roots=center_on_roots,
        scale=scale,
        max_frames=max_frames,
    )
    limits = skeleton_axis_limits(xyz)
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    frames = []
    try:
        for i, frame_xyz in enumerate(xyz, start=1):
            ax.clear()
            if show_tip_trails and i > 1:
                tips = xyz[:i, [4, 9, 14, 19, 24], :]
                for tip_i, color in enumerate(SKELETON_FINGER_COLORS):
                    trail = tips[:, tip_i]
                    ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=color, linewidth=1.2, alpha=0.35)
            draw_skeleton_frame(ax, frame_xyz, limits)
            ax.set_title(f"{skeleton_key} | frame {i}/{len(xyz)}")
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            canvas.draw()
            frames.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
    finally:
        plt.close(fig)
    return np.asarray(frames, dtype=np.uint8), skeleton_key


def render_skeleton_segment_video(
    segment_row: pd.Series,
    output_dir: Path = Path("gesture_skeleton_videos"),
    fps: int = 30,
    max_frames: int = 180,
    width: int = 720,
    height: int = 560,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stem = Path(segment_row["path"]).stem
    person = str(segment_row.get("person", "unknown"))
    segment_type = str(segment_row.get("segment_type", "segment"))
    segment_index = int(segment_row.get("segment_index", 0))
    out_path = output_dir / f"{person}__{source_stem}__{segment_type}_{segment_index:03d}.mp4"
    if out_path.exists() and not overwrite:
        return out_path

    frames, _ = render_skeleton_segment_frames(
        segment_row,
        fps=fps,
        max_frames=max_frames,
        width=width,
        height=height,
    )
    try:
        import mediapy

        mediapy.write_video(out_path, frames, fps=fps)
    except Exception:
        import imageio.v2 as imageio

        imageio.mimsave(out_path, frames, fps=fps)
    return out_path
