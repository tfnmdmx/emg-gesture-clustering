from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class FileInfo:
    path: str
    stem: str
    subject: str | None
    hand: str | None
    group: str
    parsed: bool


_SESSION_RE = re.compile(
    r"^(?P<date>\d{8})-(?P<hand>left|right|both)(?:-(?P<batch>\d+))?$",
    re.IGNORECASE,
)
# processed_data batch dir like 'fgw0917_0502_left', 'zyb0201_20260514',
# 'ghd1108_20260513_redmi': leading token is the subject with no hyphen.
_BATCH_RE = re.compile(r"^(?P<subj>[A-Za-z]{2,}\d{3,})(?:_(?P<rest>.+))?$")


def normalize_subject(s) -> str:
    """Subject id reduced to lowercase alphanumerics for matching.

    Canonical ids carry a hyphen (``fgw-0917``) while processed_data batch dirs
    drop it (``fgw0917``); both normalize to ``fgw0917`` so the same ``--subjects``
    value selects from a meta CSV and from processed_data alike.
    """
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def parse_file_info(path: str) -> FileInfo:
    """Parse subject / hand / group from path.

    Tries three layouts in order:

    1. **Processed** filename ``{subject}__{date}-{hand}[-{run}|_{note}...]__{timestamp}``.
       Used by data already aligned to the EMG sample axis.

    2. **Raw** path ``.../{subject}/{date}-{hand}[-{run}]/{timestamp}.npz`` --
       hand lives in the parent directory name (the filename is just a
       timestamp). The synthesized stem is
       ``{subject}__{session}__{filename_stem}`` so it stays unique and matches
       the processed-format pattern downstream.

    3. **Processed batch dir** ``.../{subj}{date}[_{hand}|_{note}]/{timestamp}.npz``
       (under processed_data/) -- subject lives in the dir name with no hyphen and
       the filename is a bare timestamp. The subject is re-hyphenated to the
       canonical form (``fgw0917`` -> ``fgw-0917``) and the stem synthesized
       ``{subject}__{batchdir}__{filename_stem}`` to stay unique across batches.
    """
    p = Path(path)
    stem = p.stem

    # --- Processed: filename has __subject__date-hand__ ---
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
        elif hand_lower.startswith("both"):
            hand = "both"
        else:
            hand = hand_raw
        return FileInfo(path, stem, subject, hand, f"{subject}-{hand}", True)

    # --- Raw: hand lives in parent dir like 20260423-left[-batch] ---
    session_dir = p.parent.name
    subject_dir = p.parent.parent.name
    m = _SESSION_RE.match(session_dir)
    if m and subject_dir:
        hand = m.group("hand").lower()
        subject = subject_dir
        synth = f"{subject}__{session_dir}__{stem}"
        return FileInfo(path, synth, subject, hand, f"{subject}-{hand}", True)

    # --- Processed batch dir: subject in the dir name (no hyphen), bare ts file ---
    bm = _BATCH_RE.match(session_dir)
    if bm:
        # 'fgw0917' -> 'fgw-0917' (hyphen before the trailing digit run).
        subject = re.sub(r"^([A-Za-z]+)(\d+)$", r"\1-\2", bm.group("subj"))
        rest = (bm.group("rest") or "").lower().split("_")
        hand = next((h for h in rest if h in ("left", "right", "both")), None)
        synth = f"{subject}__{session_dir}__{stem}"
        group = f"{subject}-{hand}" if hand else subject
        return FileInfo(path, synth, subject, hand, group, True)

    warnings.warn(f"Cannot parse filename, treating as own group: {stem}")
    return FileInfo(path, stem, None, None, stem, False)


def group_files(paths: list[str]) -> dict[str, list[FileInfo]]:
    groups: dict[str, list[FileInfo]] = {}
    for p in paths:
        info = parse_file_info(p)
        groups.setdefault(info.group, []).append(info)
    return groups


def _find_pose_key(files: Iterable[str], hand: str | None) -> str | None:
    files = set(files)
    if hand and hand in ("left", "right"):
        cand = f"manus_{hand}_ergonomics"
        if cand in files:
            return cand
    pool = sorted(k for k in files
                  if k.startswith("manus_") and k.endswith("_ergonomics"))
    return pool[0] if pool else None


def _timestamp_preserves_record_offset(key: str) -> bool:
    return (key.endswith("_raw") or key.endswith("_rel")
            or "_timestamps_raw" in key or "_ts_rel" in key)


def _time_axis(npz, length: int, preferred_keys: list[str],
               fallback_duration: float | None) -> np.ndarray:
    """Per-sample time array of size ``length``.

    Walks ``preferred_keys`` looking for a usable timestamp array; linearly
    interpolates if the array's length doesn't match ``length`` (raw data
    often stores one timestamp per block). Falls back to ``linspace(0,
    fallback_duration, length)`` when no timestamp key is usable.
    """
    keys = set(npz.files)
    for key in preferred_keys:
        if not key or key not in keys:
            continue
        vals = np.asarray(npz[key], dtype=np.float64)
        if vals.ndim != 1 or len(vals) < 2:
            continue
        if not np.all(np.isfinite(vals)) or float(np.nanmax(vals)) <= float(np.nanmin(vals)):
            continue
        if len(vals) != length:
            src_x = np.linspace(0.0, 1.0, len(vals))
            dst_x = np.linspace(0.0, 1.0, length)
            vals = np.interp(dst_x, src_x, vals)
        if not _timestamp_preserves_record_offset(key):
            vals = vals - vals[0]
        if float(vals[-1]) > float(vals[0]):
            return vals
    duration = (fallback_duration if fallback_duration and fallback_duration > 0
                else max((length - 1) / 2000.0, 1e-6))
    return np.linspace(0.0, float(duration), int(length))


def _align_pose_to_emg(pose: np.ndarray, pose_t: np.ndarray,
                       emg_t: np.ndarray) -> np.ndarray:
    """Linear-interp each pose column from ``pose_t`` onto ``emg_t``.

    Both axes are first rescaled to the same end-time so a small clock skew
    between modalities (a few ms over a 2-minute recording) doesn't propagate
    into the alignment. End-time rescale follows the reference implementation.
    """
    if pose_t[-1] > 0 and emg_t[-1] > 0:
        pose_t = pose_t * (emg_t[-1] / pose_t[-1])
    out = np.empty((len(emg_t), pose.shape[1]), dtype=np.float32)
    for d in range(pose.shape[1]):
        out[:, d] = np.interp(emg_t, pose_t, pose[:, d].astype(np.float64))
    return out


def resolve_npz_path(source_file: str, source_path=None,
                     input_dir: str | None = None) -> str:
    """Locate a recording's npz for a stage-2/3 reload.

    Prefers the absolute ``source_path`` recorded in the stage-1 CSVs, so the
    pipeline runs straight from a ``--meta`` CSV or a source directory WITHOUT a
    flat ``work_pool/`` -- matching the raw flow (and cluster_traj.py, which
    already reads ``source_path`` directly). Falls back to ``input_dir/source_file``
    for legacy pool-based invocations, then to the bare filename.
    """
    if source_path is not None:
        sp = str(source_path)
        if sp and sp.lower() not in ("nan", "none") and os.path.isfile(sp):
            return sp
    if input_dir:
        return os.path.join(input_dir, str(source_file))
    return str(source_file)


def load_npz(path: str, hand: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(emg, joint_angles)`` aligned on the EMG sample axis.

    Two npz layouts are auto-detected:

    * **Processed** (legacy): ``joint_angles`` already matches ``emg`` length.
      Returned as-is. A length mismatch raises (the contract for this layout
      is per-sample alignment).
    * **Raw** (reference data): pose lives under ``manus_{hand}_ergonomics``
      at a different length than ``emg``. Per-sample time axes are
      reconstructed for both modalities (from ``manus_*_timestamps_raw`` /
      ``record_t0/t1`` / uniform fallback) and the pose is linear-interpolated
      onto the EMG axis -- the rest of the pipeline (segmentation, features,
      export) stays length-agnostic at the array level.

    ``hand`` picks the pose key for files containing both ``manus_left_*`` and
    ``manus_right_*`` (typical of ``side=both`` recordings). If omitted, the
    side parsed from the path is used; if that's also missing, the first
    alphabetical ``manus_*_ergonomics`` key wins.
    """
    p = Path(path)
    d = np.load(p, allow_pickle=True)
    if "emg" not in d.files:
        raise KeyError(f"no 'emg' array in {path}")
    emg = np.asarray(d["emg"], dtype=np.float32)
    if emg.ndim != 2:
        raise ValueError(f"expected 2D emg in {path}, got {emg.shape}")

    # Processed path: explicit joint_angles, must match emg length.
    if "joint_angles" in d.files:
        ja = np.asarray(d["joint_angles"], dtype=np.float32)
        if ja.ndim != 2:
            raise ValueError(f"expected 2D joint_angles in {path}")
        if ja.shape[0] == emg.shape[0]:
            return emg, ja
        if not any(k.startswith("manus_") and k.endswith("_ergonomics")
                   for k in d.files):
            raise ValueError(
                f"joint_angles length mismatch in {path}: "
                f"{ja.shape[0]} vs {emg.shape[0]}")
        # fall through to raw alignment using manus_* keys

    if hand is None:
        hand = parse_file_info(str(p)).hand
    pose_key = _find_pose_key(d.files, hand)
    if pose_key is None:
        raise KeyError(
            f"no pose array found in {path} "
            f"(want joint_angles or manus_*_ergonomics)")
    pose = np.asarray(d[pose_key], dtype=np.float32)
    if pose.ndim != 2:
        raise ValueError(f"expected 2D {pose_key} in {path}, got {pose.shape}")

    duration: float | None = None
    if "record_t0" in d.files and "record_t1" in d.files:
        duration = float(d["record_t1"]) - float(d["record_t0"])
    elif "duration" in d.files:
        duration = float(d["duration"])

    emg_t = _time_axis(d, len(emg),
                       ["emg_ts_rel", "emg_timestamps_raw", "emg_timestamps"],
                       duration)
    pose_t = _time_axis(d, len(pose), [
        "manus_ts_rel", "manus_timestamps_raw", "manus_timestamps",
        f"manus_{hand}_ts_rel" if hand else "",
        f"manus_{hand}_timestamps_raw" if hand else "",
        f"manus_{hand}_timestamps" if hand else "",
    ], duration if duration is not None else float(emg_t[-1]))

    aligned = _align_pose_to_emg(pose, pose_t, emg_t)
    return emg, aligned


def load_skeleton(path: str, hand: str | None = None) -> np.ndarray | None:
    """Optional companion to ``load_npz``: ``(T_emg, 25, 3)`` world-XYZ.

    Returns ``None`` when the npz has no ``manus_*_skeleton`` key (typical of
    processed/legacy data). When present, the skeleton's first 3 channels are
    treated as world positions, aligned to the EMG sample axis via the same
    timestamp-aware interpolation as ``load_npz``, and returned so they can
    be indexed identically to the ``(emg, joint_angles)`` arrays.

    Used by ``export_clips.py`` for keyframe rendering without depending on
    torch / emg2pose FK: raw Manus data carries the skeleton directly.
    """
    p = Path(path)
    d = np.load(p, allow_pickle=True)
    if "emg" not in d.files:
        return None
    emg_len = int(np.asarray(d["emg"]).shape[0])

    if hand is None:
        hand = parse_file_info(str(p)).hand
    keys = set(d.files)
    cands: list[str] = []
    if hand in ("left", "right"):
        c = f"manus_{hand}_skeleton"
        if c in keys:
            cands.append(c)
    cands += sorted(k for k in keys
                    if k.startswith("manus_") and k.endswith("_skeleton")
                    and k not in cands)
    if not cands:
        return None
    skel = np.asarray(d[cands[0]], dtype=np.float32)
    if skel.ndim != 3 or skel.shape[1] != 25 or skel.shape[2] < 3:
        return None
    xyz = skel[..., :3]  # drop quaternion channels

    if xyz.shape[0] == emg_len:
        return xyz

    duration: float | None = None
    if "record_t0" in d.files and "record_t1" in d.files:
        duration = float(d["record_t1"]) - float(d["record_t0"])
    elif "duration" in d.files:
        duration = float(d["duration"])

    emg_t = _time_axis(d, emg_len,
                       ["emg_ts_rel", "emg_timestamps_raw", "emg_timestamps"],
                       duration)
    skel_t = _time_axis(d, xyz.shape[0], [
        "manus_ts_rel", "manus_timestamps_raw", "manus_timestamps",
        f"manus_{hand}_ts_rel" if hand else "",
        f"manus_{hand}_timestamps_raw" if hand else "",
        f"manus_{hand}_timestamps" if hand else "",
    ], duration if duration is not None else float(emg_t[-1]))
    if skel_t[-1] > 0 and emg_t[-1] > 0:
        skel_t = skel_t * (emg_t[-1] / skel_t[-1])

    out = np.empty((emg_len, 25, 3), dtype=np.float32)
    for j in range(25):
        for c in range(3):
            out[:, j, c] = np.interp(emg_t, skel_t,
                                     xyz[:, j, c].astype(np.float64))
    return out


def discover_npz(root: str, recursive: bool = False) -> list[str]:
    """List .npz files under ``root``.

    ``recursive=False`` matches ``root/*.npz``. ``recursive=True`` walks
    subdirectories -- useful for raw data laid out as
    ``{subject}/{date-hand}/{timestamp}.npz``.
    """
    root = Path(root)
    if not root.exists():
        return []
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(str(p) for p in root.glob(pattern))


def parse_meta_csv(meta_path: str) -> list[FileInfo]:
    """Read sample_meta.csv-style catalogue into ``FileInfo`` rows.

    Required columns: ``sample_path``, ``subject_id``, ``side``. Optional:
    ``sample_id`` (used as stem if present). ``side=both`` rows are emitted
    twice -- once with hand=left, once with hand=right -- so the loader can
    pick the matching ``manus_<hand>_ergonomics`` key for each pass.
    """
    import csv
    out: list[FileInfo] = []
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("sample_path", "").strip()
            if not path:
                continue
            subject = (row.get("subject_id") or "").strip() or None
            side = (row.get("side") or "").strip().lower()
            sample_id = (row.get("sample_id") or "").strip()
            stem = sample_id or Path(path).stem
            sides = (["left", "right"] if side == "both"
                     else [side] if side in ("left", "right")
                     else [None])
            for hand in sides:
                group = (f"{subject}-{hand}" if subject and hand else
                         stem)
                out.append(FileInfo(path, stem, subject, hand, group, True))
    return out
