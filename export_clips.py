from __future__ import annotations

"""Export each clip in clips.csv as a npz + 6-keyframe PNG for hand-labelling.

Workflow for building a ground-truth test set:
    1. ``python segment.py <input_dir> --out <out>``     -> writes clips.csv
    2. ``python export_clips.py <input_dir> --out <out>`` -> writes per-clip
       npz/png and ``index.html`` under ``<out>/clips_export/``.
    3. Open ``index.html`` to scan keyframes; fill the ``gesture_label`` column
       in ``<out>/clips.csv`` while browsing.

Each clip gets one PNG showing 6 hand poses (start, pre-motion, motion 1/3,
motion 2/3, apex, end) -- enough to recognise the gesture without opening a
renderer. Falls back gracefully when torch isn't installed: pass ``--no-png``
to skip rendering, or rely on the per-clip warning if torch fails at import.
"""

import argparse
import html
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from emg_label import io_utils  # noqa: E402


def _clip_key(stem: str, clip_id) -> str:
    return f"{stem}__c{int(clip_id):04d}"


def _keyframe_indices(row, n_clip: int):
    """Six within-clip sample indices: start, pre-motion, motion 1/3, motion 2/3,
    apex, end-1. Returned in clip-local coordinates (0..n_clip-1)."""
    cs = int(row["clip_start_sample"])
    ms = int(row["motion_start_sample"])
    me = int(row["motion_end_sample"])
    ap = int(row["apex_sample"])
    md = max(1, me - ms)
    abs_idx = [cs,
               max(cs, ms - 1),
               ms + md // 3,
               ms + 2 * md // 3,
               ap,
               cs + n_clip - 1]
    return [max(0, min(n_clip - 1, i - cs)) for i in abs_idx]


def _render_keyframes_skeleton(out_path: str, skel_clip: np.ndarray,
                               indices: list[int], titles: list[str]) -> None:
    """Render 6 keyframes from raw Manus skeleton XYZ (no torch needed)."""
    from emg_label.skeleton import (axis_limits, draw_skeleton,
                                    normalize_skeleton)

    frames = normalize_skeleton(skel_clip[indices])
    limits = axis_limits(frames)
    n = len(indices)
    fig = plt.figure(figsize=(2.4 * n, 2.6))
    for i, (frame, title) in enumerate(zip(frames, titles)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        draw_skeleton(ax, frame)
        ax.set_title(title, fontsize=8)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.view_init(elev=18, azim=-72)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def _render_keyframes_fk(out_path: str, ja_clip: np.ndarray,
                         indices: list[int], titles: list[str],
                         side: str) -> None:
    """Render 6 keyframes via emg2pose FK from joint angles (requires torch)."""
    from emg_label.hand3d import angles_to_landmarks, draw_hand

    lms = [angles_to_landmarks(ja_clip[i], side=side) for i in indices]
    pts = np.concatenate(lms, axis=0)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    mid = (mn + mx) / 2.0
    half = (mx - mn).max() / 2.0 * 1.05
    lo, hi = mid - half, mid + half
    n = len(indices)
    fig = plt.figure(figsize=(2.4 * n, 2.6))
    for i, (lm, title) in enumerate(zip(lms, titles)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        draw_hand(ax, lm)
        ax.set_title(title, fontsize=8)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.view_init(elev=25, azim=-60)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def _write_index(path: str, entries_by_source: dict, fs: int) -> None:
    parts = ["<!doctype html><meta charset=utf-8><title>Clip gallery</title>",
             "<style>",
             "body{font-family:-apple-system,Segoe UI,sans-serif;margin:16px;color:#222;background:#fafafa}",
             "h1{font-size:18px} h2{margin-top:24px;border-bottom:2px solid #ddd;padding-bottom:4px;font-size:14px}",
             "table{border-collapse:collapse;width:100%;background:#fff}",
             "th,td{border:1px solid #eee;padding:6px 10px;text-align:left;font-size:12px;vertical-align:top}",
             "th{background:#f0f0f0}",
             "img{max-width:1100px;height:auto;display:block;border:1px solid #ddd}",
             ".key{font-family:monospace;font-size:11px;color:#666;white-space:nowrap}",
             ".qc{font-size:11px;color:#444;line-height:1.5;white-space:nowrap}",
             ".warn{color:#c33;font-weight:600}",
             "</style>"]
    total = sum(len(v) for v in entries_by_source.values())
    parts.append(f"<h1>Clip gallery &mdash; {total} clips across "
                 f"{len(entries_by_source)} files (fs={fs} Hz)</h1>")
    parts.append("<p>Workflow: scan keyframes here; fill <code>gesture_label</code> "
                 "in <code>clips.csv</code>. Rows with "
                 "<span class=warn>matched_burst=-1</span> were missed by the EMG "
                 "detector &mdash; review these first.</p>")
    for source in sorted(entries_by_source):
        ents = entries_by_source[source]
        parts.append(f"<h2>{html.escape(source)} ({len(ents)} clips)</h2>")
        parts.append("<table><tr><th>clip</th><th>QC</th><th>keyframes (start &rarr; pre-motion &rarr; 1/3 &rarr; 2/3 &rarr; apex &rarr; end)</th></tr>")
        for e in sorted(ents, key=lambda x: int(x["row"]["clip_id"])):
            row = e["row"]
            matched = int(row["matched_emg_seg_idx"])
            matched_html = (f"<span class=warn>burst -1</span>" if matched < 0
                            else f"burst {matched}")
            qc = (f"dur {row['duration_s']:.2f}s &middot; "
                  f"motion {row['motion_duration_s']:.2f}s<br>"
                  f"emg_rms {row['emg_rms']:.1f} &middot; "
                  f"env_peak {row['envelope_peak']:.1f}<br>"
                  f"pose_range {row['pose_range']:.2f} &middot; "
                  f"max_spd {row['max_pose_speed']:.1f}<br>"
                  f"matched: {matched_html}")
            if e["png_rel"]:
                img = f"<img src='{html.escape(e['png_rel'])}'>"
            else:
                img = ("<em>no PNG &mdash; rerun with torch installed or use "
                       f"<code>python visualize_segment.py {html.escape(e['npz_rel'])}</code></em>")
            parts.append(f"<tr><td><div class=key>c{int(row['clip_id']):04d}</div></td>"
                         f"<td class=qc>{qc}</td><td>{img}</td></tr>")
        parts.append("</table>")
    with open(path, "w") as f:
        f.write("".join(parts))


def main():
    ap = argparse.ArgumentParser(
        description="Export each clip in clips.csv as npz + 6-keyframe PNG")
    ap.add_argument("input_dir", nargs="?", default=None,
                    help="folder of source .npz files (used when "
                         "clips.csv has no source_path column; otherwise "
                         "ignored in favour of the absolute paths in the CSV)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--no-png", action="store_true",
                    help="skip keyframe rendering (no torch needed)")
    ap.add_argument("--limit", type=int, default=None,
                    help="export at most N clips per source file (for sampling)")
    ap.add_argument("--force", action="store_true",
                    help="re-export clips even if their .npz (+ .png unless "
                         "--no-png) already exist under clips_export/.")
    args = ap.parse_args()

    clips_path = os.path.join(args.out, "clips.csv")
    if not os.path.isfile(clips_path):
        raise SystemExit(f"missing {clips_path}; run segment.py first")
    df = pd.read_csv(clips_path)
    if df.empty:
        raise SystemExit(f"{clips_path} has no rows")

    export_dir = os.path.join(args.out, "clips_export")
    os.makedirs(export_dir, exist_ok=True)

    n_npz = n_png = 0
    n_png_failed = n_npz_cached = n_png_cached = 0
    entries_by_source: dict[str, list] = {}
    png_disabled = args.no_png
    index_path = os.path.join(export_dir, "index.html")

    fk_disabled = png_disabled  # FK fallback gets disabled separately on torch failure
    for source_file, gdf in df.groupby("source_file"):
        # Resolve source path: prefer the absolute source_path column written
        # by segment.py; fall back to <input_dir>/<source_file> for older CSVs.
        row0 = gdf.iloc[0]
        if "source_path" in df.columns and str(row0.get("source_path", "")).strip():
            src_path = str(row0["source_path"])
        elif args.input_dir:
            src_path = os.path.join(args.input_dir, source_file)
        else:
            print(f"SKIP {source_file}: no source_path in CSV and no --input-dir")
            continue
        hand = str(row0.get("hand", "")).strip() or None
        info = io_utils.parse_file_info(src_path)
        side = hand or info.hand or "left"

        rows = gdf if args.limit is None else gdf.head(args.limit)

        # Skip-detection: from the clips.csv row derive {key}, then check
        # whether {key}.npz (+ {key}.png unless png-disabled) already exist.
        # Defer the heavy load_npz / load_skeleton until at least one row of
        # this source actually needs (re)processing -- a fully-cached source
        # then costs zero IO.
        def _row_done(row):
            key = _clip_key(info.stem, row["clip_id"])
            npz_done = os.path.isfile(os.path.join(export_dir, key + ".npz"))
            png_done = (png_disabled
                        or os.path.isfile(
                            os.path.join(export_dir, key + ".png")))
            return npz_done and png_done

        needs_work = (args.force
                      or any(not _row_done(r) for _, r in rows.iterrows()))

        emg = ja = skel = None
        if needs_work:
            emg, ja = io_utils.load_npz(src_path, hand=side)
            skel = (None if png_disabled
                    else io_utils.load_skeleton(src_path, hand=side))

        n_new = n_cached = 0
        for _, row in rows.iterrows():
            key = _clip_key(info.stem, row["clip_id"])
            npz_out = os.path.join(export_dir, key + ".npz")
            png_out = os.path.join(export_dir, key + ".png")
            npz_exists = os.path.isfile(npz_out)
            png_exists = os.path.isfile(png_out)

            cached = (not args.force) and npz_exists and (
                png_disabled or png_exists)
            if cached:
                n_cached += 1
                n_npz_cached += 1
                if png_exists and not png_disabled:
                    n_png_cached += 1
                entries_by_source.setdefault(source_file, []).append({
                    "key": key, "row": row,
                    "png_rel": (key + ".png") if (
                        not png_disabled and png_exists) else None,
                    "npz_rel": key + ".npz",
                })
                continue

            cs = int(row["clip_start_sample"])
            ce = int(row["clip_end_sample"])
            n_clip = ce - cs

            # --- Write the slice ------------------------------------------
            if args.force or not npz_exists:
                tmp = npz_out + "._tmp_.npz"
                np.savez(
                    tmp,
                    emg=emg[cs:ce], joint_angles=ja[cs:ce],
                    fs=int(args.fs),
                    clip_start=cs, clip_end=ce,
                    static_in_start=int(row["static_in_start_sample"]),
                    static_in_end=int(row["static_in_end_sample"]),
                    motion_start=int(row["motion_start_sample"]),
                    motion_end=int(row["motion_end_sample"]),
                    static_out_start=int(row["static_out_start_sample"]),
                    static_out_end=int(row["static_out_end_sample"]),
                    apex=int(row["apex_sample"]),
                    source_file=str(source_file),
                    group=str(row["group"]),
                    subject=str(row.get("subject", "")),
                    hand=str(row.get("hand", side)),
                    clip_id=int(row["clip_id"]),
                    matched_emg_seg_idx=int(row.get("matched_emg_seg_idx", -1)),
                )
                os.replace(tmp, npz_out)
                n_npz += 1

            # --- Render keyframes ----------------------------------------
            # Skeleton-based rendering (raw Manus XYZ) is preferred -- no
            # torch needed and matches reference's visualization. FK fallback
            # handles processed data where only joint_angles is present.
            png_made = png_exists  # keep cached PNG if we didn't redo it
            if not png_disabled and (args.force or not png_exists):
                rel = _keyframe_indices(row, n_clip)
                titles = [
                    f"clip_start (0.00s)",
                    f"pre-motion ({rel[1] / args.fs:.2f}s)",
                    f"motion 1/3 ({rel[2] / args.fs:.2f}s)",
                    f"motion 2/3 ({rel[3] / args.fs:.2f}s)",
                    f"apex ({rel[4] / args.fs:.2f}s)",
                    f"clip_end ({rel[5] / args.fs:.2f}s)",
                ]
                tmp_png = png_out + ".tmp.png"
                try:
                    if skel is not None:
                        _render_keyframes_skeleton(
                            tmp_png, skel[cs:ce], rel, titles)
                    elif not fk_disabled:
                        _render_keyframes_fk(
                            tmp_png, ja[cs:ce], rel, titles, side)
                    else:
                        raise RuntimeError("no renderer available")
                    os.replace(tmp_png, png_out)
                    n_png += 1
                    png_made = True
                except ModuleNotFoundError as ex:
                    if os.path.exists(tmp_png):
                        os.remove(tmp_png)
                    print(f"WARN: keyframe FK unavailable ({ex}); "
                          f"disabling FK fallback for remaining clips.")
                    fk_disabled = True
                except Exception as ex:
                    if os.path.exists(tmp_png):
                        os.remove(tmp_png)
                    n_png_failed += 1
                    print(f"WARN: keyframe render failed for {key}: {ex}")

            n_new += 1
            entries_by_source.setdefault(source_file, []).append({
                "key": key, "row": row,
                "png_rel": (key + ".png") if png_made else None,
                "npz_rel": key + ".npz",
            })
        if emg is not None:
            del emg, ja
        if n_cached and not n_new:
            print(f"{source_file}: {n_cached} cached (all)")
        elif n_cached:
            print(f"{source_file}: exported {n_new} clips, {n_cached} cached")
        else:
            print(f"{source_file}: exported {n_new} clips")

        # Stream index.html so the gallery is browsable mid-run.
        _write_index(index_path, entries_by_source, args.fs)

    _write_index(index_path, entries_by_source, args.fs)
    print()
    print(f"Wrote {n_npz} npz ({n_npz_cached} cached), "
          f"{n_png} PNGs ({n_png_cached} cached, {n_png_failed} failed) "
          f"to {export_dir}")
    print(f"Open {index_path} and fill `gesture_label` in {clips_path}")


if __name__ == "__main__":
    main()
