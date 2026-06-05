#!/usr/bin/env python
"""Pre-warm the labelling-UI caches so opening a recording is instant.

Each recording open costs two things the server otherwise computes lazily:
  1. Geometry + overview: pose-speed/EMG curves and the overview image (the
     latter re-rendered via matplotlib when OVERVIEW_REGEN=1) -- several seconds
     each, ~10s on a loaded box. Cached to ``<out>/cache/recording_cache`` and
     ``<out>/cache/overview_cache``. This is the "white screen at open" cost.
  2. 3D hand frames: emg2pose skins each frame on demand and caches the vertex
     JSON to ``<out>/cache/mesh_cache_v3/{stem}/f*.json`` (~12 KB/frame; frame
     count scales with POSE_FPS and duration) for jitter-free playback.

Both caches persist across restarts, so a recording is only slow the *first*
time it's opened. This script warms both ahead of time for every recording in
``<out>`` (or a subset), in parallel across recordings -- one process per
recording keeps the renderer single-threaded inside each worker (it's not
thread-safe) while still using all cores. It honours OVERVIEW_REGEN, so warm
with the SAME value you launch the server with (``OVERVIEW_REGEN=0`` reads the
fast baked shard overviews; ``=1`` regenerates them).

Usage:
    python prewarm_hands.py --out out_pose --workers 8
    WORKERS=8 ./pulse.sh label-prewarm                 # equivalent
    python prewarm_hands.py --out out_pose --limit 50  # just the first 50
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import label_server as L
from emg_label import io_utils

_STORE = None


def _init(out_dir):
    global _STORE
    _STORE = L.Store(out_dir)


def _do(stem):
    try:
        # 1) Warm the recording geometry + overview cache so the *first open*
        #    paints instantly. Without this the UI recomputes pose-speed/EMG
        #    curves and (when OVERVIEW_REGEN=1) re-renders the overview via
        #    matplotlib on every open -- several seconds each, ~10s on a loaded
        #    box -- which is the real "white screen, spinner" at open time.
        #    _recording_geometry honours OVERVIEW_REGEN: =1 regenerates+caches
        #    the overview (correct after plot/unit fixes), =0 reads the baked
        #    shard overview.png (fast), so warm whichever the server will serve.
        _STORE.recording(stem)
        # 2) Warm the per-frame 3D hand-mesh cache for jitter-free playback.
        _, _, nfr, _ = _STORE._mesh_prep(stem)
        new = 0
        for i in range(nfr):
            if not os.path.isfile(_STORE._hand_path(stem, i)):
                _STORE.hand_frame(stem, i)
                new += 1
        return (stem, nfr, new, None)
    except Exception as ex:                       # one bad recording != abort
        return (stem, 0, 0, f"{type(ex).__name__}: {ex}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out_pose")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N recordings (sampling)")
    ap.add_argument("--subjects", default=None,
                    help="comma-separated subject ids to keep, matched on the "
                         "NORMALIZED id (fgw-0917 == fgw0917) -- the same "
                         "selector as segment.py / pulse.sh SUBJECTS.")
    ap.add_argument("--subject", default=None,
                    help="(deprecated alias for a single --subjects id)")
    ap.add_argument("--first-per-folder", action="store_true",
                    help="only the FIRST recording of each folder (the left-tree "
                         "grouping in the UI) -- one representative per folder so "
                         "every folder opens instantly.")
    args = ap.parse_args()

    store = L.Store(args.out)
    stems = sorted(store.by_stem)         # deterministic order
    subjects = args.subjects or args.subject
    if subjects:
        want = {io_utils.normalize_subject(s)
                for s in subjects.split(",") if s.strip()}
        stems = [s for s in stems
                 if io_utils.normalize_subject(s.split("__")[0]) in want]
    if args.first_per_folder:
        first = {}                        # folder -> first (sorted) stem
        for s in stems:
            first.setdefault(store.by_stem[s]["folder"], s)
        stems = sorted(first.values())
        print(f"first-per-folder: {len(stems)} folders")
    if args.limit:
        stems = stems[:args.limit]
    if not stems:
        raise SystemExit("no recordings to prewarm")

    total_frames = est_mb = 0
    print(f"prewarm {len(stems)} recordings, workers={args.workers} -> "
          f"{store.hand_dir}/")
    done_recs = 0
    with Pool(args.workers, initializer=_init, initargs=(args.out,)) as pool:
        for stem, nfr, new, err in pool.imap_unordered(_do, stems):
            done_recs += 1
            if err:
                print(f"  [{done_recs}/{len(stems)}] SKIP {stem}: {err}")
                continue
            total_frames += nfr
            est_mb += nfr * 12 / 1024.0
            tag = "cached" if new == 0 else f"{new} new"
            print(f"  [{done_recs}/{len(stems)}] {stem}: {nfr} frames ({tag})")
    print(f"\ndone: {done_recs} recordings, ~{total_frames} frames, "
          f"~{est_mb:.0f} MB on disk under {store.hand_dir}/")


if __name__ == "__main__":
    main()
