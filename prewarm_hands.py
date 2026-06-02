#!/usr/bin/env python
"""Pre-render hand-frame PNGs for the labelling UI so first open is instant.

The label server renders each recording's hand frames on demand and caches them
to ``<out>/hand_frames/{stem}/f*.png`` (~13 KB/frame, ~8 MB and ~29 s per
recording). That cache already persists across restarts, so a recording is only
slow the *first* time it's opened. This script warms that cache ahead of time
for every recording in ``<out>/clips.csv`` (or a subset), in parallel across
recordings -- one process per recording keeps matplotlib single-threaded inside
each worker (it's not thread-safe) while still using all cores.

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

_STORE = None


def _init(out_dir):
    global _STORE
    _STORE = L.Store(out_dir)


def _do(stem):
    try:
        _, _, nfr, _, _ = _STORE._hand_prep(stem)
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
    ap.add_argument("--subject", default=None,
                    help="only recordings whose stem contains this string")
    args = ap.parse_args()

    store = L.Store(args.out)
    stems = list(store.by_stem)
    if args.subject:
        stems = [s for s in stems if args.subject in s]
    if args.limit:
        stems = stems[:args.limit]
    if not stems:
        raise SystemExit("no recordings to prewarm")

    total_frames = est_mb = 0
    print(f"prewarm {len(stems)} recordings, workers={args.workers} -> "
          f"{args.out}/hand_frames/")
    done_recs = 0
    with Pool(args.workers, initializer=_init, initargs=(args.out,)) as pool:
        for stem, nfr, new, err in pool.imap_unordered(_do, stems):
            done_recs += 1
            if err:
                print(f"  [{done_recs}/{len(stems)}] SKIP {stem}: {err}")
                continue
            total_frames += nfr
            est_mb += nfr * 13 / 1024.0
            tag = "cached" if new == 0 else f"{new} new"
            print(f"  [{done_recs}/{len(stems)}] {stem}: {nfr} frames ({tag})")
    print(f"\ndone: {done_recs} recordings, ~{total_frames} frames, "
          f"~{est_mb:.0f} MB on disk under {args.out}/hand_frames/")


if __name__ == "__main__":
    main()
