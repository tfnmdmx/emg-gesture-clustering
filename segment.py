from __future__ import annotations

import argparse
import csv
import glob
import os

from emg_label import io_utils, plotting, segmentation
from emg_label.config import Config


def main():
    ap = argparse.ArgumentParser(description="Stage 1: segment action vs rest")
    ap.add_argument("input_dir", help="folder containing .npz files")
    ap.add_argument("--out", default="out")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--min-action-s", type=float, default=0.4)
    ap.add_argument("--min-rest-gap-s", type=float, default=0.2)
    ap.add_argument("--smooth-ms", type=float, default=150.0)
    ap.add_argument("--enter-thresh", type=float, default=None,
                    help="manual enter threshold (default: auto via Otsu)")
    ap.add_argument("--exit-thresh", type=float, default=None,
                    help="manual exit threshold (default: auto via Otsu)")
    args = ap.parse_args()

    cfg = Config(
        fs=args.fs, out_dir=args.out, min_action_s=args.min_action_s,
        min_rest_gap_s=args.min_rest_gap_s, smooth_ms=args.smooth_ms,
        enter_thresh=args.enter_thresh, exit_thresh=args.exit_thresh,
    )
    os.makedirs(os.path.join(cfg.out_dir, "overview"), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    if not paths:
        print(f"No .npz files found in {args.input_dir}")
        return

    rows = []
    for path in paths:
        info = io_utils.parse_file_info(path)
        emg, _ja = io_utils.load_npz(path)
        segs, act, enter, exit_thr = segmentation.segment_emg(emg, cfg)
        for seg_idx, (s, e) in enumerate(segs):
            rows.append({
                "source_file": os.path.basename(path),
                "group": info.group,
                "seg_idx": seg_idx,
                "start_sample": s,
                "end_sample": e,
                "duration_s": round((e - s) / cfg.fs, 4),
            })
        png = os.path.join(cfg.out_dir, "overview", info.stem + ".png")
        plotting.plot_overview(emg, act, segs, cfg.fs, png, enter, exit_thr)
        print(f"{os.path.basename(path)}: {len(segs)} segments")

    csv_path = os.path.join(cfg.out_dir, "segments.csv")
    fields = ["source_file", "group", "seg_idx", "start_sample",
              "end_sample", "duration_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} segments)")


if __name__ == "__main__":
    main()
