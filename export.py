from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from emg_label import io_utils, plotting, segmentation
from emg_label.config import Config


def _clean_label(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    return s


def main():
    ap = argparse.ArgumentParser(description="Stage 3: export labeled segments")
    ap.add_argument("input_dir", nargs="?", default=None,
                    help="OPTIONAL legacy fallback dir of .npz. Omit it: npz are "
                         "located via each row's source_path (no work_pool needed).")
    ap.add_argument("--out", default="out")
    ap.add_argument("--labels", default=None,
                    help="labels.csv path (default <out>/labels.csv)")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--force", action="store_true",
                    help="re-export segments even if their .npz already "
                         "exists under <out>/segments/<label>/.")
    args = ap.parse_args()

    cfg = Config(fs=args.fs, out_dir=args.out)
    seg_df = pd.read_csv(os.path.join(cfg.out_dir, "segments_clustered.csv"))
    invalid = io_utils.load_invalid_recordings(cfg.out_dir)
    if invalid:
        n0 = len(seg_df)
        seg_df = seg_df[~seg_df["source_file"].astype(str).isin(invalid)] \
            .reset_index(drop=True)
        print(f"excluded {len(invalid)} invalid recording(s): "
              f"{n0 - len(seg_df)} segments dropped")
    labels_path = args.labels or os.path.join(cfg.out_dir, "labels.csv")
    lab_df = pd.read_csv(labels_path)
    lab_map = {(r["group"], int(r["cluster_id"])): _clean_label(r["label"])
               for _, r in lab_df.iterrows()}

    seg_dir = os.path.join(cfg.out_dir, "segments")
    lov_dir = os.path.join(cfg.out_dir, "labeled_overview")
    os.makedirs(lov_dir, exist_ok=True)

    n_exported = n_cached = 0
    # Process one source file at a time: export its labeled segments and draw
    # its labeled overview, then release it -- keeps memory flat for large sets.
    for fname, fdf in seg_df.groupby("source_file"):
        sp = fdf["source_path"].iloc[0] if "source_path" in fdf.columns else None
        npz_path = io_utils.resolve_npz_path(fname, sp, args.input_dir)
        info = io_utils.parse_file_info(npz_path)
        subj = info.subject or "NA"
        hand = info.hand or "NA"
        spans, labels = [], []
        emg = ja = None
        for _, row in fdf.iterrows():
            group = row["group"]
            cid = int(row["cluster_id"])
            label = lab_map.get((group, cid), "")
            if not label:
                continue
            s, e = int(row["start_sample"]), int(row["end_sample"])
            out_d = os.path.join(seg_dir, label)
            out_name = (f"{label}__{subj}-{hand}__{info.stem}"
                        f"__seg{int(row['seg_idx'])}.npz")
            out_path = os.path.join(out_d, out_name)
            # Skip-detection: derived deterministically from (label, subject,
            # hand, stem, seg_idx); existing npz = a prior run's identical
            # output. spans/labels are still appended so the labeled_overview
            # PNG reflects the full set, not just the freshly-written entries.
            already = os.path.isfile(out_path)
            if already and not args.force:
                n_cached += 1
                spans.append((s, e))
                labels.append(label)
                continue
            os.makedirs(out_d, exist_ok=True)
            if emg is None:
                emg, ja = io_utils.load_npz(npz_path)
            tmp = out_path + "._tmp_.npz"
            np.savez(
                tmp,
                emg=emg[s:e], joint_angles=ja[s:e],
                label=label, source_file=fname, group=group,
                seg_idx=int(row["seg_idx"]), start_sample=s, end_sample=e,
                fs=cfg.fs, cluster_id=cid,
            )
            os.replace(tmp, out_path)
            spans.append((s, e))
            labels.append(label)
            n_exported += 1

        if spans:
            png = os.path.join(lov_dir, info.stem + ".png")
            if os.path.isfile(png) and not args.force:
                pass
            else:
                if emg is None:
                    emg, ja = io_utils.load_npz(npz_path)
                act = segmentation.emg_envelope(emg, cfg.fs, cfg.smooth_ms)
                tmp = png + ".tmp.png"
                plotting.plot_overview(emg, act, spans, cfg.fs, tmp,
                                       labels=labels)
                os.replace(tmp, png)
        if emg is not None:
            del emg, ja

    if n_exported == 0 and n_cached == 0:
        print("WARNING: 0 segments exported. Check that labels.csv has the "
              "'label' column filled in and its group/cluster_id values match "
              "segments_clustered.csv.")
    print(f"Exported {n_exported} labeled segments "
          f"({n_cached} cached) to {seg_dir}")
    print(f"Labeled overviews in {lov_dir}")


if __name__ == "__main__":
    main()
