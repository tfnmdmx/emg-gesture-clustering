from __future__ import annotations

"""Stage 3 -- export hand-labelled clips, grouped by gesture.

Reads the per-clip gesture labels from ``OUT/index.db`` (annotations, kind
'label') joined onto clips (store.labeled_view, which already excludes invalid
recordings + invalid clips) and writes one npz per labelled clip under
``OUT/gestures/{label}/``. This is the label-driven export -- it replaces the
old "name each cluster, export per cluster" flow (per-cluster labels.csv is
gone; clusters now live in cluster_runs/ and are evaluated against labels, not
named for export).
"""

import argparse
import os

import numpy as np

from emg_label import io_utils, plotting, segmentation, store
from emg_label.config import Config


def main():
    ap = argparse.ArgumentParser(
        description="Stage 3: export hand-labelled clips grouped by gesture")
    ap.add_argument("--out", default="out")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--force", action="store_true",
                    help="re-export clips even if their .npz already exists "
                         "under <out>/gestures/<label>/.")
    args = ap.parse_args()

    cfg = Config(fs=args.fs, out_dir=args.out)
    conn = store.connect(cfg.out_dir)
    view = store.labeled_view(conn)        # clips + gesture_label, invalid excluded
    srcp = {r["source_file"]: r["source_path"]
            for r in conn.execute("SELECT source_file, source_path FROM recordings")}
    conn.close()
    if view.empty:
        print(f"no clips in {store.db_path(cfg.out_dir)}; run segment first")
        return
    view = view[view["gesture_label"].notna()
                & (view["gesture_label"].astype(str).str.strip() != "")].copy()
    if view.empty:
        print("0 labelled clips -- nothing to export. Label some clips first "
              "(./pulse.sh label).")
        return

    gestures_dir = os.path.join(cfg.out_dir, "gestures")
    lov_dir = os.path.join(cfg.out_dir, "labeled_overview")
    os.makedirs(lov_dir, exist_ok=True)
    n_exported = n_cached = 0

    # One recording at a time: load its npz once, export its labelled clips, draw
    # a labelled overview, release it (memory stays flat for large sets).
    for fname, fdf in view.groupby("source_file"):
        npz_path = io_utils.resolve_npz_path(fname, srcp.get(fname), None)
        info = io_utils.parse_file_info(npz_path)
        stem = str(fname)[:-4] if str(fname).endswith(".npz") else str(fname)
        emg = ja = None
        spans, labels = [], []
        for _, row in fdf.iterrows():
            label = str(row["gesture_label"]).strip()
            subj = str(row.get("subject") or info.subject or "NA")
            hand = str(row.get("hand") or info.hand or "NA")
            s, e = int(row["start_sample"]), int(row["end_sample"])
            cid = int(row["clip_id"])
            out_d = os.path.join(gestures_dir, label)
            out_path = os.path.join(
                out_d, f"{label}__{subj}-{hand}__{stem}__clip{cid:04d}.npz")
            spans.append((s, e)); labels.append(label)
            if os.path.isfile(out_path) and not args.force:
                n_cached += 1
                continue
            os.makedirs(out_d, exist_ok=True)
            if emg is None:
                emg, ja = io_utils.load_npz(npz_path, hand=info.hand)
            tmp = out_path + "._tmp_.npz"
            np.savez(
                tmp, emg=emg[s:e], joint_angles=ja[s:e],
                label=label, source_file=fname, clip_id=cid,
                start_sample=s, end_sample=e, fs=cfg.fs, hand=hand,
            )
            os.replace(tmp, out_path)
            n_exported += 1

        if spans:
            png = os.path.join(lov_dir, stem + ".png")
            if not (os.path.isfile(png) and not args.force):
                if emg is None:
                    emg, ja = io_utils.load_npz(npz_path, hand=info.hand)
                act = segmentation.emg_envelope(emg, cfg.fs, cfg.smooth_ms)
                tmp = png + ".tmp.png"
                plotting.plot_overview(emg, act, spans, cfg.fs, tmp, labels=labels)
                os.replace(tmp, png)
        if emg is not None:
            del emg, ja

    print(f"Exported {n_exported} labelled clips ({n_cached} cached) "
          f"to {gestures_dir}/<gesture>/")
    print(f"Labelled overviews in {lov_dir}")


if __name__ == "__main__":
    main()
