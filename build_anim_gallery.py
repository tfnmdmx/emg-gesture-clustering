from __future__ import annotations

"""Batch-animate the first N segments of every label + build an index page.

For each label directory under ``<out_root>/segments/<label>/`` this picks the
first N segment npz files, renders each as an animated 3D-hand Plotly HTML via
animate_segment.animate, and writes them to
``<out_root>/<subdir>/<label>/<stem>.html``. Finally it writes an
``index.html`` under ``<out_root>/<subdir>/`` linking every animation, grouped
by group/side, so you can browse all classes from one page.

Usage:
    python build_anim_gallery.py [--out-root out] [--subdir hand_anim]
        [--n 3] [--max-frames 80] [--fps 30] [--clean]
"""

import argparse
import glob
import html
import os

import numpy as np

# animate_segment (plotly + emg2pose/torch FK) is imported lazily inside build()
# so a missing optional dep degrades to "index cached HTMLs" instead of crashing
# the whole pulse.sh qc/run/export step that chains into the gallery.


def _meta(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    fs = int(d["fs"]) if "fs" in d.files else 2000
    T = int(np.asarray(d["joint_angles"]).shape[0])
    return {
        "label": str(d["label"]) if "label" in d.files else "?",
        "group": str(d["group"]) if "group" in d.files else "?",
        "cluster_id": str(d["cluster_id"]) if "cluster_id" in d.files else "?",
        "seg_idx": str(d["seg_idx"]) if "seg_idx" in d.files else "?",
        "source_file": str(d["source_file"]) if "source_file" in d.files else "",
        "dur_s": T / fs,
        "T": T,
    }


def _label_sort_key(label):
    # Sort left-0, left-1, ... left-17 numerically (not lexicographically).
    parts = label.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (parts[0], int(parts[1]))
    return (label, -1)


def build(out_root, subdir, n, max_frames, fps, clean):
    seg_root = os.path.join(out_root, "segments")
    anim_root = os.path.join(out_root, subdir)
    if not os.path.isdir(seg_root):
        raise SystemExit(f"not found: {seg_root}")

    try:
        from animate_segment import animate
    except ImportError as exc:
        animate = None
        print(f"!! animate_segment unavailable ({exc}); skipping new renders, "
              f"indexing only already-cached HTMLs.")

    labels = sorted(
        [d for d in os.listdir(seg_root)
         if os.path.isdir(os.path.join(seg_root, d))],
        key=_label_sort_key)

    # entries[group] = list of {label, samples=[{stem, rel, meta}]}
    entries = {}
    total_html = n_cached = 0
    index_path = os.path.join(anim_root, "index.html")
    for label in labels:
        files = sorted(glob.glob(os.path.join(seg_root, label, "*.npz")))[:n]
        if not files:
            continue
        out_dir = os.path.join(anim_root, label)
        if clean and os.path.isdir(out_dir):
            for f in glob.glob(os.path.join(out_dir, "*.html")):
                os.remove(f)
        samples = []
        for npz in files:
            stem = os.path.splitext(os.path.basename(npz))[0]
            out_path = os.path.join(out_dir, stem + ".html")
            # Skip-detection: HTML already rendered -> reuse it. The animate
            # output is a deterministic function of the input npz, so an
            # existing file is the same artifact a fresh run would produce.
            if os.path.isfile(out_path) and not clean:
                n_cached += 1
            elif animate is not None:
                animate(npz, out_path, max_frames=max_frames, fps=fps)
                total_html += 1
            else:
                continue  # renderer unavailable and no cached HTML -> skip
            m = _meta(npz)
            samples.append({
                "stem": stem,
                "rel": os.path.relpath(out_path, anim_root),
                "meta": m,
            })
        if not samples:
            continue
        grp = samples[0]["meta"]["group"]
        entries.setdefault(grp, []).append({"label": label, "samples": samples})
        # Refresh index after each label so a long run is browsable mid-way.
        _write_index(index_path, entries, n)

    _write_index(index_path, entries, n)
    print(f"\nWrote {total_html} animation HTMLs ({n_cached} cached) "
          f"across {len(labels)} labels")
    print(f"Index: {index_path}")


def _write_index(index_path, entries, n):
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         margin:24px;color:#222;background:#fafafa}
    h1{font-size:20px} h2{font-size:16px;margin-top:28px;
       border-bottom:2px solid #ddd;padding-bottom:4px}
    table{border-collapse:collapse;width:100%;background:#fff}
    th,td{border:1px solid #e2e2e2;padding:6px 10px;text-align:left;font-size:13px}
    th{background:#f0f0f0}
    td.label{font-weight:600;white-space:nowrap}
    a{color:#1a6;text-decoration:none} a:hover{text-decoration:underline}
    .src{color:#888;font-size:11px}
    .muted{color:#999}
    """
    rows_html = []
    total_labels = sum(len(v) for v in entries.values())
    rows_html.append(f"<h1>Hand-pose animations &mdash; first {n} segment(s) "
                     f"per class &nbsp;<span class='muted'>"
                     f"({total_labels} classes)</span></h1>")
    for grp in sorted(entries):
        rows_html.append(f"<h2>{html.escape(grp)}</h2>")
        rows_html.append("<table><tr><th>label</th><th>cluster_id</th>"
                         "<th>samples (click to open)</th></tr>")
        for ent in sorted(entries[grp], key=lambda e: _label_sort_key(e["label"])):
            label = ent["label"]
            cid = ent["samples"][0]["meta"]["cluster_id"]
            links = []
            for i, s in enumerate(ent["samples"]):
                m = s["meta"]
                href = html.escape(s["rel"])
                txt = (f"#{i+1} &nbsp;{m['dur_s']:.2f}s"
                       f" <span class='src'>{html.escape(m['source_file'])}"
                       f" seg{m['seg_idx']}</span>")
                links.append(f"<div><a href='{href}'>{txt}</a></div>")
            rows_html.append(
                f"<tr><td class='label'>{html.escape(label)}</td>"
                f"<td>{html.escape(cid)}</td><td>{''.join(links)}</td></tr>")
        rows_html.append("</table>")

    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>Hand-pose animation gallery</title>"
           f"<style>{css}</style></head><body>{''.join(rows_html)}</body></html>")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w") as f:
        f.write(doc)


def main():
    ap = argparse.ArgumentParser(
        description="Batch-animate first N segments per label + index page")
    ap.add_argument("--out-root", default="out")
    ap.add_argument("--subdir", default="hand_anim")
    ap.add_argument("--n", type=int, default=3,
                    help="samples per label (default 3)")
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--clean", action="store_true",
                    help="remove existing *.html in each label dir first")
    args = ap.parse_args()
    build(args.out_root, args.subdir, args.n, args.max_frames, args.fps,
          args.clean)


if __name__ == "__main__":
    main()
