#!/usr/bin/env python
"""List / compare / prune the clustering runs in OUT/index.db.

One row per run: the DATA it used (scope) + the METHOD + (if evaluate.py has
run) the label-driven metrics. Prints the table and writes
``OUT/cluster_runs/INDEX.csv`` so you can eyeball/diff many experiments. This is
the "在结果里直观看出不同聚类用了哪些数据 / 哪种方法" view.

Deleting a run's directory by hand leaves its db rows behind (cluster_runs +
cluster_assignments), so the run keeps showing up. Reconcile with:

Usage:
  python runs.py --out out                  # list (+ writes INDEX.csv)
  python runs.py --out out prune            # drop db rows of runs whose dir is gone
  python runs.py --out out rm <run_id> ...  # delete specific runs (db + dir)
  (via pulse.sh: ./pulse.sh runs [prune|rm <run_id> ...])
"""
import argparse
import json
import os

import pandas as pd

from emg_label import select, store

COLS = ["run_id", "channel", "data", "n_clips", "n_rec", "n_subj", "sessions",
        "dates", "k", "method", "n_lab", "ARI", "NMI", "purity", "created_at"]


def _row(out_dir, r):
    p = json.loads(r["params_json"] or "{}")
    scope = p.get("scope") or {}
    res = scope.get("resolved", {})
    method = p.get("repr") or p.get("group_by") or ""      # the channel-specific knob
    if p.get("subject_norm") and p["subject_norm"] != "none":
        method = f"{method}+{p['subject_norm']}".strip("+")
    m = {}
    mp = os.path.join(out_dir, "cluster_runs", r["run_id"], "metrics.json")
    if os.path.isfile(mp):
        try:
            ld = json.load(open(mp)).get("label_driven", {})
            m = {"n_lab": ld.get("n_labeled"), "ARI": ld.get("ari"),
                 "NMI": ld.get("nmi"), "purity": ld.get("purity")}
        except Exception:
            pass
    dmin, dmax = res.get("date_min", ""), res.get("date_max", "")
    return {
        "run_id": r["run_id"], "channel": r["channel"],
        "data": select.scope_tag(scope) if scope else "all",
        "n_clips": res.get("n_clips"), "n_rec": res.get("n_recordings"),
        "n_subj": res.get("n_subjects"), "sessions": len(res.get("sessions", [])),
        "dates": dmin if dmin == dmax else f"{dmin}..{dmax}",
        "k": p.get("k") if p.get("k") is not None else p.get("n_clusters"),
        "method": method, "n_lab": m.get("n_lab"), "ARI": m.get("ARI"),
        "NMI": m.get("NMI"), "purity": m.get("purity"),
        "created_at": r["created_at"],
    }


def main():
    ap = argparse.ArgumentParser(description="list/compare/prune clustering runs")
    ap.add_argument("--out", default="out")
    ap.add_argument("action", nargs="?", default="list",
                    choices=["list", "prune", "rm"],
                    help="list (default) | prune (drop runs whose dir is gone) | "
                         "rm <run_id>... (delete specific runs: db rows + dir)")
    ap.add_argument("ids", nargs="*", help="run_id(s) for 'rm'")
    args = ap.parse_args()

    if args.action == "prune":
        pruned = store.prune_runs(args.out)
        print(f"pruned {len(pruned)} orphan run(s) (dir gone):"
              if pruned else "nothing to prune (db and disk are in sync)")
        for rid in pruned:
            print(f"  - {rid}")
        return
    if args.action == "rm":
        if not args.ids:
            raise SystemExit("rm needs at least one run_id (see ./pulse.sh runs)")
        for rid in args.ids:
            had = store.drop_run(args.out, rid, remove_dir=True)
            print(f"{'removed' if had else 'not in db (dir removed if any):'} {rid}")
        return

    conn = store.connect(args.out)
    runs = list(conn.execute(
        "SELECT run_id, created_at, channel, params_json FROM cluster_runs "
        "ORDER BY created_at"))
    conn.close()
    if not runs:
        print(f"no cluster runs in {store.db_path(args.out)}; "
              f"run ./pulse.sh cluster / cluster-traj first")
        return
    df = pd.DataFrame([_row(args.out, r) for r in runs])[COLS]
    rundir = os.path.join(args.out, "cluster_runs")
    os.makedirs(rundir, exist_ok=True)
    idx = os.path.join(rundir, "INDEX.csv")
    df.to_csv(idx, index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 40,
                           "display.max_colwidth", 60):
        print(df.to_string(index=False))
    print(f"\n{len(df)} run(s)  ->  {idx}")
    print("(ARI/NMI/purity blank = not evaluated yet; run ./pulse.sh eval RUN=<run_id>)")


if __name__ == "__main__":
    main()
