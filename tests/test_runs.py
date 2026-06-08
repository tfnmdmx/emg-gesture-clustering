"""Tests for run deletion / db<->disk reconciliation (store.drop_run/prune_runs)."""
import os
import shutil

import pandas as pd

from emg_label import store


def _mk_run(out, rid):
    df = pd.DataFrame({"source_file": ["a.npz", "a.npz"],
                       "clip_id": [0, 1], "cluster_id": [0, 1]})
    store.save_run(out, rid, "apex", {"channel": "apex", "k": 2}, df,
                   "20260608-000000")


def test_drop_run_removes_db_rows_and_dir(tmp_path):
    out = str(tmp_path)
    _mk_run(out, "r1")
    _mk_run(out, "r2")
    assert os.path.isdir(os.path.join(out, "cluster_runs", "r1"))

    assert store.drop_run(out, "r1") is True
    conn = store.connect(out)
    assert {r[0] for r in conn.execute("SELECT run_id FROM cluster_runs")} == {"r2"}
    assert conn.execute(
        "SELECT COUNT(*) FROM cluster_assignments WHERE run_id='r1'").fetchone()[0] == 0
    conn.close()
    assert not os.path.isdir(os.path.join(out, "cluster_runs", "r1"))
    # dropping an unknown run is a no-op returning False
    assert store.drop_run(out, "nope") is False


def test_prune_drops_runs_whose_dir_is_gone(tmp_path):
    out = str(tmp_path)
    _mk_run(out, "keep")
    _mk_run(out, "gone")
    shutil.rmtree(os.path.join(out, "cluster_runs", "gone"))   # hand-deleted dir

    pruned = store.prune_runs(out)
    assert pruned == ["gone"]
    conn = store.connect(out)
    assert {r[0] for r in conn.execute("SELECT run_id FROM cluster_runs")} == {"keep"}
    assert conn.execute(
        "SELECT COUNT(*) FROM cluster_assignments WHERE run_id='gone'").fetchone()[0] == 0
    conn.close()
    assert store.prune_runs(out) == []                          # idempotent
