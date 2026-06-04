import os

import pandas as pd

from emg_label import store


def _write_shard(out, stem, source_file, n_clips=3, seg_version="v1",
                 clip_id_start=0, clip_span=1000):
    """Create OUT/shards/{stem}/ with recording.csv + bursts.csv + clips.csv in
    the target schema. Clips tile [clip_id_start*clip_span, ...)."""
    d = os.path.join(out, "shards", stem)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame([{
        "source_file": source_file, "source_path": f"/data/{stem}.npz",
        "group": "g-left", "subject": "g", "hand": "left",
        "n_samples": 100000, "duration_s": 50.0, "n_bursts": n_clips,
        "n_clips": n_clips, "n_burst_only": 0, "n_clip_only": 0,
        "emg_pose_lag_s": 0.1, "emg_pose_corr": 0.3, "lag_flag": "ok",
        "pose_nan_frac": 0.0, "emg_nan_frac": 0.0, "enter_thresh": 1.0,
        "exit_thresh": 0.5, "pose_thresh": 1.0, "pose_exit_thresh": 0.5,
        "rec_pose_range": 300.0, "pose_static": 0, "seg_version": seg_version,
    }]).to_csv(os.path.join(d, "recording.csv"), index=False)
    pd.DataFrame([{
        "source_file": source_file, "burst_idx": i, "group": "g-left",
        "start_sample": i * clip_span, "end_sample": i * clip_span + 400,
        "hold_end_sample": i * clip_span + 800, "apex_sample": i * clip_span + 600,
        "duration_s": 0.4, "emg_rms": 1.0, "envelope_peak": 5.0,
        "pose_range": 1.0, "matched_clip_id": i,
    } for i in range(n_clips)]).to_csv(os.path.join(d, "bursts.csv"), index=False)
    pd.DataFrame([{
        "source_file": source_file, "clip_id": clip_id_start + i, "group": "g-left",
        "subject": "g", "hand": "left",
        "start_sample": i * clip_span, "end_sample": i * clip_span + 900,
        "motion_start_sample": i * clip_span, "motion_end_sample": i * clip_span + 400,
        "hold_start_sample": i * clip_span + 400, "hold_end_sample": i * clip_span + 900,
        "apex_sample": i * clip_span + 600, "duration_s": 0.45,
        "motion_duration_s": 0.2, "hold_duration_s": 0.25, "emg_rms": 1.0,
        "envelope_peak": 5.0, "mean_pose_speed": 10.0, "max_pose_speed": 30.0,
        "pose_range": 1.0, "matched_burst_idx": i, "fusion_type": "both",
        "review_flag": "", "seg_version": seg_version,
    } for i in range(n_clips)]).to_csv(os.path.join(d, "clips.csv"), index=False)


def test_build_index_counts(tmp_path):
    out = str(tmp_path)
    _write_shard(out, "recA", "recA.npz", n_clips=3)
    _write_shard(out, "recB", "recB.npz", n_clips=2)
    counts = store.build_index(out)
    assert counts == {"recordings": 2, "bursts": 5, "clips": 5}
    conn = store.connect(out)
    assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] == 2
    conn.close()


def test_build_index_idempotent_no_stale(tmp_path):
    out = str(tmp_path)
    _write_shard(out, "recA", "recA.npz", n_clips=5)
    store.build_index(out)
    # re-segment with fewer clips -> rebuild must not leave stale rows
    _write_shard(out, "recA", "recA.npz", n_clips=2)
    store.build_index(out)
    conn = store.connect(out)
    assert conn.execute(
        "SELECT COUNT(*) FROM clips WHERE source_file='recA.npz'").fetchone()[0] == 2
    conn.close()


def test_export_csv(tmp_path):
    out = str(tmp_path)
    _write_shard(out, "recA", "recA.npz", n_clips=3)
    store.build_index(out)
    store.export_csv(out)
    df = pd.read_csv(os.path.join(out, "clips.csv"))
    assert len(df) == 3
    assert "matched_burst_idx" in df.columns and "gesture_label" not in df.columns
    assert os.path.isfile(os.path.join(out, "bursts.csv"))


def test_annotations_and_labeled_view(tmp_path):
    out = str(tmp_path)
    _write_shard(out, "recA", "recA.npz", n_clips=3)
    store.build_index(out)
    conn = store.connect(out)
    store.set_annotation(conn, "recA.npz", 0, "clip", "label", value="fist",
                         clip_start_sample=0, clip_end_sample=900, seg_version="v1")
    store.set_annotation(conn, "recA.npz", 1, "clip", "invalid",
                         clip_start_sample=1000, clip_end_sample=1900, seg_version="v1")
    view = store.labeled_view(conn)
    # clip 1 (invalid) excluded; clip 0 carries its label
    assert set(view["clip_id"]) == {0, 2}
    assert view[view["clip_id"] == 0]["gesture_label"].iloc[0] == "fist"
    conn.close()


def test_excluded_recordings(tmp_path):
    out = str(tmp_path)
    _write_shard(out, "recA", "recA.npz")
    _write_shard(out, "recB", "recB.npz")
    store.build_index(out)
    conn = store.connect(out)
    store.set_annotation(conn, "recA.npz", -1, "recording", "invalid")
    assert store.excluded_recordings(conn) == {"recA.npz"}
    conn.close()
    store.drop_recording(out, "recB.npz", note="manual")
    conn = store.connect(out)
    assert store.excluded_recordings(conn) == {"recA.npz", "recB.npz"}
    conn.close()


def test_drop_recording_not_resurrected_by_rebuild(tmp_path):
    out = str(tmp_path)
    _write_shard(out, "recA", "recA.npz", n_clips=3)
    store.build_index(out)
    store.drop_recording(out, "recA.npz")
    # shard still on disk; a rebuild must skip the tombstoned recording
    counts = store.build_index(out)
    assert counts["recordings"] == 0
    conn = store.connect(out)
    assert conn.execute(
        "SELECT COUNT(*) FROM clips WHERE source_file='recA.npz'").fetchone()[0] == 0
    conn.close()


def test_cluster_run_and_clusters_with_labels(tmp_path):
    out = str(tmp_path)
    _write_shard(out, "recA", "recA.npz", n_clips=3)
    store.build_index(out)
    conn = store.connect(out)
    store.set_annotation(conn, "recA.npz", 0, "clip", "label", value="fist",
                         seg_version="v1")
    conn.close()
    store.write_cluster_run(
        out, "20260604-120000__abcd1234", "apex", {"k": 2, "unit": "clip"},
        [("recA.npz", 0, 0), ("recA.npz", 1, 1), ("recA.npz", 2, 0)],
        created_at="20260604-120000")
    conn = store.connect(out)
    assert store.latest_run(conn, "apex") == "20260604-120000__abcd1234"
    cw = store.clusters_with_labels(conn, "20260604-120000__abcd1234")
    assert len(cw) == 3
    row0 = cw[cw["clip_id"] == 0].iloc[0]
    assert row0["cluster_id"] == 0 and row0["gesture_label"] == "fist"
    conn.close()


def test_remap_annotations(tmp_path):
    out = str(tmp_path)
    _write_shard(out, "recA", "recA.npz", n_clips=3, seg_version="v1")
    store.build_index(out)
    conn = store.connect(out)
    # label clip 0 at its v1 interval [0, 900)
    store.set_annotation(conn, "recA.npz", 0, "clip", "label", value="fist",
                         clip_start_sample=0, clip_end_sample=900, seg_version="v1")
    conn.close()
    # re-segment: same intervals but clip ids shifted by +10 (clip 0 -> 10)
    _write_shard(out, "recA", "recA.npz", n_clips=3, seg_version="v2",
                 clip_id_start=10)
    store.build_index(out)
    conn = store.connect(out)
    res = store.remap_annotations(conn, "v2")
    assert res["remapped"] == 1
    moved = [dict(r) for r in conn.execute(
        "SELECT * FROM annotations WHERE kind='label'")]
    assert len(moved) == 1 and moved[0]["clip_id"] == 10
    assert moved[0]["seg_version"] == "v2"
    conn.close()
