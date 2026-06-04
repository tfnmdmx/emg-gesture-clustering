import os

import warnings

from emg_label.io_utils import (group_files, parse_file_info,
                                resolve_npz_path)


def test_resolve_npz_path_prefers_existing_source_path(tmp_path):
    real = tmp_path / "rec.npz"
    real.write_bytes(b"x")
    # an existing source_path wins over the input_dir fallback
    assert resolve_npz_path("rec.npz", str(real), "/pool") == str(real)


def test_resolve_npz_path_falls_back_to_input_dir():
    # missing/NaN/None source_path -> input_dir/source_file (legacy pool)
    assert resolve_npz_path("a.npz", None, "/pool") == os.path.join("/pool", "a.npz")
    assert resolve_npz_path("a.npz", float("nan"), "/pool") == os.path.join("/pool", "a.npz")
    assert resolve_npz_path("a.npz", "/gone/x.npz", "/pool") == os.path.join("/pool", "a.npz")


def test_resolve_npz_path_bare_filename_when_nothing_given():
    assert resolve_npz_path("a.npz", None, None) == "a.npz"


def test_normalize_subject_strips_hyphen_and_case():
    from emg_label.io_utils import normalize_subject
    assert normalize_subject("fgw-0917") == "fgw0917"
    assert normalize_subject("fgw0917") == "fgw0917"
    assert normalize_subject("FGW-0917") == "fgw0917"
    assert normalize_subject(None) == ""
    # canonical and batch-dir forms collapse to the same key
    assert normalize_subject("zyb-0201") == normalize_subject("zyb0201")


def test_parse_processed_batch_dir_timestamp_file():
    # processed_data/<subj><date>_<hand>/<timestamp>.npz -- subject in dir name,
    # no hyphen, bare-timestamp filename.
    info = parse_file_info(
        "/data/processed_data/fgw0917_0502_left/20260502_115218.npz")
    assert info.subject == "fgw-0917"      # re-hyphenated to canonical
    assert info.hand == "left"
    assert info.group == "fgw-0917-left"
    assert info.parsed is True
    assert info.stem == "fgw-0917__fgw0917_0502_left__20260502_115218"  # unique


def test_parse_processed_batch_dir_no_hand():
    info = parse_file_info(
        "/data/processed_data/zyb0201_20260514/20260514_175435.npz")
    assert info.subject == "zyb-0201"
    assert info.hand is None
    assert info.group == "zyb-0201"
    assert info.parsed is True


def test_parse_standard_name():
    info = parse_file_info("/data/fgw-0917__20260502-left__20260502_115218.npz")
    assert info.subject == "fgw-0917"
    assert info.hand == "left"
    assert info.group == "fgw-0917-left"
    assert info.stem == "fgw-0917__20260502-left__20260502_115218"
    assert info.parsed is True


def test_parse_right_hand():
    info = parse_file_info("abc__20260101-right__t.npz")
    assert info.hand == "right"
    assert info.group == "abc-right"


def test_parse_run_suffix_pools_by_hand():
    # real data has a run number after the hand: date-hand-run
    info = parse_file_info("/d/fgw-0917__20260502-left-3__20260502_104240.npz")
    assert info.subject == "fgw-0917"
    assert info.hand == "left"
    assert info.group == "fgw-0917-left"
    assert info.parsed is True
    # different runs of the same subject+hand share one group
    other = parse_file_info("fgw-0917__20260502-left-7__20260502_140000.npz")
    assert other.group == info.group


def test_group_files_pools_runs_of_same_hand():
    paths = [
        "fgw-0917__20260502-left__t1.npz",
        "fgw-0917__20260502-left-3__t2.npz",
        "fgw-0917__20260502-left-9__t3.npz",
    ]
    groups = group_files(paths)
    assert set(groups.keys()) == {"fgw-0917-left"}
    assert len(groups["fgw-0917-left"]) == 3


def test_parse_unparseable_name_becomes_own_group():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = parse_file_info("/data/weird_file.npz")
    assert info.parsed is False
    assert info.subject is None
    assert info.hand is None
    assert info.group == "weird_file"


def test_group_files_buckets_by_subject_hand():
    paths = [
        "s1__d-left__t1.npz",
        "s1__d-left__t2.npz",
        "s1__d-right__t1.npz",
        "s2__d-left__t1.npz",
    ]
    groups = group_files(paths)
    assert set(groups.keys()) == {"s1-left", "s1-right", "s2-left"}
    assert len(groups["s1-left"]) == 2


import numpy as np
import pytest
from emg_label.io_utils import load_npz


def test_load_npz_roundtrip(tmp_path):
    p = tmp_path / "x.npz"
    emg = np.zeros((100, 16), dtype=np.float32)
    ja = np.ones((100, 20), dtype=np.float32)
    np.savez(p, emg=emg, joint_angles=ja)
    e, j = load_npz(str(p))
    assert e.shape == (100, 16)
    assert j.shape == (100, 20)


def test_load_npz_length_mismatch(tmp_path):
    # Processed-format contract: explicit `joint_angles` must match `emg`
    # length. Length mismatch with no `manus_*_ergonomics` fallback must raise.
    p = tmp_path / "bad.npz"
    np.savez(p, emg=np.zeros((100, 16), np.float32), joint_angles=np.zeros((90, 20), np.float32))
    with pytest.raises(ValueError):
        load_npz(str(p))


def test_load_npz_raw_format_aligns_pose_to_emg(tmp_path):
    # Reference-format raw: `manus_<hand>_ergonomics` at a different length
    # than `emg`. load_npz must reconstruct time axes and interpolate the pose
    # onto the EMG sample axis so downstream gets equal-length arrays.
    fs = 1000
    duration = 2.0
    emg_t = np.linspace(0, duration, int(fs * duration))                  # 2000
    pose_t = np.linspace(0, duration, int(fs * duration * 1.05))          # 2100
    pose = np.stack(
        [np.sin(2 * np.pi * pose_t),
         np.cos(2 * np.pi * pose_t)],
        axis=1).astype(np.float32)
    emg = np.random.default_rng(0).normal(0, 1, size=(len(emg_t), 16)).astype(np.float32)
    p = tmp_path / "raw.npz"
    np.savez(
        p,
        emg=emg,
        manus_left_ergonomics=pose,
        manus_left_timestamps_raw=pose_t,
        record_t0=np.float64(0.0),
        record_t1=np.float64(duration),
    )
    e, ja = load_npz(str(p), hand="left")
    assert e.shape == (2000, 16)
    assert ja.shape == (2000, 2)
    # Interpolation reconstructs the same sinusoid on the EMG axis.
    expected = np.stack([np.sin(2 * np.pi * emg_t),
                         np.cos(2 * np.pi * emg_t)], axis=1)
    assert np.allclose(ja, expected, atol=0.01)


def test_load_npz_raw_infers_hand_from_parent_dir(tmp_path):
    # Path like .../{subject}/{date}-{hand}/{stamp}.npz -- hand is parsed from
    # the parent dir name, so callers don't have to thread it in for files
    # whose own name is just a timestamp.
    sess = tmp_path / "jm-0503" / "20260423-left"
    sess.mkdir(parents=True)
    p = sess / "20260423_132054.npz"
    emg = np.zeros((100, 16), np.float32)
    pose = np.zeros((105, 20), np.float32)
    np.savez(p, emg=emg, manus_left_ergonomics=pose,
             record_t0=np.float64(0.0), record_t1=np.float64(0.05))
    e, ja = load_npz(str(p))  # no hand kwarg
    assert e.shape == (100, 16) and ja.shape == (100, 20)


def test_parse_file_info_from_parent_dir():
    info = parse_file_info("/data/jm-0503/20260423-left-2/20260423_132054.npz")
    assert info.subject == "jm-0503"
    assert info.hand == "left"
    assert info.group == "jm-0503-left"
    assert info.parsed is True
    assert info.stem == "jm-0503__20260423-left-2__20260423_132054"


def test_load_skeleton_returns_none_for_processed_format(tmp_path):
    # Processed npz has no manus_*_skeleton -> caller falls back to FK path.
    from emg_label.io_utils import load_skeleton
    p = tmp_path / "processed.npz"
    np.savez(p,
             emg=np.zeros((100, 16), np.float32),
             joint_angles=np.zeros((100, 20), np.float32))
    assert load_skeleton(str(p)) is None


def test_load_skeleton_aligns_to_emg_axis(tmp_path):
    # Raw format: manus_*_skeleton at different length than emg. The loader
    # interpolates onto the EMG axis so callers can index skel[cs:ce] in the
    # same coords as emg[cs:ce] and joint_angles[cs:ce].
    from emg_label.io_utils import load_skeleton
    fs = 1000
    duration = 1.0
    emg_t = np.linspace(0, duration, fs)            # 1000 samples
    skel_t = np.linspace(0, duration, int(fs * 1.05))  # 1050 samples
    # Joint 0 walks linearly along x; other joints constant. After alignment,
    # joint 0's x at t=0.5s should be ~0.5.
    skel = np.zeros((len(skel_t), 25, 7), dtype=np.float32)
    skel[:, 0, 0] = skel_t.astype(np.float32)
    p = tmp_path / "raw.npz"
    np.savez(p,
             emg=np.zeros((fs, 16), np.float32),
             manus_left_skeleton=skel,
             record_t0=np.float64(0.0),
             record_t1=np.float64(duration))
    out = load_skeleton(str(p), hand="left")
    assert out is not None
    assert out.shape == (fs, 25, 3)
    # Interpolation reconstructs the linear ramp on the EMG axis.
    assert abs(out[fs // 2, 0, 0] - 0.5) < 0.01


def test_parse_meta_csv_expands_both_side(tmp_path):
    import csv as _csv
    from emg_label.io_utils import parse_meta_csv
    p = tmp_path / "meta.csv"
    with open(p, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=[
            "sample_path", "subject_id", "side", "sample_id"])
        w.writeheader()
        w.writerow({"sample_path": "/d/a.npz", "subject_id": "s1",
                    "side": "left", "sample_id": "s1__sess__a"})
        w.writerow({"sample_path": "/d/b.npz", "subject_id": "s2",
                    "side": "both", "sample_id": "s2__sess__b"})
    rows = parse_meta_csv(str(p))
    # "both" expands to two virtual rows (left + right).
    assert len(rows) == 3
    assert {r.group for r in rows} == {"s1-left", "s2-left", "s2-right"}
    # Same source path is referenced by both expanded "both" rows.
    both_paths = [r.path for r in rows if r.subject == "s2"]
    assert both_paths == ["/d/b.npz", "/d/b.npz"]
