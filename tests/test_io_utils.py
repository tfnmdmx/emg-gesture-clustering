import warnings

from emg_label.io_utils import parse_file_info, group_files


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
    p = tmp_path / "bad.npz"
    np.savez(p, emg=np.zeros((100, 16), np.float32), joint_angles=np.zeros((90, 20), np.float32))
    with pytest.raises(ValueError):
        load_npz(str(p))
