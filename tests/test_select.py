"""Tests for the clustering data selector (emg_label.select)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from emg_label import select


def _clips():
    """Three recordings across 2 subjects / 3 sessions / 3 dates, 2 clips each."""
    recs = [
        ("ghd-1108__20260501-left__20260501_101519.npz", "ghd-1108", "left"),
        ("ghd-1108__20260503-left-3__20260503_131215.npz", "ghd-1108", "left"),
        ("fgw-0917__20260502-right__20260502_104240.npz", "fgw-0917", "right"),
    ]
    rows = []
    for sf, subj, hand in recs:
        for cid in (0, 1):
            rows.append({"source_file": sf, "clip_id": cid, "subject": subj,
                         "hand": hand, "group": f"{subj}-{hand}"})
    return pd.DataFrame(rows)


def _ns(**kw):
    base = dict(subjects=None, only_hand=None, dates=None, date_from=None,
                date_to=None, sessions=None, recordings=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_no_filter_keeps_all():
    df, scope = select.select_clips(_clips(), _ns())
    assert len(df) == 6
    assert scope["filters"] == {}
    assert scope["resolved"]["n_clips"] == 6
    assert scope["resolved"]["n_recordings"] == 3
    assert scope["resolved"]["n_subjects"] == 2
    assert select.scope_tag(scope) == "all"


def test_subjects_normalized_match():
    # hyphenless form must still match the canonical 'ghd-1108'
    df, scope = select.select_clips(_clips(), _ns(subjects="ghd1108"))
    assert set(df["subject"]) == {"ghd-1108"}
    assert len(df) == 4
    assert scope["resolved"]["subjects"] == ["ghd1108"]
    assert select.scope_tag(scope) == "ghd1108"


def test_only_hand():
    df, _ = select.select_clips(_clips(), _ns(only_hand="right"))
    assert set(df["hand"]) == {"right"}
    assert len(df) == 2


def test_sessions_prefix_match():
    # exact session
    df, scope = select.select_clips(_clips(), _ns(sessions="20260501-left"))
    assert set(df["source_file"].str.split("__").str[1]) == {"20260501-left"}
    assert len(df) == 2
    assert select.scope_tag(scope) == "20260501-left"
    # prefix: '20260503-left' also catches '20260503-left-3'
    df2, _ = select.select_clips(_clips(), _ns(sessions="20260503-left"))
    assert set(df2["source_file"].str.split("__").str[1]) == {"20260503-left-3"}


def test_dates_prefix_and_range():
    df, _ = select.select_clips(_clips(), _ns(dates="20260501"))
    assert len(df) == 2
    # month prefix catches all of 202605
    df_all, _ = select.select_clips(_clips(), _ns(dates="202605"))
    assert len(df_all) == 6
    # range
    df_r, scope = select.select_clips(_clips(), _ns(date_from="20260502", date_to="20260503"))
    assert set(df_r["subject"]) == {"ghd-1108", "fgw-0917"}
    assert len(df_r) == 4
    assert scope["resolved"]["date_min"] == "20260502"
    assert scope["resolved"]["date_max"] == "20260503"


def test_recordings_exact_with_and_without_npz():
    df, _ = select.select_clips(
        _clips(), _ns(recordings="fgw-0917__20260502-right__20260502_104240"))  # no .npz
    assert set(df["subject"]) == {"fgw-0917"}
    assert len(df) == 2


def test_combined_filters_and_scope():
    df, scope = select.select_clips(
        _clips(), _ns(subjects="ghd-1108", sessions="20260501-left"))
    assert len(df) == 2
    assert scope["filters"]["subjects"] == ["ghd-1108"]
    assert scope["filters"]["sessions"] == ["20260501-left"]
    assert scope["resolved"]["n_recordings"] == 1
    assert select.scope_tag(scope) == "ghd1108_20260501-left"


def test_empty_selection():
    df, scope = select.select_clips(_clips(), _ns(subjects="nobody"))
    assert df.empty
    assert scope["resolved"]["n_clips"] == 0
