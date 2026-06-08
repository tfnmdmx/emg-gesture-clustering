from __future__ import annotations

"""Clip-subset selector shared by the clustering channels (cluster.py /
cluster_traj.py).

segment + labelling run on the WHOLE library (one shared OUT/index.db); a
clustering run then picks a SUBSET of that shared, labelled db -- so you can run
many experiments over different data (per subject / session / date) without
re-segmenting or re-labelling. The selector vocabulary mirrors segment.py
(--subjects / --only-hand / --dates / --date-from / --date-to) and adds
--sessions and --recordings.

``select_clips`` also returns a ``scope`` dict (the filters asked for + the
resolved stats: how many clips/recordings/subjects, which sessions, date range)
so each run self-documents what data it used; ``scope_tag`` turns that into a
short readable string for the run-id / directory name.
"""

import re

import pandas as pd

from . import io_utils


def add_select_args(ap):
    """Add the data-selection flags to an argparse parser (both channels)."""
    g = ap.add_argument_group("data selection (a subset of the shared db)")
    g.add_argument("--subjects", default=None,
                   help="comma-sep subject ids; normalized match (fgw-0917==fgw0917)")
    g.add_argument("--only-hand", choices=["left", "right"], default=None,
                   help="keep only this hand")
    g.add_argument("--dates", default=None,
                   help="comma-sep date prefixes (YYYYMMDD day / YYYYMM month / YYYY)")
    g.add_argument("--date-from", default=None, help="keep recordings on/after YYYYMMDD")
    g.add_argument("--date-to", default=None, help="keep recordings on/before YYYYMMDD")
    g.add_argument("--sessions", default=None,
                   help="comma-sep session tokens, PREFIX match (e.g. 20260501-left "
                        "also matches 20260501-left-3); the token is the middle "
                        "field of source_file {subject}__{session}__{stamp}")
    g.add_argument("--recordings", default=None,
                   help="comma-sep source_file names (exact; trailing .npz optional)")
    return ap


def _session_of(source_file: str) -> str:
    """Session token = middle field of {subject}__{session}__{stamp}.npz."""
    parts = str(source_file).split("__")
    return parts[1] if len(parts) >= 3 else ""


def _csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()] if s else []


def select_clips(df: pd.DataFrame, ns):
    """Filter a clips DataFrame (store.labeled_view output) by the selection in
    argparse namespace `ns`. Returns (filtered_df, scope_dict).

    `scope_dict` = {"filters": <only the non-empty selectors the user gave>,
                    "resolved": <n_clips/n_recordings/n_subjects/subjects/
                                 sessions/date_min/date_max of the kept rows>}.
    """
    df = df.copy()
    sf = df["source_file"].astype(str)
    date = sf.map(io_utils._date_of)
    session = sf.map(_session_of)
    filters: dict = {}
    keep = pd.Series(True, index=df.index)

    subs = _csv(getattr(ns, "subjects", None))
    if subs:
        want = {io_utils.normalize_subject(s) for s in subs}
        keep &= df["subject"].map(lambda s: io_utils.normalize_subject(s) in want)
        filters["subjects"] = subs
    hand = getattr(ns, "only_hand", None)
    if hand:
        keep &= df["hand"].astype(str).str.lower() == hand.lower()
        filters["hand"] = hand
    dates = _csv(getattr(ns, "dates", None))
    if dates:
        keep &= date.map(lambda d: any(d.startswith(p) for p in dates))
        filters["dates"] = dates
    dfrom = getattr(ns, "date_from", None)
    dto = getattr(ns, "date_to", None)
    if dfrom or dto:
        lo, hi = (dfrom or "00000000"), (dto or "99999999")
        keep &= date.map(lambda d: bool(d) and lo <= d <= hi)
        if dfrom:
            filters["date_from"] = dfrom
        if dto:
            filters["date_to"] = dto
    sess = _csv(getattr(ns, "sessions", None))
    if sess:
        keep &= session.map(lambda s: any(s.startswith(p) for p in sess))
        filters["sessions"] = sess
    recs = _csv(getattr(ns, "recordings", None))
    if recs:
        norm = {r if r.endswith(".npz") else r + ".npz" for r in recs}
        keep &= sf.isin(norm)
        filters["recordings"] = sorted(norm)

    out = df[keep].copy()
    osf = out["source_file"].astype(str)
    odate = [d for d in osf.map(io_utils._date_of) if d]
    osess = sorted(set(osf.map(_session_of)) - {""})
    osubj = sorted({io_utils.normalize_subject(s)
                    for s in out["subject"].astype(str)} - {""})
    scope = {
        "filters": filters,
        "resolved": {
            "n_clips": int(len(out)),
            "n_recordings": int(osf.nunique()),
            "n_subjects": len(osubj),
            "subjects": osubj,
            "sessions": osess,
            "date_min": min(odate) if odate else "",
            "date_max": max(odate) if odate else "",
        },
    }
    return out, scope


def scope_tag(scope: dict) -> str:
    """Short, filesystem-safe readable tag for the run-id (e.g. 'ghd1108_20260501-left',
    '5subj_left', 'all'). Derived from the filters + resolved subjects."""
    f = scope.get("filters", {}) if scope else {}
    r = scope.get("resolved", {}) if scope else {}
    parts = []
    if f.get("subjects"):
        subs = r.get("subjects", [])
        parts.append(subs[0] if len(subs) == 1 else f"{len(subs)}subj")
    if f.get("sessions"):
        ss = f["sessions"]
        parts.append(ss[0] if len(ss) == 1 else f"{len(ss)}sess")
    elif f.get("dates"):
        ds = f["dates"]
        parts.append(ds[0] if len(ds) == 1 else f"{len(ds)}dates")
    elif f.get("date_from") or f.get("date_to"):
        parts.append(f"{r.get('date_min', '')}-{r.get('date_max', '')}")
    if f.get("hand"):
        parts.append(f["hand"])
    if f.get("recordings"):
        parts.append(f"{len(f['recordings'])}rec")
    tag = "_".join(str(p) for p in parts) if parts else "all"
    return re.sub(r"[^A-Za-z0-9._+-]", "", tag)[:48] or "all"
