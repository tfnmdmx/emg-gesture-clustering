#!/usr/bin/env bash
# pulse.sh -- one entry point for the EMG gesture segment/cluster/export pipeline.
# Handles the conda python, OMP thread caps, pool building, and stage chaining,
# so you don't have to remember any of it. See RUNBOOK.md.
#
#   ./pulse.sh pool   <batch|path> [...]   build work pool from processed_data batches
#   ./pulse.sh segment                     stage 1: segment
#   ./pulse.sh cluster [K]                  stage 2: cluster (K default 18; "auto" = silhouette)
#   ./pulse.sh eval                         subject-invariance metrics (pooled runs)
#   ./pulse.sh qc                           quality check: feature maps (+ gallery if exported)
#   ./pulse.sh export                       stage 3: export labeled npz (+ build gallery)
#   ./pulse.sh gallery                      build the per-label animation gallery (after export)
#   ./pulse.sh run                          segment + cluster + eval + qc (existing pool, no batches)
#   ./pulse.sh prep   <batch|path> [...]    pool + segment + cluster + eval + qc (everything pre-labeling)
#   ./pulse.sh status                       show what exists so far
#   ./pulse.sh help
#
# Config (override by exporting before calling, e.g. `OUT=out2 ./pulse.sh ...`):
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

: "${PY:=/home/chenglin/anaconda3/envs/emg2pose/bin/python}"
: "${DATA_ROOT:=/data/cl_data/ai-infra/processed_data}"
# NAME ties pool+out to one run name: NAME=foo -> work_pool_foo + out_foo.
# Explicit POOL/OUT still win. No NAME -> plain work_pool/out.
: "${NAME:=}"
if [ -n "$NAME" ]; then
  : "${POOL:=work_pool_$NAME}"
  : "${OUT:=out_$NAME}"
else
  : "${POOL:=work_pool}"
  : "${OUT:=out}"
fi
: "${K:=18}"
: "${GROUP_BY:=subject-hand}"   # subject-hand (per subject+hand) | hand | all
: "${SUBJECT_NORM:=none}"  # per-subject feature norm: none | center | zscore (pooled runs)
: "${N_GALLERY:=3}"        # samples per label in the animation gallery
: "${WORKERS:=1}"          # segment.py multiprocessing workers (one per recording)
# KMeans on this box over-subscribes threads without these caps.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

c_green=$'\033[32m'; c_yellow=$'\033[33m'; c_red=$'\033[31m'; c_off=$'\033[0m'
say()  { echo "${c_green}==>${c_off} $*"; }
warn() { echo "${c_yellow}!! ${c_off} $*"; }
die()  { echo "${c_red}xx ${c_off} $*" >&2; exit 1; }

cmd_pool() {
  [ "$#" -ge 1 ] || die "pool needs >=1 batch name or path. e.g. ./pulse.sh pool fgw0917_0502_left fgw0917_0504_right"
  mkdir -p "$POOL"
  rm -f "$POOL"/*.npz 2>/dev/null || true
  local n=0
  for arg in "$@"; do
    # Accept either a full path or a batch name under DATA_ROOT.
    local dir="$arg"
    [ -d "$dir" ] || dir="$DATA_ROOT/$arg"
    [ -d "$dir" ] || { warn "skip (not a dir): $arg"; continue; }
    local found=0
    for f in "$dir"/*__*__*.npz; do
      [ -e "$f" ] || continue
      ln -sf "$f" "$POOL/$(basename "$f")"
      found=$((found+1))
    done
    [ "$found" -gt 0 ] || warn "no '<subj>__<date-hand>__<ts>.npz' files in $dir"
    n=$((n+found))
  done
  say "pooled $n files into $POOL/"
  [ "$n" -gt 0 ] || die "pool is empty -- check batch names (see: ls $DATA_ROOT)"
  # Show grouping + warn on unparseable files.
  "$PY" - "$POOL" <<'PYEOF'
import sys, glob
from emg_label.io_utils import parse_file_info
pool = sys.argv[1]
groups, bad = {}, []
for p in sorted(glob.glob(f"{pool}/*.npz")):
    info = parse_file_info(p)
    groups.setdefault(info.group, 0)
    groups[info.group] += 1
    if not info.parsed:
        bad.append(p)
print("groups:")
for g, c in sorted(groups.items()):
    print(f"  {g}: {c}")
if bad:
    print(f"\n!! {len(bad)} unparseable file(s) became their own group; "
          f"consider removing those batches:")
    for p in bad[:5]:
        print(f"   {p}")
PYEOF
}

cmd_segment() {
  [ -d "$POOL" ] || die "no pool. run: ./pulse.sh pool <batches...>"
  say "stage 1: segment ($POOL -> $OUT, workers=$WORKERS)"
  "$PY" segment.py "$POOL" --out "$OUT" --workers "$WORKERS"
}

cmd_cluster() {
  [ -f "$OUT/segments.csv" ] || die "no $OUT/segments.csv. run: ./pulse.sh segment"
  local k="${1:-$K}"
  if [ "$k" = "auto" ]; then
    say "stage 2: cluster (auto k via silhouette, group-by=$GROUP_BY, subject-norm=$SUBJECT_NORM)"
    "$PY" cluster.py "$POOL" --out "$OUT" --group-by "$GROUP_BY" \
      --subject-norm "$SUBJECT_NORM"
  else
    say "stage 2: cluster (k=$k, group-by=$GROUP_BY, subject-norm=$SUBJECT_NORM)"
    "$PY" cluster.py "$POOL" --out "$OUT" --k "$k" --group-by "$GROUP_BY" \
      --subject-norm "$SUBJECT_NORM"
  fi
}

cmd_eval() {
  [ -f "$OUT/segments_clustered.csv" ] || die "no clusters yet. run: ./pulse.sh cluster"
  say "evaluation: subject-invariance metrics ($OUT/eval_metrics.csv, subject-norm=$SUBJECT_NORM)"
  "$PY" evaluate.py "$POOL" --out "$OUT" --subject-norm "$SUBJECT_NORM"
}

cmd_qc() {
  [ -f "$OUT/segments_clustered.csv" ] || die "no clusters yet. run: ./pulse.sh cluster"
  say "quality check: feature maps ($OUT/feature_maps/)"
  "$PY" plot_cluster_features.py "$POOL" --out "$OUT"
  # The animation gallery reads exported per-label dirs, which only exist after
  # export. Build it now only if they're there; otherwise export builds it.
  if [ -d "$OUT/segments" ]; then
    cmd_gallery
  else
    warn "animation gallery skipped: needs exported segments (out/segments/)."
    warn "It will be built automatically by './pulse.sh export'."
  fi
}

cmd_gallery() {
  [ -d "$OUT/segments" ] || die "no $OUT/segments/. run ./pulse.sh export first"
  say "animation gallery ($OUT/hand_anim/index.html)"
  "$PY" build_anim_gallery.py --out-root "$OUT" --n "$N_GALLERY" --clean
}

cmd_export() {
  [ -f "$OUT/segments_clustered.csv" ] || die "no clusters yet. run: ./pulse.sh cluster"
  if [ ! -f "$OUT/labels.csv" ]; then
    warn "no $OUT/labels.csv -- generating placeholder labels (label = <group>-<cluster_id>)."
    warn "That exports every cluster as its own gesture. Edit labels.csv for real"
    warn "names / merging (same name = merge, blank = drop), then re-run export."
    # The template's label column is empty; fill it so export is non-empty.
    "$PY" - "$OUT" <<'PYEOF'
import os, sys, pandas as pd
out = sys.argv[1]
df = pd.read_csv(os.path.join(out, "labels_template.csv"))
df["label"] = df["group"].astype(str) + "-" + df["cluster_id"].astype(str)
df.to_csv(os.path.join(out, "labels.csv"), index=False)
print(f"wrote {out}/labels.csv with {len(df)} placeholder labels")
PYEOF
  fi
  say "stage 3: export ($OUT/segments/<label>/)"
  "$PY" export.py "$POOL" --out "$OUT"
  cmd_gallery   # per-label animations now that segments/ exists
}

cmd_run() {
  [ -d "$POOL" ] || die "no pool. run: ./pulse.sh pool <batches...>"
  cmd_segment
  cmd_cluster "$K"
  cmd_eval
  cmd_qc
}

cmd_prep() {
  cmd_pool "$@"
  cmd_run
  echo
  say "PRE-LABELING DONE. Next:"
  echo "  1. Look at  $OUT/clusters/*_hands.png  and  $OUT/feature_maps/*_features.png"
  echo "  2. cp $OUT/labels_template.csv $OUT/labels.csv  then fill the 'label' column"
  echo "     (same name on multiple rows = merge; blank = drop)"
  echo "  3. ./pulse.sh export   (also builds the per-label animation gallery)"
  echo "     -- or skip labeling and just run export for placeholder names."
}

cmd_status() {
  echo "PY      = $PY"
  echo "POOL    = $POOL    ($(ls "$POOL"/*.npz 2>/dev/null | wc -l) files)"
  echo "OUT     = $OUT"
  echo "K       = $K    GROUP_BY = $GROUP_BY    SUBJECT_NORM = $SUBJECT_NORM"
  for f in segments.csv segments_clustered.csv eval_metrics.csv labels_template.csv labels.csv; do
    if [ -f "$OUT/$f" ]; then echo "  [x] $OUT/$f"; else echo "  [ ] $OUT/$f"; fi
  done
  [ -d "$OUT/segments" ] && echo "  [x] $OUT/segments/ ($(ls "$OUT/segments" 2>/dev/null | wc -l) labels, $(find "$OUT/segments" -name '*.npz' 2>/dev/null | wc -l) npz)"
}

cmd_help() {
  # Print the leading comment block (usage), stopping at the first code line.
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

sub="${1:-help}"; shift || true
case "$sub" in
  pool)    cmd_pool "$@";;
  segment) cmd_segment "$@";;
  cluster) cmd_cluster "$@";;
  eval)    cmd_eval "$@";;
  run)     cmd_run "$@";;
  qc)      cmd_qc "$@";;
  gallery) cmd_gallery "$@";;
  export)  cmd_export "$@";;
  prep)    cmd_prep "$@";;
  status)  cmd_status "$@";;
  help|-h|--help) cmd_help;;
  *) die "unknown command: $sub (try: ./pulse.sh help)";;
esac
