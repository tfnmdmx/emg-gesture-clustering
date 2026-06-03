#!/usr/bin/env bash
# pulse.sh -- one entry point for the EMG gesture pipeline. See RUNBOOK.md.
#
# Two flows -- pick by data shape:
#
# A. RAW data -> ground-truth clip gallery (path A in RUNBOOK):
#   ./pulse.sh raw       <meta.csv | raw-dir>   segment + QC filter + clip gallery (chained)
#   ./pulse.sh raw-segment <meta.csv | raw-dir> stage 1 only (raw input)
#   ./pulse.sh raw-qc                            write recordings_keep.csv (informational subset)
#   ./pulse.sh raw-export                        stage 3 only: per-clip npz/png + index.html
#   ./pulse.sh raw-index                         rebuild top-level segments/clips/recordings.csv from existing shards
#   ./pulse.sh raw-status                        show raw-flow progress
#   ./pulse.sh label                             interactive web UI to hand-label clips (writes labels.csv)
#   ./pulse.sh label-prewarm                     pre-render hand-frame cache (WORKERS=N) so first open is instant
#
# B. PROCESSED data -> unsupervised cluster -> labeled npz (path B in RUNBOOK):
#   ./pulse.sh pool   <batch|path> [...]   build work pool from processed_data batches
#   ./pulse.sh segment                     stage 1: segment
#   ./pulse.sh cluster [K]                 stage 2: cluster (K default 18; "auto" = silhouette)
#   ./pulse.sh eval                        subject-invariance metrics (pooled runs)
#   ./pulse.sh qc                          quality check: feature maps (+ gallery if exported)
#   ./pulse.sh export                      stage 3: export labeled npz (+ build gallery)
#   ./pulse.sh gallery                     build the per-label animation gallery (after export)
#   ./pulse.sh run                         segment + cluster + eval + qc (existing pool, no batches)
#   ./pulse.sh prep   <batch|path> [...]   pool + segment + cluster + eval + qc (everything pre-labeling)
#
# Shared:
#   ./pulse.sh status                      show flow-B progress
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

# --- flow-A (raw data -> ground-truth) defaults ----------------------------
: "${RAW_OUT:=out_pose}"           # output dir for raw-data flow
: "${RAW_LAG_FLAGS:=ok}"           # recordings.lag_flag values to keep (comma-sep)
: "${RAW_NAN_MAX:=0.01}"           # max pose_nan_frac to keep in recordings_keep.csv
: "${SUBJECT:=}"                   # filter segment.py to one subject (--only-subject)
: "${ONLY_HAND:=}"                 # filter segment.py to one hand (left|right)
: "${NO_OVERVIEW:=0}"              # set to 1 to skip overview.png in segment.py
: "${RAW_LIMIT:=}"                 # cap clips per source in export_clips.py
: "${RAW_NO_PNG:=0}"               # set to 1 to skip keyframe PNG rendering
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

# ---------- flow A (raw data -> ground-truth clip gallery) -----------------

# Resolve the raw-flow input arg. We accept either a sample_meta.csv path or
# a directory tree of raw npz; downstream segment.py picks the right entry
# point based on which flag we set.
_raw_input_to_args() {
  local arg="$1"
  if [ -f "$arg" ] && [[ "$arg" == *.csv ]]; then
    echo "--meta $arg"
  elif [ -d "$arg" ]; then
    echo "$arg --recursive"
  else
    die "raw input must be a sample_meta CSV or a raw-data directory: $arg"
  fi
}

# Build the optional-flag tail shared by raw_segment and raw_export.
_raw_seg_extra() {
  local extra=""
  [ -n "$SUBJECT" ] && extra="$extra --only-subject $SUBJECT"
  [ -n "$ONLY_HAND" ] && extra="$extra --only-hand $ONLY_HAND"
  [ "$NO_OVERVIEW" = "1" ] && extra="$extra --no-overview"
  # pose-segmentation thresholds: lower these to over-segment (more clips),
  # then prune by labelling. Only passed when set, else segment.py defaults.
  [ -n "${POSE_PCT:-}" ]     && extra="$extra --pose-pct $POSE_PCT"          # default 35 -> lower = more motion
  [ -n "${POSE_MAD:-}" ]     && extra="$extra --pose-mad $POSE_MAD"          # default 1.5
  [ -n "${MIN_STATIC_S:-}" ] && extra="$extra --min-static-s $MIN_STATIC_S"  # default 0.35
  [ -n "${MIN_MOTION_S:-}" ] && extra="$extra --min-motion-s $MIN_MOTION_S"  # default 0.20
  [ -n "${MERGE_GAP_S:-}" ]  && extra="$extra --merge-gap-s $MERGE_GAP_S"    # default 0.20
  echo "$extra"
}

cmd_raw_segment() {
  [ "$#" -ge 1 ] || die "raw-segment needs <meta.csv | raw-dir>"
  local args
  args=$(_raw_input_to_args "$1")
  local extra
  extra=$(_raw_seg_extra)
  say "stage 1 (raw): segment ($1 -> $RAW_OUT, workers=$WORKERS$extra)"
  # shellcheck disable=SC2086
  "$PY" segment.py $args --out "$RAW_OUT" --workers "$WORKERS" $extra
}

# Write recordings_keep.csv = subset that passes (lag_flag in $RAW_LAG_FLAGS,
# pose_nan_frac < $RAW_NAN_MAX). Purely informational; export_clips.py still
# reads the full clips.csv. The user decides whether to label clips from the
# rejected recordings (typically: no).
cmd_raw_qc() {
  [ -f "$RAW_OUT/recordings.csv" ] || die "no $RAW_OUT/recordings.csv. run: ./pulse.sh raw-segment <input>"
  say "QC filter: lag_flag in ($RAW_LAG_FLAGS), pose_nan_frac < $RAW_NAN_MAX"
  "$PY" - "$RAW_OUT" "$RAW_LAG_FLAGS" "$RAW_NAN_MAX" <<'PYEOF'
import os, sys, pandas as pd
out_dir, flags_raw, nan_max = sys.argv[1], sys.argv[2], float(sys.argv[3])
flags = [s.strip() for s in flags_raw.split(",") if s.strip()]
r = pd.read_csv(os.path.join(out_dir, "recordings.csv"))
mask = r["lag_flag"].isin(flags) & (r["pose_nan_frac"] < nan_max)
keep = r[mask]
keep_path = os.path.join(out_dir, "recordings_keep.csv")
keep.to_csv(keep_path, index=False)
print(f"keep {len(keep)}/{len(r)} recordings -> {keep_path}")
# Bonus: per-flag breakdown so the user sees what was dropped.
print("\nbreakdown:")
print(r["lag_flag"].value_counts().to_string())
print(f"\npose_nan_frac>={nan_max}: {(r['pose_nan_frac']>=nan_max).sum()} recordings")
PYEOF
}

cmd_raw_export() {
  [ -f "$RAW_OUT/clips.csv" ] || die "no $RAW_OUT/clips.csv. run: ./pulse.sh raw-segment <input>"
  local extra=""
  [ -n "$RAW_LIMIT" ] && extra="$extra --limit $RAW_LIMIT"
  [ "$RAW_NO_PNG" = "1" ] && extra="$extra --no-png"
  say "stage 3 (raw): export clips ($RAW_OUT/clips_export/$extra)"
  # shellcheck disable=SC2086
  "$PY" export_clips.py --out "$RAW_OUT" $extra
}

# Rebuild the three top-level index CSVs (segments/clips/recordings.csv) by
# concatenating every completed shard under $RAW_OUT/shards/. Use this when a
# `raw` run was interrupted before segment.py reached its end-of-run index step
# (each shard is committed independently, so the per-recording data survives --
# only the top-level concat is missing). No recordings are reprocessed.
cmd_raw_index() {
  [ -d "$RAW_OUT/shards" ] || die "no $RAW_OUT/shards/. run: ./pulse.sh raw-segment <input>"
  say "rebuild index from shards ($RAW_OUT/{segments,clips,recordings}.csv)"
  "$PY" segment.py --out "$RAW_OUT" --index-only
}

cmd_raw() {
  [ "$#" -ge 1 ] || die "raw needs <meta.csv | raw-dir>. e.g. ./pulse.sh raw reference/sample_meta.csv"
  cmd_raw_segment "$1"
  cmd_raw_qc
  cmd_raw_export
  echo
  say "RAW PIPELINE DONE. Next:"
  echo "  1. Open $RAW_OUT/clips_export/index.html in a browser"
  echo "  2. Fill 'gesture_label' column in $RAW_OUT/clips.csv while scanning keyframes"
  echo "  3. (Optional) drop clips whose recording isn't in $RAW_OUT/recordings_keep.csv"
}

cmd_raw_status() {
  echo "PY        = $PY"
  echo "RAW_OUT   = $RAW_OUT"
  echo "WORKERS   = $WORKERS    SUBJECT=$SUBJECT    ONLY_HAND=$ONLY_HAND"
  echo "QC keep   = lag_flag in ($RAW_LAG_FLAGS), pose_nan_frac<$RAW_NAN_MAX"
  for f in segments.csv clips.csv recordings.csv recordings_keep.csv; do
    if [ -f "$RAW_OUT/$f" ]; then
      local n; n=$(($(wc -l < "$RAW_OUT/$f") - 1))
      echo "  [x] $RAW_OUT/$f  ($n rows)"
    else
      echo "  [ ] $RAW_OUT/$f"
    fi
  done
  if [ -d "$RAW_OUT/shards" ]; then
    local nshard; nshard=$(find "$RAW_OUT/shards" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  [x] $RAW_OUT/shards/  ($nshard recordings)"
  fi
  if [ -d "$RAW_OUT/clips_export" ]; then
    local nnpz; nnpz=$(ls "$RAW_OUT/clips_export"/*.npz 2>/dev/null | wc -l)
    local npng; npng=$(ls "$RAW_OUT/clips_export"/*.png 2>/dev/null | wc -l)
    echo "  [x] $RAW_OUT/clips_export/  ($nnpz npz, $npng png)"
    [ -f "$RAW_OUT/clips_export/index.html" ] && \
      echo "      browse: $RAW_OUT/clips_export/index.html"
  fi
}

cmd_label() {
  : "${PORT:=8000}" ; : "${HOST:=127.0.0.1}"
  # Regenerate overviews live by default: shard-baked static overview.png are
  # stale after pose/plot fixes (units, axis caps). Set OVERVIEW_REGEN=0 to use
  # the static ones (faster first open, but only correct right after segment).
  export OVERVIEW_REGEN="${OVERVIEW_REGEN:-1}"
  say "interactive labelling UI: $RAW_OUT -> http://$HOST:$PORT (OVERVIEW_REGEN=$OVERVIEW_REGEN)"
  exec "$PY" label_server.py --out "$RAW_OUT" --host "$HOST" --port "$PORT"
}

cmd_label_prewarm() {
  : "${WORKERS:=4}"
  say "prewarm hand-frame cache for $RAW_OUT (workers=$WORKERS)"
  exec "$PY" prewarm_hands.py --out "$RAW_OUT" --workers "$WORKERS" "$@"
}

cmd_help() {
  # Print the leading comment block (usage), stopping at the first code line.
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

sub="${1:-help}"; shift || true
case "$sub" in
  # flow A: raw -> ground-truth clip gallery
  raw)         cmd_raw "$@";;
  raw-segment) cmd_raw_segment "$@";;
  raw-qc)      cmd_raw_qc "$@";;
  raw-export)  cmd_raw_export "$@";;
  raw-index)   cmd_raw_index "$@";;
  raw-status)  cmd_raw_status "$@";;
  label)         cmd_label "$@";;
  label-prewarm) cmd_label_prewarm "$@";;
  # flow B: processed -> cluster -> labeled npz
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
