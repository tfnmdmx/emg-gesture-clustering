#!/usr/bin/env bash
# pulse.sh -- one entry point for the EMG gesture pipeline. See docs/RUNBOOK.md.
#
# ONE flow. segment.py auto-detects the data shape (processed flat npz, raw
# {subject}/{date-hand}/{stamp}.npz tree, or a sample_meta.csv), so there is a
# single segment command and a single output dir OUT (default `out`).
#
#   ./pulse.sh segment <meta.csv|dir|processed>   stage 1: segment -> OUT/index.db
#                                          'processed' = scan $DATA_ROOT.
#                                          pick subjects with SUBJECTS=id1,id2
#                                          (normalized: fgw-0917 == fgw0917).
#   ./pulse.sh label                       web UI to hand-label clips (writes index.db annotations)
#   ./pulse.sh label-prewarm               pre-render the 3D hand-frame cache (WORKERS=N)
#   ./pulse.sh cluster      [K]            apex (static formed-pose) clustering -> OUT/cluster_runs/{run_id}/
#   ./pulse.sh cluster-traj [K]            trajectory (time-series) clustering  -> OUT/cluster_runs/{run_id}/ (REPR=)
#   ./pulse.sh eval                        label-driven metrics for a run (RUN= or latest) -> metrics.json
#   ./pulse.sh export                      export hand-labelled clips by gesture -> OUT/gestures/<label>/
#   ./pulse.sh qc                          feature maps for a run (+ animation gallery if exported)
#   ./pulse.sh gallery                     per-gesture animation gallery (after export)
#   ./pulse.sh db                          rebuild OUT/index.db from shards
#   ./pulse.sh run     <meta.csv|dir|processed>   segment + cluster + eval (one-click)
#   ./pulse.sh status                      progress
#   ./pulse.sh help
#
# Config (export before calling, e.g. `OUT=out_fgw K=24 ./pulse.sh cluster`):
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

: "${PY:=/home/chenglin/anaconda3/envs/emg2pose/bin/python}"
: "${DATA_ROOT:=/data/cl_data/ai-infra/processed_data}"   # root for `segment processed`
: "${OUT:=out}"            # the single output dir (index.db + shards + cluster_runs + gestures)
: "${K:=18}"               # cluster count ('auto' = silhouette sweep)
: "${GROUP_BY:=subject-hand}"   # apex clustering granularity: subject-hand | hand | all
: "${SUBJECT_NORM:=none}"  # per-subject feature norm: none | center | zscore (pooled runs)
: "${REPR:=centered}"      # cluster-traj representation: centered | velocity | raw
: "${N_GALLERY:=3}"        # samples per gesture in the animation gallery
: "${WORKERS:=1}"          # segment.py workers (one process per recording)
: "${SUBJECTS:=}"          # by person: comma-sep subject ids (normalized match); empty = all
: "${ONLY_HAND:=}"         # by hand: left | right
: "${DATES:=}"             # by time: comma-sep date prefixes (20260502 day | 202605 month | 2026 year)
: "${DATE_FROM:=}"         # by time: keep recordings on/after YYYYMMDD (inclusive)
: "${DATE_TO:=}"           # by time: keep recordings on/before YYYYMMDD (inclusive)
: "${NO_OVERVIEW:=0}"      # set 1 to skip overview.png (CSV-only, ~3-5x faster / no torch)
# KMeans over-subscribes threads on this box without these caps.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

c_green=$'\033[32m'; c_yellow=$'\033[33m'; c_red=$'\033[31m'; c_off=$'\033[0m'
say()  { echo "${c_green}==>${c_off} $*"; }
warn() { echo "${c_yellow}!! ${c_off} $*"; }
die()  { echo "${c_red}xx ${c_off} $*" >&2; exit 1; }

# Optional segment flags, passed only when the env var is set.
_seg_extra() {
  local extra=""
  [ -n "$SUBJECTS" ]  && extra="$extra --subjects $SUBJECTS"     # by person
  [ -n "$ONLY_HAND" ] && extra="$extra --only-hand $ONLY_HAND"   # by hand
  [ -n "$DATES" ]     && extra="$extra --dates $DATES"           # by time (prefixes)
  [ -n "$DATE_FROM" ] && extra="$extra --date-from $DATE_FROM"   # by time (range)
  [ -n "$DATE_TO" ]   && extra="$extra --date-to $DATE_TO"
  [ "$NO_OVERVIEW" = "1" ] && extra="$extra --no-overview"
  # pose-speed motion-threshold knobs (feed the auto move-enter/exit derivation);
  # lower POSE_PCT/POSE_MAD to over-segment. Only passed when set, else defaults.
  [ -n "${POSE_PCT:-}" ]     && extra="$extra --pose-pct $POSE_PCT"
  [ -n "${POSE_MAD:-}" ]     && extra="$extra --pose-mad $POSE_MAD"
  [ -n "${MIN_STATIC_S:-}" ] && extra="$extra --min-static-s $MIN_STATIC_S"
  [ -n "${MIN_MOTION_S:-}" ] && extra="$extra --min-motion-s $MIN_MOTION_S"
  echo "$extra"
}

cmd_segment() {
  # ONE source arg: a sample_meta.csv, a directory (processed flat OR raw
  # {subject}/{date-hand}/{stamp}.npz, scanned recursively), or 'processed' (=
  # $DATA_ROOT). Pick subjects with SUBJECTS=. segment.py writes source_path into
  # index.db, so downstream stages locate the npz with no work_pool needed.
  [ "$#" -ge 1 ] && [ -n "${1:-}" ] || die "segment needs a source: a sample_meta.csv, a dir, or 'processed' (\$DATA_ROOT). e.g. SUBJECTS=fgw-0917 ./pulse.sh segment processed"
  local src="$1"
  [ "$src" = "processed" ] && src="$DATA_ROOT"
  local extra; extra=$(_seg_extra)
  if [ -f "$src" ] && [[ "$src" == *.csv ]]; then
    say "segment (--meta $src -> $OUT, workers=$WORKERS, subjects=${SUBJECTS:-all})"
    # shellcheck disable=SC2086
    "$PY" segment.py --meta "$src" --out "$OUT" --workers "$WORKERS" $extra
  elif [ -d "$src" ]; then
    say "segment ($src -> $OUT, workers=$WORKERS, subjects=${SUBJECTS:-all})"
    # shellcheck disable=SC2086
    "$PY" segment.py "$src" --recursive --out "$OUT" --workers "$WORKERS" $extra
  else
    die "segment source not found: '$src'. Pass a sample_meta.csv, a source dir, or 'processed' (\$DATA_ROOT=$DATA_ROOT)."
  fi
}

cmd_cluster() {
  [ -f "$OUT/index.db" ] || die "no $OUT/index.db. run: ./pulse.sh segment <source>"
  local k="${1:-$K}"
  local kflag=""; [ "$k" != "auto" ] && kflag="--k $k"
  say "apex clustering (k=$k, group-by=$GROUP_BY, subject-norm=$SUBJECT_NORM) -> $OUT/cluster_runs/"
  # shellcheck disable=SC2086
  "$PY" cluster.py --out "$OUT" --group-by "$GROUP_BY" \
    --subject-norm "$SUBJECT_NORM" $kflag
}

cmd_cluster_traj() {
  [ -f "$OUT/index.db" ] || die "no $OUT/index.db. run: ./pulse.sh segment <source>"
  local k="${1:-$K}"
  local kflag=""; [ "$k" != "auto" ] && kflag="--k $k"
  say "trajectory (time-series) clustering (k=$k, repr=$REPR) -> $OUT/cluster_runs/"
  # shellcheck disable=SC2086
  "$PY" cluster_traj.py --out "$OUT" --repr "$REPR" $kflag
}

cmd_eval() {
  [ -f "$OUT/index.db" ] || die "no $OUT/index.db. run: ./pulse.sh cluster first"
  say "evaluation: label-driven + subject-invariance (--run ${RUN:-latest}, writes cluster_runs/<run>/metrics.json)"
  local runflag=""; [ -n "${RUN:-}" ] && runflag="--run $RUN"
  # shellcheck disable=SC2086
  "$PY" evaluate.py --out "$OUT" --subject-norm "$SUBJECT_NORM" $runflag
}

cmd_db() {
  # Rebuild OUT/index.db (+ convenience CSVs) from the shards on disk.
  say "rebuild index.db from $OUT/shards/"
  "$PY" segment.py --out "$OUT" --index-only
}

cmd_qc() {
  [ -f "$OUT/index.db" ] || die "no clusters yet. run: ./pulse.sh cluster"
  say "quality check: feature maps for run ${RUN:-latest} -> cluster_runs/<run>/feature_maps.png"
  local runflag=""; [ -n "${RUN:-}" ] && runflag="--run $RUN"
  # shellcheck disable=SC2086
  "$PY" plot_cluster_features.py --out "$OUT" $runflag
  if [ -d "$OUT/gestures" ]; then
    cmd_gallery
  else
    warn "animation gallery skipped: needs labelled clips exported to $OUT/gestures/"
    warn "(label clips via './pulse.sh label', then './pulse.sh export')."
  fi
}

cmd_gallery() {
  [ -d "$OUT/gestures" ] || die "no $OUT/gestures/. run ./pulse.sh export first"
  say "animation gallery ($OUT/hand_anim/index.html)"
  "$PY" build_anim_gallery.py --out-root "$OUT" --n "$N_GALLERY" --clean
}

cmd_export() {
  [ -f "$OUT/index.db" ] || die "no $OUT/index.db. run: ./pulse.sh segment <source>"
  # Label-driven: exports the hand-labelled clips (annotations) grouped by
  # gesture. Label clips in the web UI first (./pulse.sh label).
  say "export labelled clips by gesture ($OUT/gestures/<label>/)"
  "$PY" export.py --out "$OUT"
}

cmd_label() {
  : "${PORT:=8000}" ; : "${HOST:=127.0.0.1}"
  [ -f "$OUT/index.db" ] || die "no $OUT/index.db. run: ./pulse.sh segment <source>"
  # Regenerate overviews live by default: shard-baked static overview.png can be
  # stale after pose/plot fixes (units, axis caps). Set OVERVIEW_REGEN=0 to use
  # the static ones -- much faster first open (no per-open matplotlib re-render,
  # ~10s on a loaded box), and correct as long as the shards were baked by the
  # current segment.py. For large datasets, pre-warm the open-time caches first
  # (with the SAME OVERVIEW_REGEN you serve with), then opens are instant:
  #   OVERVIEW_REGEN=0 WORKERS=8 ./pulse.sh label-prewarm
  export OVERVIEW_REGEN="${OVERVIEW_REGEN:-1}"
  say "labelling UI: $OUT -> http://$HOST:$PORT (OVERVIEW_REGEN=$OVERVIEW_REGEN)"
  local nwarm nrec
  nwarm=$(ls "$OUT/cache/recording_cache" 2>/dev/null | wc -l)
  nrec=$(ls "$OUT/shards" 2>/dev/null | wc -l)
  if [ "$nrec" -gt 200 ] && [ "$nwarm" -lt "$((nrec / 2))" ]; then
    warn "open-time caches are cold ($nwarm/$nrec warmed): first open of each"
    warn "recording will be slow. Speed it up with OVERVIEW_REGEN=0 (serve the"
    warn "baked overviews) and/or ./pulse.sh label-prewarm."
  fi
  exec "$PY" label_server.py --out "$OUT" --host "$HOST" --port "$PORT"
}

cmd_label_prewarm() {
  : "${WORKERS:=4}"
  export OVERVIEW_REGEN="${OVERVIEW_REGEN:-1}"   # warm the same caches label serves
  say "prewarm open-time + hand-frame caches for $OUT (workers=$WORKERS, OVERVIEW_REGEN=$OVERVIEW_REGEN)"
  exec "$PY" prewarm_hands.py --out "$OUT" --workers "$WORKERS" "$@"
}

cmd_run() {
  # one-click: segment -> apex cluster -> evaluate. (labels needed for a
  # meaningful eval; with none it still writes a run + N/A label metrics. Label
  # via './pulse.sh label', then re-run eval / export.)
  cmd_segment "$@"
  cmd_cluster "$K"
  cmd_eval
}

cmd_status() {
  echo "PY      = $PY"
  echo "OUT     = $OUT"
  echo "K       = $K    GROUP_BY = $GROUP_BY    SUBJECT_NORM = $SUBJECT_NORM    REPR = $REPR"
  for f in index.db clips.csv bursts.csv recordings.csv; do
    if [ -f "$OUT/$f" ]; then echo "  [x] $OUT/$f"; else echo "  [ ] $OUT/$f"; fi
  done
  [ -d "$OUT/cluster_runs" ] && echo "  [x] $OUT/cluster_runs/ ($(ls "$OUT/cluster_runs" 2>/dev/null | wc -l) run(s))"
  [ -d "$OUT/gestures" ] && echo "  [x] $OUT/gestures/ ($(ls "$OUT/gestures" 2>/dev/null | wc -l) gestures, $(find "$OUT/gestures" -name '*.npz' 2>/dev/null | wc -l) npz)"
}

cmd_help() {
  # Print the leading comment block (usage), stopping at the first code line.
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

sub="${1:-help}"; shift || true
case "$sub" in
  segment)      cmd_segment "$@";;
  label)        cmd_label "$@";;
  label-prewarm) cmd_label_prewarm "$@";;
  cluster)      cmd_cluster "$@";;
  cluster-traj) cmd_cluster_traj "$@";;
  eval)         cmd_eval "$@";;
  export)       cmd_export "$@";;
  qc)           cmd_qc "$@";;
  gallery)      cmd_gallery "$@";;
  db)           cmd_db "$@";;
  run)          cmd_run "$@";;
  status)       cmd_status "$@";;
  help|-h|--help) cmd_help;;
  *) die "unknown command: $sub (try: ./pulse.sh help)";;
esac
