#!/usr/bin/env bash
# LoCoMo full experiment pipeline
#
# NOTE: This script is Unix/macOS only. For Windows (or cross-platform),
#       use the Python equivalent instead:
#       python experiments/locomo/benchmark/start_experiment.py
#
# Reads experiment_sessions.json for the conversation list, then for each
# conversation runs graph construction + full QA evaluation with two strategies:
#   1. hybrid     — TF-IDF + LLM joint graph construction
#   2. hash_only  — TF-IDF/hash baseline graph construction
#
# Uses `python -m mosaic build` and `python -m mosaic query` directly.
#
# Directory layout (per conversation):
#   runs/<conv_id>/artifacts/{hybrid,hash_only}/  — graph pkl, tags, snapshots
#   runs/<conv_id>/results/                        — QA eval full & summary JSONs
#   runs/<conv_id>/log/                            — telemetry, llm_io, progress
#
# Usage (one command, background, everything done):
#   cd experiments/locomo/benchmark && nohup ./start_experiment.sh > runs/experiment.log 2>&1 &
#   tail -f runs/experiment.log        # monitor progress
set -euo pipefail

LOCOMO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LOCOMO_DIR"

# ── Conda / env setup ───────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" ]]; then
  echo "[error] conda not found."
  exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mosaic

REPO="$(cd "$LOCOMO_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO"

STRATEGIES="hybrid hash_only"

# ── Read session list from dataset/locomo/experiment_sessions.json ──
DATASET_DIR="$REPO/dataset/locomo"
SESSIONS_JSON="$DATASET_DIR/experiment_sessions.json"
if [[ ! -f "$SESSIONS_JSON" ]]; then
  echo "[error] $SESSIONS_JSON not found."
  exit 1
fi

mkdir -p "$LOCOMO_DIR/runs"

# Parse sessions into parallel arrays
eval "$(python3 -c "
import json
data = json.load(open('$SESSIONS_JSON'))
ids = [s['conv_id'] for s in data['sessions']]
cjs = [s['conv_json'] for s in data['sessions']]
qjs = [s['qa_json'] for s in data['sessions']]
print('CONV_IDS=(' + ' '.join(ids) + ')')
print('CONV_JSONS=(' + ' '.join(['\"$DATASET_DIR/' + c + '\"' for c in cjs]) + ')')
print('QA_JSONS=(' + ' '.join(['\"$DATASET_DIR/' + q + '\"' for q in qjs]) + ')')
")"

echo "[check] strategies: $STRATEGIES"
echo "[check] conversations: ${CONV_IDS[*]}"
echo "[check] repo root: $REPO"
echo "[check] python: $(which python)"
echo ""

# ── Validate all input files exist ───────────────────────────────────
for i in "${!CONV_IDS[@]}"; do
  CONV="${CONV_IDS[$i]}"
  CONV_JSON="${CONV_JSONS[$i]}"
  QA_JSON="${QA_JSONS[$i]}"
  if [[ ! -f "$CONV_JSON" ]]; then
    echo "[error] conversation file not found: $CONV_JSON"
    exit 1
  fi
  if [[ ! -f "$QA_JSON" ]]; then
    echo "[error] QA file not found: $QA_JSON"
    exit 1
  fi
  echo "[ok] $CONV: conv=$CONV_JSON  qa=$QA_JSON"
done
echo ""

# ── Main pipeline ────────────────────────────────────────────────────
T_PIPELINE_START=$(date +%s)

for i in "${!CONV_IDS[@]}"; do
  CONV="${CONV_IDS[$i]}"
  CONV_JSON="${CONV_JSONS[$i]}"
  QA_JSON="${QA_JSONS[$i]}"
  QA_IDX="${CONV#conv}"

  RUN_DIR="$LOCOMO_DIR/runs/$CONV"
  LOG_DIR="$RUN_DIR/log"
  ARTIFACTS_DIR="$RUN_DIR/artifacts"
  RESULTS_DIR="$RUN_DIR/results"
  mkdir -p "$LOG_DIR" "$RESULTS_DIR"

  export MOSAIC_LLM_IO_LOG="${LOG_DIR}/llm_io.jsonl"
  export MOSAIC_INGEST_JSONL="${LOG_DIR}/ingest.jsonl"
  export MOSAIC_NCS_TRACE_JSONL="${LOG_DIR}/ncs_trace.jsonl"
  export MOSAIC_CONTROL_PROFILE="${MOSAIC_CONTROL_PROFILE:-static}"

  for STRATEGY in $STRATEGIES; do
    LABEL="${CONV}_${STRATEGY}"
    ART_DIR="$ARTIFACTS_DIR/$STRATEGY"
    mkdir -p "$ART_DIR/graph_snapshots"

    echo ""
    echo "############################################################"
    echo "#  $LABEL"
    echo "############################################################"

    HASH_FLAG=""
    if [[ "$STRATEGY" == hash_only ]]; then
      HASH_FLAG="--hash"
    fi

    G_PKL="$ART_DIR/graph_network_${CONV}.pkl"
    T_JSON="$ART_DIR/${CONV}_tags.json"

    # --- 1. Build graph ---
    echo "========== [$LABEL 1/3] build ${STRATEGY} =========="
    T0=$(date +%s)
    python -m mosaic build \
      --conv-json "$CONV_JSON" \
      --conv-name "$CONV" \
      --graph-save-dir "$ART_DIR/graph_snapshots" \
      --graph-out "$G_PKL" \
      --tags-out "$T_JSON" \
      --progress-file "$LOG_DIR/${CONV}_progress.txt" \
      --log-prompt "$LOG_DIR/llm_io.jsonl" \
      $HASH_FLAG
    T1=$(date +%s)
    eval "TIMING_${CONV}_${STRATEGY}_graph=$((T1 - T0))"

    # --- 2. Smoke query ---
    echo "========== [$LABEL 2/3] smoke query ${STRATEGY} method=hash =========="
    T0=$(date +%s)
    python -m mosaic query \
      --graph-pkl "$G_PKL" \
      --tags-json "$T_JSON" \
      --method hash \
      --question "Who are the people in this conversation?" \
      || echo "[warn] $LABEL smoke query failed"
    T1=$(date +%s)
    eval "TIMING_${CONV}_${STRATEGY}_smoke=$((T1 - T0))"

    # --- 3. Full QA evaluation ---
    echo "========== [$LABEL 3/3] QA eval method=hash results-tag=${STRATEGY} =========="
    T0=$(date +%s)
    python -m mosaic query \
      --graph-pkl "$G_PKL" \
      --tags-json "$T_JSON" \
      --qa-json "$QA_JSON" \
      --method hash \
      --output "$RESULTS_DIR/qa_${QA_IDX}_eval_full_${STRATEGY}.json" \
      --summary-out "$RESULTS_DIR/qa_${QA_IDX}_eval_summary_${STRATEGY}.json" \
      --resume
    T1=$(date +%s)
    eval "TIMING_${CONV}_${STRATEGY}_qa=$((T1 - T0))"

    eval "_g=\$TIMING_${CONV}_${STRATEGY}_graph"
    eval "_s=\$TIMING_${CONV}_${STRATEGY}_smoke"
    eval "_q=\$TIMING_${CONV}_${STRATEGY}_qa"
    echo "========== [$LABEL] timing =========="
    echo "  graph=$_g  smoke=$_s  qa=$_q"
  done
done

# ── Timing summary ──────────────────────────────────────────────────
T_PIPELINE_END=$(date +%s)
TOTAL_WALL=$((T_PIPELINE_END - T_PIPELINE_START))

echo ""
echo "========== Timing Summary (seconds) =========="
for CONV in "${CONV_IDS[@]}"; do
  for STRATEGY in $STRATEGIES; do
    LABEL="${CONV}_${STRATEGY}"
    eval "_g=\${TIMING_${CONV}_${STRATEGY}_graph:-?}"
    eval "_s=\${TIMING_${CONV}_${STRATEGY}_smoke:-?}"
    eval "_q=\${TIMING_${CONV}_${STRATEGY}_qa:-?}"
    echo "  $LABEL: graph=$_g smoke=$_s qa=$_q"
  done
done
echo "  total_wall=$TOTAL_WALL"

echo ""
echo "========== Done (all conversations × strategies) =========="
echo "Results: $LOCOMO_DIR/runs/*/results/"
echo "Dataset: $DATASET_DIR"
