#!/usr/bin/env bash
# Quick test: re-evaluate conv5 only on category 1 (multi-hop) and category 3 (single-hop).
#
# Uses existing conv5 graph from runs/runs/conv5/artifacts/.
#
# Usage:
#   cd experiments/locomo/benchmark && ./_update_c1_c3_query_conv5.sh
set -euo pipefail

LOCOMO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LOCOMO_DIR"

REPO="$(cd "$LOCOMO_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO"
export MOSAIC_VERBOSE=1

STRATEGIES="hybrid hash_only"
CATEGORIES="1 3"
TAG="c1_c3"
CONV="conv5"
QA_IDX="5"

DATASET_DIR="$REPO/dataset/locomo"
QA_JSON="$DATASET_DIR/qa_${QA_IDX}.json"

RUNS_BASE="$LOCOMO_DIR/runs/runs"
if [[ ! -d "$RUNS_BASE/$CONV" ]]; then
  RUNS_BASE="$LOCOMO_DIR/runs"
fi

UPDATE_DIR="$LOCOMO_DIR/runs/update_${TAG}_${CONV}"
RESULTS_DIR="$UPDATE_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "=========================================================="
echo "  Conv5 test — categories ${CATEGORIES}"
echo "  Runs base: $RUNS_BASE/$CONV"
echo "  Results:   $RESULTS_DIR"
echo "=========================================================="

T_START=$(date +%s)

for STRATEGY in $STRATEGIES; do
  LABEL="${CONV}_${STRATEGY}"
  G_PKL="$RUNS_BASE/$CONV/artifacts/$STRATEGY/graph_network_${CONV}.pkl"
  T_JSON="$RUNS_BASE/$CONV/artifacts/$STRATEGY/${CONV}_tags.json"

  if [[ ! -f "$G_PKL" ]] || [[ ! -f "$T_JSON" ]]; then
    echo "[skip] $LABEL — missing artifacts"
    echo "  graph: $G_PKL"
    echo "  tags:  $T_JSON"
    continue
  fi

  echo ""
  echo "############################################################"
  echo "#  $LABEL — categories ${CATEGORIES}"
  echo "############################################################"

  OUT_FILE="$RESULTS_DIR/qa_${QA_IDX}_eval_${TAG}_${STRATEGY}.json"
  SUM_FILE="$RESULTS_DIR/qa_${QA_IDX}_summary_${TAG}_${STRATEGY}.json"

  T0=$(date +%s)
  python -m mosaic query \
    --graph-pkl "$G_PKL" \
    --tags-json "$T_JSON" \
    --qa-json "$QA_JSON" \
    --method hash \
    --only-categories $CATEGORIES \
    --output "$OUT_FILE" \
    --summary-out "$SUM_FILE" \
    --resume
  T1=$(date +%s)
  echo "  [$LABEL] QA time: $((T1 - T0))s"
done

T_END=$(date +%s)
TOTAL=$((T_END - T_START))

echo ""
echo "=========================================================="
echo "  Results"
echo "=========================================================="

python3 -c "
import json, os

results_dir = '$RESULTS_DIR'
strategies = '$STRATEGIES'.split()
tag = '$TAG'
cat_names = {1: 'Multi-hop', 3: 'Single-hop'}

for strategy in strategies:
    sf = os.path.join(results_dir, 'qa_${QA_IDX}_summary_${TAG}_' + strategy + '.json')
    if not os.path.isfile(sf):
        print(f'  {strategy}: no results')
        continue
    with open(sf) as f:
        data = json.load(f)
    s = data.get('summary', {})
    tc = s.get('total_correct', 0)
    tt = s.get('total_questions', 0)
    acc = tc / tt if tt > 0 else 0
    print(f'  {strategy}: {tc}/{tt} = {acc:.1%}')
    for ck, cs in sorted(s.get('category_stats', {}).items(), key=lambda x: int(x[0])):
        c = cs.get('correct', 0)
        t = cs.get('total', 0)
        a = c / t if t > 0 else 0
        name = cat_names.get(int(ck), f'cat{ck}')
        print(f'    {name}: {c}/{t} = {a:.1%}')
"

echo ""
echo "Total wall time: ${TOTAL}s"
echo "Results: $RESULTS_DIR"
