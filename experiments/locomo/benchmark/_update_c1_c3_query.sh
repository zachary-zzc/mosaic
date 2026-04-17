#!/usr/bin/env bash
# Re-evaluate existing graphs on category 1 (multi-hop) and category 3 (single-hop) only.
#
# Uses graphs already built by start_experiment.sh (in runs/runs/<conv>/artifacts/).
# Skips graph construction entirely — only runs QA evaluation with the updated
# query pipeline (DFS neighbor expansion, no context truncation).
#
# Usage:
#   cd experiments/locomo/benchmark && nohup ./_update_c1_c3_query.sh > runs/update_c1_c3.log 2>&1 &
#   tail -f runs/update_c1_c3.log
set -euo pipefail

LOCOMO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LOCOMO_DIR"

REPO="$(cd "$LOCOMO_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO"
export MOSAIC_VERBOSE=1

STRATEGIES="hybrid hash_only"
CATEGORIES="1 3"
TAG="c1_c3"

# ── Dataset & session list ──────────────────────────────────────────
DATASET_DIR="$REPO/dataset/locomo"
SESSIONS_JSON="$DATASET_DIR/experiment_sessions.json"
if [[ ! -f "$SESSIONS_JSON" ]]; then
  echo "[error] $SESSIONS_JSON not found."
  exit 1
fi

# ── Locate existing runs (prefer runs/runs/ which has all conv0-conv9) ──
RUNS_BASE="$LOCOMO_DIR/runs/runs"
if [[ ! -d "$RUNS_BASE" ]]; then
  RUNS_BASE="$LOCOMO_DIR/runs"
fi
echo "[info] Using runs base: $RUNS_BASE"

# Results go to a dedicated update directory
UPDATE_DIR="$LOCOMO_DIR/runs/update_${TAG}"
RESULTS_DIR="$UPDATE_DIR/results"
mkdir -p "$RESULTS_DIR"

# Parse sessions
eval "$(python3 -c "
import json
data = json.load(open('$SESSIONS_JSON'))
ids = [s['conv_id'] for s in data['sessions']]
qjs = [s['qa_json'] for s in data['sessions']]
print('CONV_IDS=(' + ' '.join(ids) + ')')
print('QA_JSONS=(' + ' '.join(['\"$DATASET_DIR/' + q + '\"' for q in qjs]) + ')')
")"

echo "[check] strategies: $STRATEGIES"
echo "[check] categories: $CATEGORIES"
echo "[check] conversations: ${CONV_IDS[*]}"
echo "[check] python: $(which python)"
echo "[check] results: $RESULTS_DIR"
echo ""

# ── Validate all graph files exist ──────────────────────────────────
MISSING=0
for CONV in "${CONV_IDS[@]}"; do
  for STRATEGY in $STRATEGIES; do
    G_PKL="$RUNS_BASE/$CONV/artifacts/$STRATEGY/graph_network_${CONV}.pkl"
    T_JSON="$RUNS_BASE/$CONV/artifacts/$STRATEGY/${CONV}_tags.json"
    if [[ ! -f "$G_PKL" ]]; then
      echo "[warn] missing graph: $G_PKL"
      MISSING=$((MISSING + 1))
    fi
    if [[ ! -f "$T_JSON" ]]; then
      echo "[warn] missing tags: $T_JSON"
      MISSING=$((MISSING + 1))
    fi
  done
done
if [[ $MISSING -gt 0 ]]; then
  echo "[warn] $MISSING missing artifact(s). Will skip those conversations."
fi
echo ""

# ── Main QA loop ────────────────────────────────────────────────────
T_PIPELINE_START=$(date +%s)

for i in "${!CONV_IDS[@]}"; do
  CONV="${CONV_IDS[$i]}"
  QA_JSON="${QA_JSONS[$i]}"
  QA_IDX="${CONV#conv}"

  if [[ ! -f "$QA_JSON" ]]; then
    echo "[skip] QA file not found: $QA_JSON"
    continue
  fi

  CONV_RESULTS="$RESULTS_DIR/$CONV"
  mkdir -p "$CONV_RESULTS"

  for STRATEGY in $STRATEGIES; do
    LABEL="${CONV}_${STRATEGY}"
    G_PKL="$RUNS_BASE/$CONV/artifacts/$STRATEGY/graph_network_${CONV}.pkl"
    T_JSON="$RUNS_BASE/$CONV/artifacts/$STRATEGY/${CONV}_tags.json"

    if [[ ! -f "$G_PKL" ]] || [[ ! -f "$T_JSON" ]]; then
      echo "[skip] $LABEL — missing artifacts"
      continue
    fi

    echo ""
    echo "############################################################"
    echo "#  $LABEL — categories ${CATEGORIES}"
    echo "############################################################"

    OUT_FILE="$CONV_RESULTS/qa_${QA_IDX}_eval_${TAG}_${STRATEGY}.json"
    SUM_FILE="$CONV_RESULTS/qa_${QA_IDX}_summary_${TAG}_${STRATEGY}.json"

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
    eval "TIMING_${CONV}_${STRATEGY}=$((T1 - T0))"
    echo "  [$LABEL] QA time: $((T1 - T0))s"
  done
done

# ── Aggregate results across all conversations ──────────────────────
T_PIPELINE_END=$(date +%s)
TOTAL_WALL=$((T_PIPELINE_END - T_PIPELINE_START))

echo ""
echo "=========================================================="
echo "  Aggregating results → $RESULTS_DIR/summary_all.json"
echo "=========================================================="

python3 -c "
import json, os, glob, sys

results_dir = '$RESULTS_DIR'
strategies = '$STRATEGIES'.split()
tag = '$TAG'
conv_ids = '${CONV_IDS[*]}'.split()

aggregate = {}
for strategy in strategies:
    cat_totals = {}  # category -> {correct, total}
    for conv in conv_ids:
        summary_file = os.path.join(results_dir, conv, f'qa_{conv[4:]}_summary_{tag}_{strategy}.json')
        if not os.path.isfile(summary_file):
            continue
        with open(summary_file) as f:
            data = json.load(f)
        summary = data.get('summary', {})
        for cat_key, cat_stats in summary.get('category_stats', {}).items():
            cat = int(cat_key)
            if cat not in cat_totals:
                cat_totals[cat] = {'correct': 0, 'total': 0}
            cat_totals[cat]['correct'] += cat_stats.get('correct', 0)
            cat_totals[cat]['total'] += cat_stats.get('total', 0)

    total_c = sum(v['correct'] for v in cat_totals.values())
    total_t = sum(v['total'] for v in cat_totals.values())
    acc = total_c / total_t if total_t > 0 else 0

    strat_result = {
        'overall': {'correct': total_c, 'total': total_t, 'accuracy': round(acc, 4)},
        'per_category': {}
    }
    cat_names = {1: 'Multi-hop', 2: 'Temporal', 3: 'Single-hop', 4: 'Open-domain'}
    for cat in sorted(cat_totals.keys()):
        cs = cat_totals[cat]
        ca = cs['correct'] / cs['total'] if cs['total'] > 0 else 0
        strat_result['per_category'][cat] = {
            'name': cat_names.get(cat, f'cat{cat}'),
            'correct': cs['correct'],
            'total': cs['total'],
            'accuracy': round(ca, 4)
        }
    aggregate[strategy] = strat_result

    print(f'  {strategy}: {total_c}/{total_t} = {acc:.1%}')
    for cat in sorted(cat_totals.keys()):
        cs = cat_totals[cat]
        ca = cs['correct'] / cs['total'] if cs['total'] > 0 else 0
        print(f'    {cat_names.get(cat, \"cat\" + str(cat))}: {cs[\"correct\"]}/{cs[\"total\"]} = {ca:.1%}')

# Per-conversation breakdown
per_conv = {}
for conv in conv_ids:
    per_conv[conv] = {}
    for strategy in strategies:
        summary_file = os.path.join(results_dir, conv, f'qa_{conv[4:]}_summary_{tag}_{strategy}.json')
        if not os.path.isfile(summary_file):
            continue
        with open(summary_file) as f:
            data = json.load(f)
        per_conv[conv][strategy] = data.get('summary', {})

output = {
    'tag': tag,
    'categories': [1, 3],
    'aggregate': aggregate,
    'per_conversation': per_conv,
    'total_wall_seconds': $TOTAL_WALL,
}
out_path = os.path.join(results_dir, 'summary_all.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'  Saved: {out_path}')
"

# ── Timing summary ──────────────────────────────────────────────────
echo ""
echo "========== Timing Summary (seconds) =========="
for CONV in "${CONV_IDS[@]}"; do
  for STRATEGY in $STRATEGIES; do
    eval "_t=\${TIMING_${CONV}_${STRATEGY}:-?}"
    echo "  ${CONV}_${STRATEGY}: qa=$_t"
  done
done
echo "  total_wall=$TOTAL_WALL"

echo ""
echo "========== Done =========="
echo "Results: $RESULTS_DIR"
echo "Summary: $RESULTS_DIR/summary_all.json"
