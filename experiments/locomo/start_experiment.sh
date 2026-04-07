#!/usr/bin/env bash
# LoCoMo dataset — run all sub-experiments in sequence.
#
# Runs: benchmark → ablation → scalability, then exports LaTeX tables.
#
# Usage:
#   cd experiments/locomo && ./start_experiment.sh
#   nohup ./start_experiment.sh > experiment.log 2>&1 &
#
# Or run individual sub-experiments:
#   ./benchmark/start_experiment.sh
#   ./ablation/start_experiment.sh
#   ./scalability/start_experiment.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Conda / env setup ───────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" ]]; then
  echo "[error] conda not found."
  exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mosaic

export PYTHONPATH="$REPO"
cd "$REPO"

T_START=$(date +%s)

echo "============================================================"
echo "  LoCoMo Experiments — Full Pipeline"
echo "  $(date)"
echo "============================================================"
echo ""

# ── 1. Benchmark ────────────────────────────────────────────────────
echo ">>> [1/4] Benchmark (hybrid + hash_only)"
python experiments/locomo/benchmark/run.py "$@"
echo ""

# ── 2. Ablation ─────────────────────────────────────────────────────
echo ">>> [2/4] Ablation (C0-C6)"
python experiments/locomo/ablation/run.py
echo ""

# ── 3. Scalability ──────────────────────────────────────────────────
echo ">>> [3/4] Scalability"
python experiments/locomo/scalability/run.py
echo ""

# ── 4. Export tables ────────────────────────────────────────────────
echo ">>> [4/4] Export tables to Manuscript/generated/"
python experiments/export_to_manuscript.py
echo ""

T_END=$(date +%s)
TOTAL=$((T_END - T_START))
echo "============================================================"
echo "  All LoCoMo experiments complete — ${TOTAL}s total"
echo "============================================================"
