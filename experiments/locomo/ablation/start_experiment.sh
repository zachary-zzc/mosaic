#!/usr/bin/env bash
# Ablation experiment — run all ablation conditions on LoCoMo.
#
# NOTE: This script is Unix/macOS only. For Windows (or cross-platform),
#       use the Python equivalent instead:
#       python experiments/locomo/ablation/start_experiment.py
#
# Each condition disables one MOSAIC component and re-evaluates on LoCoMo.
# C0/C1 reuse benchmark results; C2-C6 run with env var overrides.
#
# Usage:
#   cd experiments/locomo/ablation && ./start_experiment.sh
#   nohup ./start_experiment.sh > runs/experiment.log 2>&1 &
#
# Prerequisites: benchmark experiment must have completed first (C0/C1 reuse).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Conda / env setup ───────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" ]]; then
  echo "[error] conda not found."
  exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mosaic

export PYTHONPATH="$REPO"
mkdir -p "$SCRIPT_DIR/runs"

echo "========== Ablation Experiment =========="
echo "  repo: $REPO"
echo "  python: $(which python)"
echo ""

cd "$REPO"
python experiments/locomo/ablation/run.py "$@"
