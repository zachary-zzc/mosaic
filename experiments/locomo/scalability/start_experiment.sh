#!/usr/bin/env bash
# Scalability experiment — measure build time and accuracy vs conversation length.
#
# NOTE: This script is Unix/macOS only. For Windows (or cross-platform),
#       use the Python equivalent instead:
#       python experiments/locomo/scalability/start_experiment.py
#
# Truncates LoCoMo conversations to varying lengths and runs build + QA.
#
# Usage:
#   cd experiments/locomo/scalability && ./start_experiment.sh
#   nohup ./start_experiment.sh > runs/experiment.log 2>&1 &
#   ./start_experiment.sh --conv conv0 --lengths 100 200 400
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

echo "========== Scalability Experiment =========="
echo "  repo: $REPO"
echo "  python: $(which python)"
echo ""

cd "$REPO"
python experiments/locomo/scalability/run.py "$@"
