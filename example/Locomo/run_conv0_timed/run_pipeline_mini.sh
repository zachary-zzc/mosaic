#!/usr/bin/env bash
# 前台运行 mini 三步骤（便于在 mosaic conda 下立刻看报错）。用法：./run_pipeline_mini.sh
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$RUN_DIR"
mkdir -p log log_mini

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" ]]; then
  echo "[错误] 未找到 conda。"; exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mosaic

REPO="$(cd "$RUN_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO"
PATHS_MINI="$RUN_DIR/paths_mini.json"

echo "========== [1/3] 构图（mini）=========="
python run.py --verbose-log --paths "$PATHS_MINI"

echo "========== [2/3] 单条 query =========="
python -m mosaic query \
  --graph-pkl "$RUN_DIR/artifacts_mini/graph_network_conv0.pkl" \
  --tags-json "$RUN_DIR/artifacts_mini/conv0_tags.json" \
  --method llm \
  --question "What is Caroline's identity?"

echo "========== [3/3] QA 评测 + 打分 =========="
python run_qa_eval.py --paths "$PATHS_MINI" --method llm

echo "完成。结果: $RUN_DIR/results_mini/"
