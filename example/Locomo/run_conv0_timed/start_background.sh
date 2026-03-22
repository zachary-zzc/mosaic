#!/usr/bin/env bash
# 使用极小数据集（data_mini）后台跑通：1) TF-IDF/hash 构图 2) 单条 query 冒烟（hash）3) 全量 QA 评测（作答+LLM 评判+统计）
# 日志：控制信息 -> log/task_stdout.log；mosaic 详情 -> log_mini/mosaic_server.log、log_mini/qa_eval.log
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$RUN_DIR"
mkdir -p log log_mini

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" ]]; then
  echo "[错误] 未找到 conda。"
  exit 1
fi

REPO="$(cd "$RUN_DIR/../../.." && pwd)"
PATHS_MINI="$RUN_DIR/paths_mini.json"
: > log/task_stdout.log

nohup bash -lc "
  set -euo pipefail
  source \"$CONDA_BASE/etc/profile.d/conda.sh\"
  conda activate mosaic
  cd \"$RUN_DIR\"
  export PYTHONPATH=\"$REPO\"

  echo \"========== [1/3] 构图（hash 启发式，mini: data_mini/locomo_conv0_mini.json）==========\"
  python run.py --verbose-log --hash --paths \"$PATHS_MINI\"

  echo \"========== [2/3] 单条查询冒烟（mosaic query, method=hash）==========\"
  python -m mosaic query \\
    --graph-pkl \"$RUN_DIR/artifacts_mini/graph_network_conv0.pkl\" \\
    --tags-json \"$RUN_DIR/artifacts_mini/conv0_tags.json\" \\
    --method hash \\
    --question \"What is Caroline's identity?\"

  echo \"========== [3/3] QA 评测：逐题检索+作答 + LLM 评判 + 分类/整体统计 ==========\"
  python run_qa_eval.py --paths \"$PATHS_MINI\" --method hash

  echo \"========== 全部完成 ==========\"
  echo \"图/tags: $RUN_DIR/artifacts_mini/\"
  echo \"评测结果: $RUN_DIR/results_mini/\"
  echo \"日志: $RUN_DIR/log_mini/\"
" >> log/task_stdout.log 2>&1 &

echo $! > log/run.pid
echo "[任务] 已在后台启动 mini 流水线（1 构图 → 2 单条 query → 3 QA 评测+打分）"
echo "  PID: $(cat log/run.pid)"
echo "  流水线 stdout: $RUN_DIR/log/task_stdout.log  （tail -f）"
echo "  详细参数:      $RUN_DIR/log_mini/run_verbose.log  （若已生成）"
echo "  构图日志:      $RUN_DIR/log_mini/mosaic_server.log"
echo "  评测日志:      $RUN_DIR/log_mini/qa_eval.log"
echo "  产物:          $RUN_DIR/artifacts_mini/  与  $RUN_DIR/results_mini/"
