#!/usr/bin/env bash
# 在已有构图产物（artifacts/graph_network_conv0.pkl、conv0_tags.json）上后台执行：
#   1) 单条 query 冒烟（mosaic query, method=llm，与主路径一致）
#   2) 全量 QA 评测（run_qa_eval.py，使用 paths.json 中的 qa_0.json）
# 适用：已单独跑完 run.py / start_background_full.sh 仅构图阶段，或图已就绪需重跑评测。
# 日志：log/query_eval_task_stdout.log；mosaic → log/mosaic_server.log；评测详情 → log/qa_eval.log
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$RUN_DIR"
mkdir -p log

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" ]]; then
  echo "[错误] 未找到 conda。"
  exit 1
fi

REPO="$(cd "$RUN_DIR/../../.." && pwd)"
PATHS_FULL="$RUN_DIR/paths.json"
OUT_LOG="$RUN_DIR/log/query_eval_task_stdout.log"
: > "$OUT_LOG"

nohup bash -lc "
  set -euo pipefail
  source \"$CONDA_BASE/etc/profile.d/conda.sh\"
  conda activate mosaic
  cd \"$RUN_DIR\"
  export PYTHONPATH=\"$REPO\"

  echo \"========== [1/2] 单条查询冒烟（mosaic query, method=llm）==========\"
  python -m mosaic query \\
    --graph-pkl \"$RUN_DIR/artifacts/graph_network_conv0.pkl\" \\
    --tags-json \"$RUN_DIR/artifacts/conv0_tags.json\" \\
    --method llm \\
    --question \"What is Caroline's identity?\"

  echo \"========== [2/2] QA 评测（qa_0.json）==========\"
  python run_qa_eval.py --paths \"$PATHS_FULL\" --method llm

  echo \"========== query + 评测完成 ==========\"
  echo \"评测结果: $RUN_DIR/results/\"
" >> "$OUT_LOG" 2>&1 &

echo $! > log/query_eval.pid
echo "[任务] 已在后台启动 query + QA 评测（依赖已有 artifacts/）"
echo "  PID: $(cat log/query_eval.pid)"
echo "  任务 stdout: $OUT_LOG  （tail -f）"
echo "  评测日志:    $RUN_DIR/log/qa_eval.log"
echo "  结果目录:    $RUN_DIR/results/"
