#!/usr/bin/env bash
# 全量 conv0（paths.json → example/Locomo/locomo_conv0.json + qa_0.json）后台流水线：
#   1) TF-IDF/hash 构图（run.py --hash）
#   2) 单条 query 冒烟（mosaic query, method=hash，与构图一致）
#   3) 全量 QA 评测（run_qa_eval.py，qa_0.json + LLM 评判 + 统计）
# 日志：控制与 tqdm → log/task_stdout.log；mosaic 详情 → log/mosaic_server.log；评测 → log/qa_eval.log
# 结束后自动生成评测报告：log/run_report_full.md（构图/query 耗时、准确率、按类统计等）
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
: > log/task_stdout.log

nohup bash -lc "
  set -euo pipefail
  source \"$CONDA_BASE/etc/profile.d/conda.sh\"
  conda activate mosaic
  cd \"$RUN_DIR\"
  export PYTHONPATH=\"$REPO\"

  T_PIPELINE_START=\$(date +%s)

  echo \"========== [1/3] 构图（hash 启发式，全量: locomo_conv0.json）==========\"
  T0=\$(date +%s)
  python run.py --verbose-log --hash --paths \"$PATHS_FULL\"
  T1=\$(date +%s)
  GRAPH_WALL=\$((T1 - T0))

  echo \"========== [2/3] 单条查询冒烟（mosaic query, method=hash）==========\"
  T0=\$(date +%s)
  python -m mosaic query \\
    --graph-pkl \"$RUN_DIR/artifacts/graph_network_conv0.pkl\" \\
    --tags-json \"$RUN_DIR/artifacts/conv0_tags.json\" \\
    --method hash \\
    --question \"What is Caroline's identity?\"
  T1=\$(date +%s)
  SMOKE_WALL=\$((T1 - T0))

  echo \"========== [3/3] QA 评测（qa_0.json：逐题检索+作答 + LLM 评判 + 统计）==========\"
  T0=\$(date +%s)
  python run_qa_eval.py --paths \"$PATHS_FULL\" --method hash
  T1=\$(date +%s)
  QA_WALL=\$((T1 - T0))

  T_PIPELINE_END=\$(date +%s)
  TOTAL_WALL=\$((T_PIPELINE_END - T_PIPELINE_START))

  python \"$RUN_DIR/write_run_report.py\" --mode full --run-dir \"$RUN_DIR\" \\
    --wall-graph \"\$GRAPH_WALL\" --wall-smoke \"\$SMOKE_WALL\" \\
    --wall-qa-eval \"\$QA_WALL\" --wall-total \"\$TOTAL_WALL\"

  echo \"========== 全部完成 ==========\"
  echo \"图/tags: $RUN_DIR/artifacts/\"
  echo \"评测结果: $RUN_DIR/results/\"
  echo \"日志: $RUN_DIR/log/\"
  echo \"Markdown 报告: $RUN_DIR/log/run_report_full.md\"
" >> log/task_stdout.log 2>&1 &

echo $! > log/run.pid
echo "[任务] 已在后台启动全量 conv0 流水线（1 构图 → 2 单条 query → 3 QA 评测+打分）"
echo "  PID: $(cat log/run.pid)"
echo "  流水线 stdout: $RUN_DIR/log/task_stdout.log  （tail -f）"
echo "  详细参数:      $RUN_DIR/log/run_verbose.log  （若已生成）"
echo "  构图日志:      $RUN_DIR/log/mosaic_server.log"
echo "  评测日志:      $RUN_DIR/log/qa_eval.log"
echo "  产物:          $RUN_DIR/artifacts/  与  $RUN_DIR/results/"
echo "  完成后报告:    $RUN_DIR/log/run_report_full.md"
