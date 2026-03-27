#!/usr/bin/env bash
# 使用极小数据集（data_mini）后台跑通：1) hybrid 构图（[BUILD] / MOSAIC_BUILD_MODE，默认 LLM+TF-IDF）2) 单条 query 冒烟（llm）3) QA 评测（作答+LLM 评判+统计）
# 全量 conv0（locomo_conv0.json + 上级目录 qa_0.json）请用同目录的 start_background_full.sh；仅对已构图产物跑 query+评测请用 query_eval_background.sh。
# 日志：控制信息 -> log/task_stdout.log；mosaic 详情 -> log_mini/mosaic_server.log、log_mini/qa_eval.log
# 结束后自动生成评测报告：log/run_report_mini.md（构图/query 耗时、准确率、按类统计等）
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

  T_PIPELINE_START=\$(date +%s)

  echo \"========== [1/3] 构图（hybrid，mini: data_mini/locomo_conv0_mini.json；基线请加 run.py --hash）==========\"
  T0=\$(date +%s)
  python run.py --verbose-log --paths \"$PATHS_MINI\"
  T1=\$(date +%s)
  GRAPH_WALL=\$((T1 - T0))

  echo \"========== [2/3] 单条查询冒烟（mosaic query, method=llm）==========\"
  T0=\$(date +%s)
  python -m mosaic query \\
    --graph-pkl \"$RUN_DIR/artifacts_mini/graph_network_conv0.pkl\" \\
    --tags-json \"$RUN_DIR/artifacts_mini/conv0_tags.json\" \\
    --method llm \\
    --question \"What is Caroline's identity?\"
  T1=\$(date +%s)
  SMOKE_WALL=\$((T1 - T0))

  echo \"========== [3/3] QA 评测：逐题检索+作答 + LLM 评判 + 分类/整体统计 ==========\"
  T0=\$(date +%s)
  python run_qa_eval.py --paths \"$PATHS_MINI\" --method llm
  T1=\$(date +%s)
  QA_WALL=\$((T1 - T0))

  T_PIPELINE_END=\$(date +%s)
  TOTAL_WALL=\$((T_PIPELINE_END - T_PIPELINE_START))

  python \"$RUN_DIR/write_run_report.py\" --mode mini --run-dir \"$RUN_DIR\" \\
    --wall-graph \"\$GRAPH_WALL\" --wall-smoke \"\$SMOKE_WALL\" \\
    --wall-qa-eval \"\$QA_WALL\" --wall-total \"\$TOTAL_WALL\"

  echo \"========== 全部完成 ==========\"
  echo \"图/tags: $RUN_DIR/artifacts_mini/\"
  echo \"评测结果: $RUN_DIR/results_mini/\"
  echo \"日志: $RUN_DIR/log_mini/\"
  echo \"Markdown 报告: $RUN_DIR/log/run_report_mini.md\"
" >> log/task_stdout.log 2>&1 &

echo $! > log/run.pid
echo "[任务] 已在后台启动 mini 流水线（1 构图 → 2 单条 query → 3 QA 评测+打分）"
echo "  PID: $(cat log/run.pid)"
echo "  流水线 stdout: $RUN_DIR/log/task_stdout.log  （tail -f）"
echo "  详细参数:      $RUN_DIR/log_mini/run_verbose.log  （若已生成）"
echo "  构图日志:      $RUN_DIR/log_mini/mosaic_server.log"
echo "  评测日志:      $RUN_DIR/log_mini/qa_eval.log"
echo "  产物:          $RUN_DIR/artifacts_mini/  与  $RUN_DIR/results_mini/"
echo "  完成后报告:    $RUN_DIR/log/run_report_mini.md"
