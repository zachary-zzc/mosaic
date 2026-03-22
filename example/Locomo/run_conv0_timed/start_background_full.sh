#!/usr/bin/env bash
# 仅全量 conv0 构图（paths.json，TF-IDF/hash 启发式），不跑 QA。
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$RUN_DIR"
mkdir -p log

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" ]]; then
  echo "[错误] 未找到 conda。"
  exit 1
fi

: > log/task_stdout.log
nohup bash -lc "
  source \"$CONDA_BASE/etc/profile.d/conda.sh\"
  conda activate mosaic
  cd \"$RUN_DIR\"
  python run.py --verbose-log --hash
" >> log/task_stdout.log 2>&1 &

echo $! > log/run.pid
echo "[任务] 已在后台启动全量 conv0 构图（仅 run.py，paths.json）"
echo "  PID: $(cat log/run.pid)"
echo "  输出: $RUN_DIR/log/task_stdout.log"
