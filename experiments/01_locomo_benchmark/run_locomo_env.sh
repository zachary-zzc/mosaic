#!/usr/bin/env bash
# 可选：独立虚拟环境中对 example/Locomo 构图并跑 QA（Experiment 1 子流程）。
# 在仓库根目录执行: bash experiments/01_locomo_benchmark/run_locomo_env.sh

set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
VENV="${ROOT}/.venv_locomo_exp"

echo "[1/3] 创建虚拟环境 .venv_locomo_exp ..."
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
echo "[2/3] 安装 mosaic 依赖..."
"$VENV/bin/pip" install -r mosaic/requirements.txt
echo "[3/3] 构图 + QA（example/Locomo）..."
"$VENV/bin/python" experiments/01_locomo_benchmark/build_graph.py --all
"$VENV/bin/python" experiments/01_locomo_benchmark/qa_eval.py --all
echo "完成。结果见 experiments/01_locomo_benchmark/results/"
