#!/usr/bin/env bash
# 创建独立环境并运行 02_baselines 的 DualGraph-on-Locomo 实验。
# 在仓库根目录执行: bash experiments/02_baselines/run_locomo_env.sh

set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
VENV="${ROOT}/.venv_baseline"

echo "[1/3] 创建虚拟环境 .venv_baseline ..."
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
echo "[2/3] 安装 mosaic 依赖（可能需要几分钟）..."
"$VENV/bin/pip" install -r mosaic/requirements.txt
echo "[3/3] 运行 DualGraph on example/Locomo（构图 + QA + Qwen 评测）..."
"$VENV/bin/python" experiments/02_baselines/run.py --run-dualgraph-on-locomo
echo "完成。结果见 experiments/02_baselines/results/"
