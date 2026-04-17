#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

# HaluMem Medium
python experiments/halumem/benchmark/start_experiment.py \
    --data HaluMem-Medium.jsonl \
    --run-id medium \
    --api-key-file experiments/halumem/benchmark/keys.txt

# HaluMem Long
python experiments/halumem/benchmark/start_experiment.py \
    --data HaluMem-Long.jsonl \
    --run-id long \
    --api-key-file experiments/halumem/benchmark/keys.txt
