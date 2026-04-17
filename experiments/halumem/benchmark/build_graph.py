#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HaluMem Benchmark — graph construction for all users in a JSONL file.

Builds a MOSAIC graph per user and persists the augmented user data
(with ``extracted_memories``) into ``runs/<run_id>/tmp/<uuid>.json``.

Usage (from project root):
  python experiments/halumem/benchmark/build_graph.py --data test.jsonl
  python experiments/halumem/benchmark/build_graph.py --data HaluMem-Medium.jsonl --max-users 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.halumem._utils import (
    DATASET_HALUMEM_DIR,
    build_graph_for_user,
    load_jsonl,
    save_json,
    setup_mosaic_path,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(EXPERIMENT_DIR, "runs")


def main() -> int:
    parser = argparse.ArgumentParser(description="HaluMem — build graphs for benchmark")
    parser.add_argument("--data", default="test.jsonl", help="JSONL filename inside dataset/halumem/data/")
    parser.add_argument("--run-id", default="default", help="Run identifier (subdirectory under runs/)")
    parser.add_argument("--max-users", type=int, default=None, help="Limit users processed")
    args = parser.parse_args()

    data_path = os.path.join(DATASET_HALUMEM_DIR, args.data)
    if not os.path.isfile(data_path):
        print(f"[error] Data file not found: {data_path}")
        return 1

    users = load_jsonl(data_path)
    if args.max_users:
        users = users[: args.max_users]

    save_path = os.path.join(RUNS_DIR, args.run_id)
    os.makedirs(save_path, exist_ok=True)

    setup_mosaic_path()

    print(f"Building graphs for {len(users)} user(s) → {save_path}")
    t0 = time.time()

    for idx, user_data in enumerate(users, 1):
        uuid = user_data.get("uuid", f"user_{idx}")
        print(f"[{idx}/{len(users)}] Building graph for {uuid} …")

        _memory, new_user_data = build_graph_for_user(user_data, save_path)

        tmp_dir = os.path.join(save_path, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        save_json(os.path.join(tmp_dir, f"{uuid}.json"), new_user_data)
        print(f"  ✅ done")

    print(f"\nAll builds done in {time.time() - t0:.1f}s  →  {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
