#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HaluMem Benchmark Experiment — Full pipeline: graph construction + evaluation.
Supports multi-API-key parallelism for both build and eval.

Usage (from project root):
  python experiments/halumem/benchmark/run.py --data HaluMem-Medium.jsonl --api-keys key1,key2,...
  python experiments/halumem/benchmark/run.py --data test.jsonl --api-keys key1,key2
"""
import os
import sys
import json
import time
import argparse
from multiprocessing import Pool
from experiments.halumem._utils import setup_mosaic_path, load_jsonl, save_json, DATASET_HALUMEM_DIR

setup_mosaic_path()

# Placeholder: import your mosaic graph build and eval functions here
# from src.data.graph import ...
# from dataset.halumem.evaluation import main_eval

def parse_api_keys(keys_str):
    return [k.strip() for k in keys_str.split(",") if k.strip()]

def run_single_user(user_data, api_key, out_dir):
    """Run graph build + eval for a single user (placeholder)."""
    # TODO: Replace with actual build/eval logic, passing api_key to LLM client
    user_id = user_data.get("uuid", "unknown")
    result = {"uuid": user_id, "status": "ok", "api_key": api_key}
    # Simulate work
    time.sleep(0.1)
    save_json(os.path.join(out_dir, f"{user_id}.json"), result)
    return result

def main():
    parser = argparse.ArgumentParser(description="HaluMem Benchmark Experiment")
    parser.add_argument("--data", type=str, default="HaluMem-Medium.jsonl", help="Which HaluMem JSONL file to use")
    parser.add_argument("--api-keys", type=str, required=True, help="Comma-separated list of API keys for parallelism")
    parser.add_argument("--out-dir", type=str, default="results/benchmark", help="Output directory")
    parser.add_argument("--max-users", type=int, default=None, help="Limit number of users (for quick test)")
    args = parser.parse_args()

    data_path = os.path.join(DATASET_HALUMEM_DIR, args.data)
    users = load_jsonl(data_path)
    if args.max_users:
        users = users[:args.max_users]
    api_keys = parse_api_keys(args.api_keys)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running HaluMem benchmark on {len(users)} users with {len(api_keys)} API keys...")
    pool_args = [(user, api_keys[i % len(api_keys)], out_dir) for i, user in enumerate(users)]
    with Pool(len(api_keys)) as pool:
        results = list(pool.starmap(run_single_user, pool_args))
    print(f"Done. Results in {out_dir}")

if __name__ == "__main__":
    main()
