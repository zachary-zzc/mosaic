#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HaluMem Scalability Experiment — Measure build time and accuracy as user count increases.

Usage:
  python experiments/halumem/scalability/run.py --data HaluMem-Medium.jsonl
"""
import os
import sys
import argparse
from experiments.halumem._utils import setup_mosaic_path, load_jsonl, save_json, DATASET_HALUMEM_DIR

setup_mosaic_path()

def main():
    parser = argparse.ArgumentParser(description="HaluMem Scalability Experiment")
    parser.add_argument("--data", type=str, default="HaluMem-Medium.jsonl")
    parser.add_argument("--out-dir", type=str, default="results/scalability")
    parser.add_argument("--user-steps", type=int, nargs="*", default=[1,5,10,20])
    args = parser.parse_args()
    data_path = os.path.join(DATASET_HALUMEM_DIR, args.data)
    users = load_jsonl(data_path)
    print(f"Scalability: {len(users)} users available.")
    for n in args.user_steps:
        subset = users[:n]
        print(f"[step] Running on {n} users...")
        # TODO: Implement build/eval logic for subset
        print("[placeholder] Not implemented.")

if __name__ == "__main__":
    main()
