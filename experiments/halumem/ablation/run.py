#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HaluMem Ablation Experiment — MOSAIC component ablation (C0–C6) on HaluMem.

Usage:
  python experiments/halumem/ablation/run.py --data HaluMem-Medium.jsonl
"""
import os
import sys
import argparse
from experiments.halumem._utils import setup_mosaic_path, load_jsonl, save_json, DATASET_HALUMEM_DIR

setup_mosaic_path()

# Placeholder: define ablation conditions and logic
ABLATION_CONDITIONS = {
    "C0": {"label": "Full MOSAIC"},
    "C1": {"label": "Static graph (hash_only)"},
    # Add more as needed
}

def main():
    parser = argparse.ArgumentParser(description="HaluMem Ablation Experiment")
    parser.add_argument("--data", type=str, default="HaluMem-Medium.jsonl")
    parser.add_argument("--condition", type=str, default="C0")
    parser.add_argument("--out-dir", type=str, default="results/ablation")
    args = parser.parse_args()
    data_path = os.path.join(DATASET_HALUMEM_DIR, args.data)
    users = load_jsonl(data_path)
    print(f"Running ablation {args.condition} on {len(users)} users...")
    # TODO: Implement ablation logic
    print("[placeholder] Not implemented.")

if __name__ == "__main__":
    main()
