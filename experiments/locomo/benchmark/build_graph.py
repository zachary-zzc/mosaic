#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoCoMo graph construction: build graph from conv JSON using hash method.

Usage (from project root):
  python experiments/locomo/benchmark/build_graph.py \\
    --conv dataset/locomo/locomo_conv0.json \\
    --out experiments/locomo/benchmark/results/locomo_cache/conv_0

  python experiments/locomo/benchmark/build_graph.py --all
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from experiments.locomo._utils import setup_mosaic_path, get_locomo_example_pairs, build_locomo_graph_hash

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")


def main():
    parser = argparse.ArgumentParser(description="Build graph from conv JSON (hash mode), output graph.pkl + tags.json")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--conv", type=str, help="Single conv, e.g. dataset/locomo/locomo_conv0.json")
    g.add_argument("--all", action="store_true", help="Build graphs for all locomo_conv*.json in dataset/locomo/")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory; --all uses results/locomo_cache/conv_*",
    )
    args = parser.parse_args()

    setup_mosaic_path()

    if args.all:
        pairs = get_locomo_example_pairs()
        if not pairs:
            print("No (locomo_conv*.json, qa_*.json) pairs found in dataset/locomo/.")
            return 1
        cache_dir = os.path.join(RESULTS_DIR, "locomo_cache")
        for idx, (conv_path, _qa_path) in enumerate(pairs):
            out_dir = os.path.join(cache_dir, f"conv_{idx}")
            print(f"[{idx+1}/{len(pairs)}] Building graph: {os.path.basename(conv_path)} -> {out_dir}")
            build_locomo_graph_hash(conv_path, out_dir)
        print("All graphs built.")
        return 0

    conv_path = os.path.abspath(args.conv)
    if not os.path.isfile(conv_path):
        print(f"File not found: {conv_path}")
        return 1
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(
        RESULTS_DIR, "locomo_cache", "conv_0"
    )
    print(f"Building graph: {conv_path} -> {out_dir}")
    build_locomo_graph_hash(conv_path, out_dir)
    print("Graph built:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
