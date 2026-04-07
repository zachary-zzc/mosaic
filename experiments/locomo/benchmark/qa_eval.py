#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoCoMo QA evaluation: run QA + LLM judge on existing graph.pkl / tags.json.

Usage (from project root):
  python experiments/locomo/benchmark/qa_eval.py --single \\
    --qa dataset/locomo/qa_0.json \\
    --graph experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0.pkl \\
    --tags experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/conv0_tags.json \\
    --out experiments/locomo/benchmark/results

  python experiments/locomo/benchmark/qa_eval.py --all
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from experiments.locomo._utils import setup_mosaic_path, get_locomo_example_pairs

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")


def main():
    parser = argparse.ArgumentParser(description="Run QA + LLM judge on existing graph")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--single", action="store_true", help="Requires --qa, --graph, --tags")
    g.add_argument("--all", action="store_true", help="Run on all conv/qa pairs from dataset/locomo/")
    parser.add_argument("--qa", type=str, help="QA JSON path")
    parser.add_argument("--graph", type=str, help="graph.pkl path")
    parser.add_argument("--tags", type=str, help="tags.json path")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Results directory (default: experiments/locomo/results)",
    )
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--method",
        choices=("llm", "hash"),
        default="hash",
        help="Retrieval method: hash or llm",
    )
    args = parser.parse_args()

    out_dir = args.out
    if out_dir is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_dir = RESULTS_DIR
    else:
        out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    setup_mosaic_path()
    from query import process_single_qa

    if args.all:
        pairs = get_locomo_example_pairs()
        if not pairs:
            print("No conv/qa pairs found in dataset/locomo/.")
            return 1
        results_dir = RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        cache_dir = os.path.join(results_dir, "locomo_cache")
        for idx, (_conv_path, qa_path) in enumerate(pairs):
            graph_path = os.path.join(cache_dir, f"conv_{idx}", "graph.pkl")
            tag_path = os.path.join(cache_dir, f"conv_{idx}", "tags.json")
            if not os.path.isfile(graph_path) or not os.path.isfile(tag_path):
                print(f"Skipping conv_{idx}: run build_graph.py first")
                continue
            out_path = os.path.join(out_dir, f"mosaic_qa_{idx}_results.json")
            sum_path = os.path.join(out_dir, f"mosaic_qa_{idx}_summary.json")
            print(f"[{idx+1}/{len(pairs)}] QA: {os.path.basename(qa_path)}")
            process_single_qa(
                qa_path,
                graph_path,
                tag_path,
                out_path,
                sum_path,
                max_questions=args.max_questions,
                method=args.method,
            )
        print("All QA evaluations complete.")
        return 0

    if not args.qa or not args.graph or not args.tags:
        print("--single requires --qa, --graph, --tags")
        return 1
    qa_path = os.path.abspath(args.qa)
    graph_path = os.path.abspath(args.graph)
    tag_path = os.path.abspath(args.tags)
    for p, name in [(qa_path, "qa"), (graph_path, "graph"), (tag_path, "tags")]:
        if not os.path.isfile(p):
            print(f"File not found ({name}): {p}")
            return 1
    base = os.path.splitext(os.path.basename(qa_path))[0]
    out_path = os.path.join(out_dir, f"mosaic_{base}_results.json")
    sum_path = os.path.join(out_dir, f"mosaic_{base}_summary.json")
    process_single_qa(
        qa_path,
        graph_path,
        tag_path,
        out_path,
        sum_path,
        max_questions=args.max_questions,
        method=args.method,
    )
    print("QA 完成:", sum_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
