#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1 — QA：对已有 graph.pkl / tags.json 跑 query 与 LLM 评判。

用法（在仓库根目录）：
  python experiments/01_locomo_benchmark/qa_eval.py --single \\
    --qa example/Locomo/qa_0.json \\
    --graph experiments/01_locomo_benchmark/results/locomo_cache/conv_0/graph.pkl \\
    --tags experiments/01_locomo_benchmark/results/locomo_cache/conv_0/tags.json \\
    --out experiments/01_locomo_benchmark/results

  python experiments/01_locomo_benchmark/qa_eval.py --all
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import setup_mosaic_path, get_locomo_example_pairs, ensure_results_dir

EXP = "01_locomo_benchmark"


def main():
    parser = argparse.ArgumentParser(description="对已有图做 QA + 评判")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--single", action="store_true", help="需同时指定 --qa, --graph, --tags")
    g.add_argument("--all", action="store_true", help="对 locomo_cache/conv_* 与 example/Locomo/qa_*.json 逐对跑")
    parser.add_argument("--qa", type=str, help="QA JSON 路径")
    parser.add_argument("--graph", type=str, help="graph.pkl")
    parser.add_argument("--tags", type=str, help="tags.json")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="结果目录，默认 experiments/01_locomo_benchmark/results",
    )
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--method",
        choices=("llm", "hash"),
        default="hash",
        help="检索：hash 或 llm",
    )
    args = parser.parse_args()

    out_dir = args.out
    if out_dir is None:
        out_dir = ensure_results_dir(EXP)
    else:
        out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    setup_mosaic_path()
    from query import process_single_qa

    if args.all:
        pairs = get_locomo_example_pairs()
        if not pairs:
            print("未找到 example/Locomo 下的 conv/qa 对。")
            return 1
        results_dir = ensure_results_dir(EXP)
        cache_dir = os.path.join(results_dir, "locomo_cache")
        for idx, (_conv_path, qa_path) in enumerate(pairs):
            graph_path = os.path.join(cache_dir, f"conv_{idx}", "graph.pkl")
            tag_path = os.path.join(cache_dir, f"conv_{idx}", "tags.json")
            if not os.path.isfile(graph_path) or not os.path.isfile(tag_path):
                print(f"跳过 conv_{idx}: 请先运行 build_graph.py")
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
        print("全部 QA 完成。")
        return 0

    if not args.qa or not args.graph or not args.tags:
        print("--single 时必须指定 --qa, --graph, --tags")
        return 1
    qa_path = os.path.abspath(args.qa)
    graph_path = os.path.abspath(args.graph)
    tag_path = os.path.abspath(args.tags)
    for p, name in [(qa_path, "qa"), (graph_path, "graph"), (tag_path, "tags")]:
        if not os.path.isfile(p):
            print(f"文件不存在 ({name}): {p}")
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
