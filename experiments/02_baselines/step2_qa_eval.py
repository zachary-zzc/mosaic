#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤 2：仅 query + 打分。读取已构建的 graph.pkl 与 tags.json，对 qa JSON 逐题 query（hash 检索 + LLM 作答），
再用 Qwen 评判对错，写出结果与汇总。不涉及构图。

用法（在仓库根目录）：
  python experiments/02_baselines/step2_qa_eval.py \\
    --qa example/Locomo/qa_0.json \\
    --graph experiments/02_baselines/results/locomo_cache/conv_0/graph.pkl \\
    --tags experiments/02_baselines/results/locomo_cache/conv_0/tags.json \\
    --out experiments/02_baselines/results

  结果写入 --out/dualgraph_qa_0_results.json（逐题完整）与 dualgraph_qa_0_summary.json（分类与整体统计）。
  若使用 --all，则对 locomo_cache 下每个 conv_* 与对应的 qa_*.json 跑一遍 QA。

  与 example/Locomo/run_conv0_timed 一致：检索方式可用 --method llm（默认）或 hash；评判始终用 mosaic 同 API 的 LLM。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import setup_mosaic_path, get_locomo_example_pairs, load_json_safe
from experiments.run_utils import ensure_results_dir


def main():
    parser = argparse.ArgumentParser(description="步骤2：对已有图做 QA + Qwen 打分")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--single",
        action="store_true",
        help="单次运行：需同时指定 --qa, --graph, --tags",
    )
    g.add_argument(
        "--all",
        action="store_true",
        help="对 results/locomo_cache/conv_* 与 example/Locomo/qa_*.json 逐对跑 QA",
    )
    parser.add_argument("--qa", type=str, help="QA 文件路径，如 example/Locomo/qa_0.json")
    parser.add_argument("--graph", type=str, help="图文件路径，如 .../locomo_cache/conv_0/graph.pkl")
    parser.add_argument("--tags", type=str, help="tags 文件路径，如 .../locomo_cache/conv_0/tags.json")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="结果输出目录，默认 experiments/02_baselines/results",
    )
    parser.add_argument("--max-questions", type=int, default=None, help="最多评估题目数，不指定则全部")
    parser.add_argument(
        "--method",
        choices=("llm", "hash"),
        default="hash",
        help="检索方式：hash=TF-IDF（与 step1 hash 构图一致，默认）；llm=类感知+实例 LLM 检索。评判均为 LLM。",
    )
    args = parser.parse_args()

    out_dir = args.out
    if out_dir is None:
        out_dir = ensure_results_dir("02_baselines")
    else:
        out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    setup_mosaic_path()
    from query import process_single_qa

    if args.all:
        pairs = get_locomo_example_pairs()
        if not pairs:
            print("未找到 example/Locomo 下的 (locomo_conv*.json, qa_*.json) 对。")
            return 1
        results_dir = ensure_results_dir("02_baselines")
        cache_dir = os.path.join(results_dir, "locomo_cache")
        for idx, (_conv_path, qa_path) in enumerate(pairs):
            graph_path = os.path.join(cache_dir, f"conv_{idx}", "graph.pkl")
            tag_path = os.path.join(cache_dir, f"conv_{idx}", "tags.json")
            if not os.path.isfile(graph_path) or not os.path.isfile(tag_path):
                print(f"跳过 conv_{idx}: 未找到 {graph_path} 或 {tag_path}，请先运行 step1_build_graph.py")
                continue
            out_path = os.path.join(out_dir, f"dualgraph_qa_{idx}_results.json")
            sum_path = os.path.join(out_dir, f"dualgraph_qa_{idx}_summary.json")
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

    # 单次
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
    # 输出文件名用 qa 的 basename 无后缀，如 qa_0
    base = os.path.splitext(os.path.basename(qa_path))[0]
    out_path = os.path.join(out_dir, f"dualgraph_{base}_results.json")
    sum_path = os.path.join(out_dir, f"dualgraph_{base}_summary.json")
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
