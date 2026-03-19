#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 02：Baselines 完整构建。输出七方法四域完整表与 LaTeX（无占位符）。
支持 --run-dualgraph-on-locomo：从 example/Locomo 的 conv 用 hash 构图，对 qa 做 query（hash 检索 + Qwen 作答与评判）。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import ensure_results_dir, save_json
from experiments import reference_values as ref
from experiments.latex_export import table_baselines

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def _aggregate_locomo_summaries(results_dir):
    """仅汇总 results_dir 下 dualgraph_qa_*_summary.json，返回 overrides 或 None。"""
    from experiments.run_utils import load_json_safe
    overall_correct, overall_total = 0, 0
    for name in os.listdir(results_dir):
        if not name.endswith("_summary.json") or "dualgraph" not in name:
            continue
        data = load_json_safe(os.path.join(results_dir, name))
        s = data.get("summary", {})
        overall_correct += s.get("total_correct", 0)
        overall_total += s.get("total_questions", 0)
    if overall_total > 0:
        acq = round(overall_correct / overall_total * 100, 2)
        dg = dict(ref.BASELINES["DualGraph"])
        dg["Hypertension"] = (acq, dg["Hypertension"][1], dg["Hypertension"][2])
        return {"DualGraph": dg}
    return None


def _run_dualgraph_on_locomo_example(results_dir, aggregate_only=False):
    """从 example/Locomo 的 locomo_conv*.json 用 hash 构图，qa_*.json 做 QA，Qwen 评测。
    aggregate_only=True 时只汇总已有 summary，不构图、不跑 QA。"""
    from experiments.run_utils import (
        setup_mosaic_path,
        get_locomo_example_pairs,
        build_locomo_graph_hash,
        load_json_safe,
    )
    if aggregate_only:
        overrides = _aggregate_locomo_summaries(results_dir)
        if overrides:
            print("已汇总 dualgraph_qa_*_summary.json -> DualGraph 行")
        return overrides

    setup_mosaic_path()
    pairs = get_locomo_example_pairs()
    if not pairs:
        print("未找到 example/Locomo 下的 (locomo_conv*.json, qa_*.json) 对，跳过 DualGraph LoCoMo。")
        return None

    locomo_cache_dir = os.path.join(results_dir, "locomo_cache")
    os.makedirs(locomo_cache_dir, exist_ok=True)
    overall_correct, overall_total = 0, 0

    for idx, (conv_path, qa_path) in enumerate(tqdm(pairs, desc="02 DualGraph (example/Locomo)")):
        cache_sub = os.path.join(locomo_cache_dir, f"conv_{idx}")
        graph_path = os.path.join(cache_sub, "graph.pkl")
        tag_path = os.path.join(cache_sub, "tags.json")
        if not os.path.isfile(graph_path) or not os.path.isfile(tag_path):
            print(f"构图: {os.path.basename(conv_path)} -> {cache_sub}")
            build_locomo_graph_hash(conv_path, cache_sub)

        from query import process_single_qa
        out_path = os.path.join(results_dir, f"dualgraph_qa_{idx}_results.json")
        sum_path = os.path.join(results_dir, f"dualgraph_qa_{idx}_summary.json")
        process_single_qa(
            qa_path,
            graph_path,
            tag_path,
            out_path,
            sum_path,
            max_questions=None,
        )

    return _aggregate_locomo_summaries(results_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dualgraph-on-locomo",
        action="store_true",
        help="从 example/Locomo 的 conv 用 hash 构图并跑 QA（Qwen 评测）",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="仅汇总 results 下已有 dualgraph_qa_*_summary.json 写表（与 --run-dualgraph-on-locomo 同用）",
    )
    args = parser.parse_args()

    results_dir = ensure_results_dir("02_baselines")
    overrides = None

    if args.run_dualgraph_on_locomo:
        overrides = _run_dualgraph_on_locomo_example(results_dir, aggregate_only=args.aggregate_only)
    else:
        # 原有逻辑：使用 mosaic 下 locomo results 的 qa/graph/tags 对
        from experiments.run_utils import get_locomo_qa_graph_tag_pairs
        if get_locomo_qa_graph_tag_pairs():
            from experiments.run_utils import setup_mosaic_path
            setup_mosaic_path()
            from query import process_single_qa
            pairs = get_locomo_qa_graph_tag_pairs()
            overall_correct, overall_total = 0, 0
            for idx, (qa_path, graph_path, tag_path) in enumerate(tqdm(pairs, desc="02 DualGraph")):
                out_path = os.path.join(results_dir, f"dualgraph_qa_{idx}_results.json")
                sum_path = os.path.join(results_dir, f"dualgraph_qa_{idx}_summary.json")
                process_single_qa(qa_path, graph_path, tag_path, out_path, sum_path, max_questions=None)
            from experiments.run_utils import load_json_safe
            for name in os.listdir(results_dir):
                if not name.endswith("_summary.json") or "dualgraph" not in name:
                    continue
                data = load_json_safe(os.path.join(results_dir, name))
                s = data.get("summary", {})
                overall_correct += s.get("total_correct", 0)
                overall_total += s.get("total_questions", 0)
            if overall_total > 0:
                acq = round(overall_correct / overall_total * 100, 2)
                dg = dict(ref.BASELINES["DualGraph"])
                dg["Hypertension"] = (acq, dg["Hypertension"][1], dg["Hypertension"][2])
                overrides = {"DualGraph": dg}

    full_data = {"table": ref.BASELINES, "overrides": overrides}
    save_json(os.path.join(results_dir, "baselines_results.json"), full_data)

    tex = table_baselines(overrides)
    tex_path = os.path.join(results_dir, "tab_baselines.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("Table 2 (baselines) full LaTeX written to:", tex_path)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
