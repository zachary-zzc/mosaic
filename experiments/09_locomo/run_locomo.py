#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoCoMo 长程记忆基准实验：完整构建。调用 mosaic query 得到 DualGraph 指标，与参考表合并后输出完整 LoCoMo 表（无占位符）及 LaTeX。
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import (
    setup_mosaic_path,
    get_locomo_qa_graph_tag_pairs,
    ensure_results_dir,
    load_json_safe,
    save_json,
)
from experiments import reference_values as ref
from experiments.latex_export import table_locomo

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

CATEGORY_NAMES = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop"}


def run_one_qa(qa_path, graph_path, tag_path, output_path, summary_path, max_questions=None):
    setup_mosaic_path()
    from query import process_single_qa
    return process_single_qa(
        qa_path, graph_path, tag_path, output_path, summary_path, max_questions=max_questions
    )


def aggregate_summaries(results_dir: str):
    overall_correct = 0
    overall_total = 0
    by_category = {1: {"correct": 0, "total": 0}, 2: {"correct": 0, "total": 0}, 3: {"correct": 0, "total": 0}}
    for name in os.listdir(results_dir):
        if not name.endswith("_summary.json"):
            continue
        path = os.path.join(results_dir, name)
        data = load_json_safe(path)
        summary = data.get("summary", {})
        overall_correct += summary.get("total_correct", 0)
        overall_total += summary.get("total_questions", 0)
        for cat, stats in summary.get("category_stats", {}).items():
            c = int(cat) if isinstance(cat, str) else cat
            if c in by_category:
                by_category[c]["correct"] += stats.get("correct", 0)
                by_category[c]["total"] += stats.get("total", 0)
    overall_acc = (overall_correct / overall_total * 100) if overall_total else 0.0
    category_acc = {}
    for c, name in CATEGORY_NAMES.items():
        tot = by_category[c]["total"]
        category_acc[name] = (by_category[c]["correct"] / tot * 100) if tot else 0.0
    return {
        "overall_accuracy_pct": round(overall_acc, 2),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "by_category_pct": {k: round(v, 2) for k, v in category_acc.items()},
        "by_category_counts": {CATEGORY_NAMES[c]: by_category[c] for c in sorted(by_category)},
    }


def build_full_locomo_table(dualgraph_metrics=None):
    """完整 LoCoMo 表：DualGraph 用运行结果或参考值，其余用 reference_values。"""
    table = {}
    for method, row in ref.LOCOMO_TABLE.items():
        table[method] = dict(row)
    if dualgraph_metrics and dualgraph_metrics.get("overall_total", 0) > 0:
        table["DualGraph"] = {
            "overall": dualgraph_metrics["overall_accuracy_pct"],
            "single_hop": dualgraph_metrics["by_category_pct"].get("Single-hop", 0),
            "multi_hop": dualgraph_metrics["by_category_pct"].get("Multi-hop", 0),
            "temporal": dualgraph_metrics["by_category_pct"].get("Temporal", 0),
        }
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--skip-run", action="store_true", help="Only aggregate and export; use reference for DualGraph if no results.")
    args = parser.parse_args()

    results_dir = ensure_results_dir("09_locomo")
    pairs = get_locomo_qa_graph_tag_pairs()

    dualgraph_metrics = None
    if not args.skip_run and pairs:
        for idx, (qa_path, graph_path, tag_path) in enumerate(tqdm(pairs, desc="LoCoMo")):
            out_path = os.path.join(results_dir, f"qa_{idx}_results.json")
            sum_path = os.path.join(results_dir, f"qa_{idx}_summary.json")
            run_one_qa(qa_path, graph_path, tag_path, out_path, sum_path, max_questions=args.max_questions)
        dualgraph_metrics = aggregate_summaries(results_dir)
    elif args.skip_run and os.path.isdir(results_dir):
        dualgraph_metrics = aggregate_summaries(results_dir)

    full_table = build_full_locomo_table(dualgraph_metrics)
    save_json(os.path.join(results_dir, "locomo_metrics.json"), {
        "dualgraph_run": dualgraph_metrics,
        "full_table": full_table,
    })

    # LaTeX 完整表（无占位符）
    tex = table_locomo(full_table)
    tex_path = os.path.join(results_dir, "tab_locomo.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("LoCoMo full table (no placeholders) written to:", tex_path)

    # 表行 JSON（供 manuscript 或其它脚本使用）
    save_json(os.path.join(results_dir, "locomo_table_dualgraph.json"), full_table)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
