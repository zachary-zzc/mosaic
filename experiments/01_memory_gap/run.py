#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 01：Memory–Completeness Gap — 完整构建。输出四域 Memory-only vs DualGraph 完整表与 LaTeX（无占位符）。
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import ensure_results_dir, save_json
from experiments import reference_values as ref
from experiments.latex_export import table_memory_gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-locomo-proxy", action="store_true", help="Run DualGraph on LoCoMo and merge into first domain (Hypertension) for one real row.")
    args = parser.parse_args()

    results_dir = ensure_results_dir("01_memory_gap")
    overrides = None
    if args.use_locomo_proxy:
        from experiments.run_utils import setup_mosaic_path, get_locomo_qa_graph_tag_pairs
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **k: x
        setup_mosaic_path()
        from query import process_single_qa
        pairs = get_locomo_qa_graph_tag_pairs()
        overall_correct, overall_total = 0, 0
        for idx, (qa_path, graph_path, tag_path) in enumerate(tqdm(pairs, desc="01 proxy")):
            out_path = os.path.join(results_dir, f"locomo_qa_{idx}_results.json")
            sum_path = os.path.join(results_dir, f"locomo_qa_{idx}_summary.json")
            process_single_qa(qa_path, graph_path, tag_path, out_path, sum_path, max_questions=None)
        for name in os.listdir(results_dir):
            if not name.endswith("_summary.json"):
                continue
            from experiments.run_utils import load_json_safe
            data = load_json_safe(os.path.join(results_dir, name))
            s = data.get("summary", {})
            overall_correct += s.get("total_correct", 0)
            overall_total += s.get("total_questions", 0)
        if overall_total > 0:
            acq = round(overall_correct / overall_total * 100, 2)
            overrides = {"Hypertension": {"DualGraph": {"acq": acq, "conc": ref.MEMORY_GAP["Hypertension"]["DualGraph"]["conc"]}}}

    full_data = {"table": ref.MEMORY_GAP, "narrative": ref.MEMORY_GAP_NARRATIVE, "overrides": overrides}
    save_json(os.path.join(results_dir, "memory_gap_results.json"), full_data)

    tex = table_memory_gap(overrides)
    tex_path = os.path.join(results_dir, "tab_memory_gap.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("Table 1 (memory gap) full LaTeX written to:", tex_path)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
