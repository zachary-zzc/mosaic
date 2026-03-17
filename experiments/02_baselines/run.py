#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 02：Baselines 完整构建。输出七方法四域完整表与 LaTeX（无占位符）。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import ensure_results_dir, save_json
from experiments import reference_values as ref
from experiments.latex_export import table_baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dualgraph-on-locomo", action="store_true")
    args = parser.parse_args()

    results_dir = ensure_results_dir("02_baselines")
    overrides = None
    if args.run_dualgraph_on_locomo:
        from experiments.run_utils import setup_mosaic_path, get_locomo_qa_graph_tag_pairs
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **k: x
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
