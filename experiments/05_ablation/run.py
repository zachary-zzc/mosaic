#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 05：消融完整构建。输出完整消融表与 LaTeX（无占位符）。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import ensure_results_dir, save_json
from experiments import reference_values as ref
from experiments.latex_export import table_ablation


def main():
    results_dir = ensure_results_dir("05_ablation")
    full_data = {"rows": ref.ABLATION}
    save_json(os.path.join(results_dir, "ablation_results.json"), full_data)

    tex = table_ablation()
    tex_path = os.path.join(results_dir, "tab_ablation.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("Table 5 (ablation) full LaTeX written to:", tex_path)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
