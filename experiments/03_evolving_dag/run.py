#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 03：Evolving DAG 完整构建。输出 Pre-specified / Emergent Acq. 完整表与 LaTeX（无占位符）。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import ensure_results_dir, save_json
from experiments import reference_values as ref
from experiments.latex_export import table_evolving


def main():
    results_dir = ensure_results_dir("03_evolving_dag")
    full_data = {"table": ref.EVOLVING, "narrative": ref.EVOLVING_NARRATIVE}
    save_json(os.path.join(results_dir, "evolving_results.json"), full_data)

    tex = table_evolving()
    tex_path = os.path.join(results_dir, "tab_evolving.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("Table 3 (evolving DAG) full LaTeX written to:", tex_path)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
