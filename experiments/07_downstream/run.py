#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 07：下游决策完整构建。输出四域下游指标完整数据与 LaTeX（无占位符）。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import ensure_results_dir, save_json
from experiments import reference_values as ref
from experiments.latex_export import table_downstream_hypertension_diabetes


def main():
    results_dir = ensure_results_dir("07_downstream")
    full_data = {"table": ref.DOWNSTREAM}
    save_json(os.path.join(results_dir, "downstream_results.json"), full_data)

    tex = table_downstream_hypertension_diabetes()
    tex_path = os.path.join(results_dir, "tab_downstream.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("Table 7 (downstream) clinical LaTeX written to:", tex_path)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
