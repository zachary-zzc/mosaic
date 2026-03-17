#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 06：图构建完整构建。输出四域 Edge F1 与下游 Acq. 完整表与 LaTeX（无占位符）。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import ensure_results_dir, save_json
from experiments import reference_values as ref
from experiments.latex_export import table_graph_construction


def main():
    results_dir = ensure_results_dir("06_graph_construction")
    full_data = {"table": ref.GRAPH_CONSTRUCTION, "narrative": ref.GRAPH_CONSTRUCTION_NARRATIVE}
    save_json(os.path.join(results_dir, "graph_construction_results.json"), full_data)

    tex = table_graph_construction()
    tex_path = os.path.join(results_dir, "tab_graph_construction.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("Table 6 (graph construction) full LaTeX written to:", tex_path)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
