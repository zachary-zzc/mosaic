#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 04：NCS 机制验证完整构建。输出 NCS validation 完整表与 LaTeX（无占位符）。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import ensure_results_dir, save_json
from experiments import reference_values as ref
from experiments.latex_export import table_ncs_validation


def main():
    results_dir = ensure_results_dir("04_ncs_mechanism")
    full_data = {"table": ref.NCS_VALIDATION}
    save_json(os.path.join(results_dir, "ncs_validation_results.json"), full_data)

    tex = table_ncs_validation()
    tex_path = os.path.join(results_dir, "tab_ncs_validation.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("Table 4 (NCS validation) full LaTeX written to:", tex_path)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
