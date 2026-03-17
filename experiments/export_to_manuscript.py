#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
汇总各实验生成的完整表格，写入 Manuscript/generated/ 供稿件 \input 或手工替换。
先运行 python experiments/run_all.py 再运行本脚本。
"""
import os
import sys
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GENERATED_DIR = os.path.join(PROJECT_ROOT, "Manuscript", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

EXPERIMENT_TABLES = [
    ("01_memory_gap", "tab_memory_gap.tex", "tab:memory_gap"),
    ("02_baselines", "tab_baselines.tex", "tab:baselines"),
    ("03_evolving_dag", "tab_evolving.tex", "tab:evolving"),
    ("04_ncs_mechanism", "tab_ncs_validation.tex", "tab:ncs_validation"),
    ("05_ablation", "tab_ablation.tex", "tab:ablation_summary"),
    ("06_graph_construction", "tab_graph_construction.tex", "tab:graph_construction"),
    ("07_downstream", "tab_downstream.tex", "tab:downstream"),
    ("09_locomo", "tab_locomo.tex", "tab:locomo"),
]


def main():
    # 1) 复制各实验的 tab_*.tex 到 Manuscript/generated/
    for exp_name, filename, label in EXPERIMENT_TABLES:
        src = os.path.join(PROJECT_ROOT, "experiments", exp_name, "results", filename)
        if os.path.exists(src):
            dst = os.path.join(GENERATED_DIR, filename)
            with open(src, "r", encoding="utf-8") as f:
                content = f.read()
            with open(dst, "w", encoding="utf-8") as f:
                f.write(content)
            print("Written:", dst)
        else:
            print("Skip (not found):", src)

    # 2) 生成图结构表 (Methods)
    sys.path.insert(0, PROJECT_ROOT)
    from experiments.latex_export import table_graph_structure
    graph_tex = table_graph_structure()
    path = os.path.join(GENERATED_DIR, "tab_graph_structure.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(graph_tex)
    print("Written:", path)

    # 3) 生成汇总说明
    readme = os.path.join(GENERATED_DIR, "README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("Generated tables (no placeholders). Replace corresponding \\begin{table}...\\end{table} in manuscript.tex with \\input{generated/<file>} or paste content.\n")
        f.write("Files: " + ", ".join([t[1] for t in EXPERIMENT_TABLES]) + ", tab_graph_structure.tex\n")
    print("Written:", readme)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
