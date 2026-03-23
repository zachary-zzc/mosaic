#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
将各实验生成的 tab_*.tex 复制到 Manuscript/generated/，并生成图结构示例表。
先运行 python experiments/run_all.py（或单独跑各实验）再运行本脚本。
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GENERATED_DIR = os.path.join(PROJECT_ROOT, "Manuscript", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

# (experiments 子目录, 源 tex 文件名, 说明用 label)
EXPERIMENT_TABLES = [
    ("01_locomo_benchmark", "tab_locomo.tex", "tab:locomo"),
    ("03_ablation", "tab_ablation_mosaic.tex", "tab:ablation_mosaic"),
]


def main():
    for exp_name, filename, _label in EXPERIMENT_TABLES:
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

    sys.path.insert(0, PROJECT_ROOT)
    from experiments.latex_export import table_graph_structure

    graph_tex = table_graph_structure()
    path = os.path.join(GENERATED_DIR, "tab_graph_structure.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(graph_tex)
    print("Written:", path)

    readme = os.path.join(GENERATED_DIR, "README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "Generated tables. Experiment 2 (scalability) adds tab_scalability*.tex when implemented.\n"
        )
        f.write("Files: " + ", ".join([t[1] for t in EXPERIMENT_TABLES]) + ", tab_graph_structure.tex\n")
    print("Written:", readme)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
