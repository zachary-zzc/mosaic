#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
将 Manuscript/generated/*.tex 表格插入 manuscript.tex，替换对应 \begin{tabular}...\end{tabular} 块。
先运行 run_all.py 与 export_to_manuscript.py。
"""
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANUSCRIPT = os.path.join(PROJECT_ROOT, "Manuscript", "manuscript.tex")
GENERATED = os.path.join(PROJECT_ROOT, "Manuscript", "generated")

TABLE_MAP = [
    (r"\label{tab:locomo}", "tab_locomo.tex"),
    (r"\label{tab:ablation_mosaic}", "tab_ablation_mosaic.tex"),
    (r"\label{tab:graph_structure}", "tab_graph_structure.tex"),
]


def find_tabular_block(lines, start_idx):
    i = start_idx
    while i < len(lines) and r"\begin{tabular}" not in lines[i]:
        i += 1
    if i >= len(lines):
        return -1, -1
    begin = i
    i += 1
    while i < len(lines) and r"\end{tabular}" not in lines[i]:
        i += 1
    if i >= len(lines):
        return -1, -1
    return begin, i


def main():
    if not os.path.isfile(MANUSCRIPT):
        print("No manuscript at", MANUSCRIPT)
        return 1
    with open(MANUSCRIPT, "r", encoding="utf-8") as f:
        lines = f.readlines()
    changed = False
    for label, tex_name in TABLE_MAP:
        gen = os.path.join(GENERATED, tex_name)
        if not os.path.isfile(gen):
            continue
        label_idx = None
        for i, line in enumerate(lines):
            if label in line:
                label_idx = i
                break
        if label_idx is None:
            continue
        b, e = find_tabular_block(lines, label_idx)
        if b < 0:
            continue
        with open(gen, "r", encoding="utf-8") as f:
            new_tab = f.read().strip().splitlines(keepends=True)
        if new_tab and not new_tab[-1].endswith("\n"):
            new_tab[-1] += "\n"
        lines[b : e + 1] = new_tab
        changed = True
        print("Patched:", label, "<-", tex_name)
    if changed:
        with open(MANUSCRIPT, "w", encoding="utf-8") as f:
            f.writelines(lines)
    else:
        print("No tables patched (labels or generated files missing).")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
