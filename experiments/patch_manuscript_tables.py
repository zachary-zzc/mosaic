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

# label -> generated filename (only tabular body)
TABLE_MAP = [
    (r"\label{tab:memory_gap}", "tab_memory_gap.tex"),
    (r"\label{tab:baselines}", "tab_baselines.tex"),
    (r"\label{tab:evolving}", "tab_evolving.tex"),
    (r"\label{tab:ncs_validation}", "tab_ncs_validation.tex"),
    (r"\label{tab:ablation_summary}", "tab_ablation.tex"),
    (r"\label{tab:graph_construction}", "tab_graph_construction.tex"),
    (r"\label{tab:downstream}", "tab_downstream.tex"),
    (r"\label{tab:locomo}", "tab_locomo.tex"),
    (r"\label{tab:graph_structure}", "tab_graph_structure.tex"),
]


def find_tabular_block(lines, start_idx):
    r"""从 start_idx 起找 \begin{tabular} 到对应 \end{tabular}。"""
    i = start_idx
    while i < len(lines) and r"\begin{tabular}" not in lines[i]:
        i += 1
    if i >= len(lines):
        return -1, -1
    begin = i
    # find matching \end{tabular}
    i += 1
    while i < len(lines) and r"\end{tabular}" not in lines[i]:
        i += 1
    if i >= len(lines):
        return begin, -1
    return begin, i


def main():
    if not os.path.exists(MANUSCRIPT):
        print("Not found:", MANUSCRIPT)
        return 1
    with open(MANUSCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")
    # 从后往前替换，避免下标偏移
    for label, filename in reversed(TABLE_MAP):
        path = os.path.join(GENERATED, filename)
        if not os.path.exists(path):
            print("Skip (no file):", path)
            continue
        with open(path, "r", encoding="utf-8") as f:
            replacement = f.read().strip()
        # 找 label 所在行
        label_idx = None
        for i, line in enumerate(lines):
            if label in line:
                label_idx = i
                break
        if label_idx is None:
            print("Label not found:", label)
            continue
        begin, end = find_tabular_block(lines, label_idx + 1)
        if begin < 0 or end < 0:
            print("Tabular block not found for", label)
            continue
        # 替换为 \input{generated/filename}
        input_line = "\\input{generated/" + filename + "}"
        new_lines = lines[:begin] + [input_line] + lines[end + 1:]
        lines = new_lines
        print("Patched:", label, "->", input_line)

    with open(MANUSCRIPT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Written:", MANUSCRIPT)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
