# -*- coding: utf-8 -*-
r"""从 reference_values 与运行结果生成 LaTeX 表格，供稿件 \input。"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, os.path.dirname(_here))
try:
    from experiments import reference_values as ref
except ImportError:
    import reference_values as ref


def table_locomo(overrides=None):
    """Experiment 1：LoCoMo 主表（B1–B8）。overrides 为整表时逐行覆盖（如 run.py 的 full_table）。"""
    data = ref.LOCOMO_TABLE.copy()
    if overrides:
        for k, v in overrides.items():
            data[k] = v
    order = [
        "Full Context",
        "Recency",
        "Summary",
        "RAG",
        "Mem0",
        "A-mem",
        "Letta",
        "MOSAIC",
    ]
    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Overall} & \textbf{Single-hop}",
        r"  & \textbf{Multi-hop} & \textbf{Temporal} \\",
        r"\midrule",
    ]
    for m in order:
        if m not in data:
            continue
        r = data[m]
        bold = "\\textbf{" if m == "MOSAIC" else ""
        bold_end = "}" if m == "MOSAIC" else ""
        lines.append(
            f"{m} & {bold}{r['overall']}{bold_end} & {bold}{r['single_hop']}{bold_end} & "
            f"{bold}{r['multi_hop']}{bold_end} & {bold}{r['temporal']}{bold_end} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_mosaic_ablation(overrides=None):
    """Experiment 3：MOSAIC 消融表（占位可覆盖）。"""
    rows = ref.MOSAIC_ABLATION if overrides is None else overrides
    lines = [
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        r"\textbf{ID} & \textbf{Condition} & \textbf{Overall Acc.\ (\%)} & $\Delta$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['id']} & {row['label']} & {row['overall']} & {row['delta']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_graph_structure():
    """图结构示例表（可选 Methods）。"""
    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{Setting} & $|V|$ & $|E|$ & Depth $L$ & Communities \\",
        r"\midrule",
    ]
    for name, r in ref.GRAPH_STRUCTURE_EXAMPLE.items():
        lines.append(
            f"{name} & {r['|V|']} & {r['|E|']} & {r['depth_L']} & {r['communities']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)
