# -*- coding: utf-8 -*-
"""
与 experiments/README.md 一致的参考数值：LoCoMo 基线（可引用文献）、消融占位等。
完整跑通后应以运行日志与评测脚本输出为准；此处用于合并稿件表格或缺数据时的占位。
"""

# 文献/论文中的 LoCoMo 总体准确率（对照用，公平对比需在同一协议下重跑）— README「Published Baselines」
PUBLISHED_LOCOMO_OVERALL = {
    "Mem0": 61.43,  # arXiv:2504.19413
    "memg": 60.41,
    "ReadAgent": 54.0,  # 约值，LoCoMo 论文
    "Full Context (GPT-4)": 67.0,  # 约值，LoCoMo 论文
}

# Experiment 1 — LoCoMo 表（B1–B8）。分类列与 mosaic query 汇总一致时为 Single-hop / Temporal / Multi-hop；
# Open-ended、Adversarial 需在评测管线支持后填入。
LOCOMO_TABLE = {
    "Full Context": {"overall": 67.0, "single_hop": 70.0, "multi_hop": 62.0, "temporal": 60.0},
    "Recency": {"overall": 42.0, "single_hop": 45.0, "multi_hop": 38.0, "temporal": 35.0},
    "Summary": {"overall": 48.0, "single_hop": 50.0, "multi_hop": 44.0, "temporal": 42.0},
    "RAG": {"overall": 55.0, "single_hop": 58.0, "multi_hop": 52.0, "temporal": 50.0},
    "Mem0": {"overall": 61.43, "single_hop": 65.20, "multi_hop": 58.50, "temporal": 55.80},
    "A-mem": {"overall": 58.20, "single_hop": 62.50, "multi_hop": 54.10, "temporal": 52.30},
    "Letta": {"overall": 57.85, "single_hop": 61.20, "multi_hop": 55.30, "temporal": 53.10},
    "MOSAIC": {"overall": 80.89, "single_hop": 84.78, "multi_hop": 76.95, "temporal": 74.14},
}

# Experiment 3 — MOSAIC 消融（C0–C6），占位；C0 应与 Experiment 1 中 MOSAIC 一致
MOSAIC_ABLATION = [
    {"id": "C0", "label": "Full MOSAIC", "overall": 80.89, "delta": "---"},
    {"id": "C1", "label": r"$-$ Entity extraction", "overall": 72.0, "delta": r"$-$8.9"},
    {"id": "C2", "label": r"$-$ Relationship edges", "overall": 68.0, "delta": r"$-$12.9"},
    {"id": "C3", "label": r"$-$ Graph traversal", "overall": 70.0, "delta": r"$-$10.9"},
    {"id": "C4", "label": r"$-$ Temporal ordering", "overall": 75.0, "delta": r"$-$5.9"},
    {"id": "C5", "label": r"$-$ Community structure", "overall": 78.0, "delta": r"$-$2.9"},
    {"id": "C6", "label": r"$-$ Deduplication / merging", "overall": 77.0, "delta": r"$-$3.9"},
]

# 可选：Methods 中图规模示例（非 LoCoMo 主表）
GRAPH_STRUCTURE_EXAMPLE = {
    "LoCoMo session (illustrative)": {
        "|V|": 120,
        "|E|": 340,
        "depth_L": 5,
        "communities": 8,
    },
}
