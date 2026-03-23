#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 3 — MOSAIC 消融（C0–C6）。

C0 与 Experiment 1 中 MOSAIC（B8）一致，可复用日志无需重跑。
各消融需在 MOSAIC 代码路径中开关对应组件后，在 LoCoMo 上重跑 ingest+QA。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.latex_export import table_mosaic_ablation
from experiments.run_utils import ensure_results_dir


def main():
    results_dir = ensure_results_dir("03_ablation")
    tex_path = os.path.join(results_dir, "tab_ablation_mosaic.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(table_mosaic_ablation())
    print("占位表已写入（来自 reference_values.MOSAIC_ABLATION）:", tex_path)
    print("真实消融请按 experiments/README.md Experiment 3 在 LoCoMo 上重跑并覆盖结果。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
