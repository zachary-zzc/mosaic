#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤 1：仅构图。读取 conv JSON，用 hash 方式构建图并生成 tags，写入指定目录。
不跑 QA、不调打分。构图耗时较长，可单独在后台执行。

用法（在仓库根目录）：
  # 单个 conv
  python experiments/02_baselines/step1_build_graph.py \\
    --conv example/Locomo/locomo_conv0.json \\
    --out experiments/02_baselines/results/locomo_cache/conv_0

  # 对 example/Locomo 下所有 locomo_conv*.json 构图（按索引写入 conv_0, conv_1, ...）
  python experiments/02_baselines/step1_build_graph.py --all
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.run_utils import setup_mosaic_path, get_locomo_example_pairs, build_locomo_graph_hash
from experiments.run_utils import ensure_results_dir


def main():
    parser = argparse.ArgumentParser(description="步骤1：从 conv JSON 构图（hash），输出 graph.pkl + tags.json")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--conv", type=str, help="单个 conv 文件路径，如 example/Locomo/locomo_conv0.json")
    g.add_argument("--all", action="store_true", help="对 example/Locomo 下所有 locomo_conv*.json 依次构图")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出目录，写入 graph.pkl 与 tags.json；--all 时忽略，使用 results/locomo_cache/conv_0 等",
    )
    args = parser.parse_args()

    setup_mosaic_path()

    if args.all:
        pairs = get_locomo_example_pairs()
        if not pairs:
            print("未找到 example/Locomo 下的 (locomo_conv*.json, qa_*.json) 对。")
            return 1
        results_dir = ensure_results_dir("02_baselines")
        cache_dir = os.path.join(results_dir, "locomo_cache")
        for idx, (conv_path, _qa_path) in enumerate(pairs):
            out_dir = os.path.join(cache_dir, f"conv_{idx}")
            print(f"[{idx+1}/{len(pairs)}] 构图: {os.path.basename(conv_path)} -> {out_dir}")
            build_locomo_graph_hash(conv_path, out_dir)
        print("全部构图完成。")
        return 0

    # 单文件
    conv_path = os.path.abspath(args.conv)
    if not os.path.isfile(conv_path):
        print(f"文件不存在: {conv_path}")
        return 1
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(
        ensure_results_dir("02_baselines"), "locomo_cache", "conv_0"
    )
    print(f"构图: {conv_path} -> {out_dir}")
    build_locomo_graph_hash(conv_path, out_dir)
    print("构图完成:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
