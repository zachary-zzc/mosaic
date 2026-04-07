#!/usr/bin/env python3
"""
A-3 验收脚本：核对 (1) graph_edge_*.json 与 entity_graph_*.json 边集一致；
(2) 可选 ClassGraph 全量 pickle 上 G_p/G_a 与导出一致。

用法（在 mosaic 目录下，PYTHONPATH=.）::

    python scripts/verify_dual_graph_exports.py \\
        --graph-edge results/graph/graph_edge_conv0_20260322.json \\
        --entity-graph results/graph/entity_graph_conv0_20260322.json

    python scripts/verify_dual_graph_exports.py --classgraph-pkl /path/to/build_state.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys


def _ensure_mosaic_root_on_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> int:
    _ensure_mosaic_root_on_path()

    ap = argparse.ArgumentParser(description="A-3：EntityGraph / graph_edge / ClassGraph nx 一致性")
    ap.add_argument("--graph-edge", help="graph_edge_*.json 路径")
    ap.add_argument("--entity-graph", help="entity_graph_*.json 路径")
    ap.add_argument("--classgraph-pkl", help="ClassGraph 全量 pickle（如构图断点 .build_state.pkl）")
    args = ap.parse_args()

    if not args.classgraph_pkl and (not args.graph_edge or not args.entity_graph):
        ap.error("请提供 --classgraph-pkl，或同时提供 --graph-edge 与 --entity-graph")

    from src.graph.dual.verify_exports import (
        load_json_path,
        verify_classgraph_full,
        verify_entity_json_matches_graph_edge_json,
    )

    ok_all = True
    if args.graph_edge and args.entity_graph:
        eg = load_json_path(args.entity_graph)
        edges = load_json_path(args.graph_edge)
        if not isinstance(edges, list):
            print("错误: graph-edge 文件应为 JSON 数组", file=sys.stderr)
            return 2
        ok, msg = verify_entity_json_matches_graph_edge_json(eg, edges)
        if ok:
            print("OK: entity_graph 与 graph_edge 星形展开边集一致 (E_P / E_A)")
        else:
            ok_all = False
            print(f"FAIL: {msg}", file=sys.stderr)

    if args.classgraph_pkl:
        with open(args.classgraph_pkl, "rb") as f:
            cg = pickle.load(f)
        from src.data.graph import ClassGraph

        if not isinstance(cg, ClassGraph):
            print(f"错误: pickle 非 ClassGraph: {type(cg)}", file=sys.stderr)
            return 2
        ok, errs = verify_classgraph_full(cg)
        if ok:
            print("OK: ClassGraph G_p/G_a 与 edges、EntityGraph 导出、DAG 校验通过")
        else:
            ok_all = False
            for e in errs:
                print(f"FAIL: {e}", file=sys.stderr)

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
