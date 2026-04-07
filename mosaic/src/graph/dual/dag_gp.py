"""
手稿 \(G_P\) 有向无环约束（docs/optimization.md §1、规划 dag_gp）。

在增量添加先决弧 u→v 时做试探插入，若破坏 DAG 则拒绝（与手稿 Evolving 破环策略一致的最小实现）。
"""
from __future__ import annotations

import copy
from typing import Any

import networkx as nx


def trial_add_edge_preserves_dag(G: nx.DiGraph, u: str, v: str) -> bool:
    """若加入 (u→v) 后仍为 DAG 则返回 True（不修改 G）。"""
    if not u or not v or u == v:
        return False
    if G.has_edge(u, v):
        return True
    H: nx.DiGraph = copy.deepcopy(G)
    H.add_edge(u, v)
    return nx.is_directed_acyclic_graph(H)


def add_edge_if_acyclic(G: nx.DiGraph, u: str, v: str, **attr: Any) -> bool:
    """
    向 ``G`` 添加 u→v；若会形成环则 **不添加** 并返回 False。
    若已存在同向边，视为成功（返回 True），并合并边属性。
    """
    if not u or not v or u == v:
        return False
    if G.has_edge(u, v):
        for k, val in attr.items():
            G.edges[u, v][k] = val
        return True
    G.add_edge(u, v, **attr)
    if nx.is_directed_acyclic_graph(G):
        return True
    G.remove_edge(u, v)
    return False


def dag_violation_detail(G: nx.DiGraph) -> str | None:
    if G.number_of_nodes() == 0 or nx.is_directed_acyclic_graph(G):
        return None
    try:
        cyc = nx.find_cycle(G, orientation="original")
        return repr(cyc)
    except Exception as err:
        return str(err)
