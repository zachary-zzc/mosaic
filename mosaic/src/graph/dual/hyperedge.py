"""
共现超边 → 实例级星形展开（与 EntityGraph 导出一致，docs/optimization.md A-2/A-3）。

``connections`` 为 ``graph_edge_*.json`` 中单条记录内的列表；
``entity_id`` 规则为 ``{class_id}:{instance_id}``。
"""
from __future__ import annotations

from typing import Any

from src.data.dual_graph import EDGE_LEG_ASSOCIATIVE, EDGE_LEG_PRAGMATIC, normalize_edge_leg


def sorted_entity_ids_from_connections(connections: list[Any]) -> list[str]:
    ids: list[str] = []
    for c in connections or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("class_id", "")
        iid = c.get("instance_id", "")
        if cid is None or iid is None:
            continue
        eid = f"{cid}:{iid}"
        if not eid or eid.endswith(":"):
            continue
        ids.append(eid)
    return sorted(set(ids))


def star_oriented_pairs_from_connections(connections: list[Any]) -> list[tuple[str, str]]:
    """字典序最小 entity 为 hub，向其余节点连有向边 (hub→leaf)，保证可形成 DAG。"""
    ids = sorted_entity_ids_from_connections(connections)
    if len(ids) < 2:
        return []
    hub = ids[0]
    return [(hub, t) for t in ids[1:]]


def unique_directed_star_pairs_p(edge_records: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """所有 E_P 记录星形展开后的有向边集合（去重），应与 ``DiGraph(G_p).edges`` 一致。"""
    out: set[tuple[str, str]] = set()
    for rec in edge_records or []:
        if normalize_edge_leg(rec.get("edge_leg")) != EDGE_LEG_PRAGMATIC:
            continue
        for u, v in star_oriented_pairs_from_connections(rec.get("connections") or []):
            out.add((u, v))
    return out


def unique_undirected_star_pairs_a(edge_records: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """E_A 记录展开后的无向边键 (min,max)，应与 ``Graph(G_a)`` 的规范化边集一致。"""
    out: set[tuple[str, str]] = set()
    for rec in edge_records or []:
        if normalize_edge_leg(rec.get("edge_leg")) != EDGE_LEG_ASSOCIATIVE:
            continue
        for u, v in star_oriented_pairs_from_connections(rec.get("connections") or []):
            a, b = (u, v) if u <= v else (v, u)
            out.add((a, b))
    return out
