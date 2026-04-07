"""
A-3：EntityGraph JSON、graph_edge 记录与 ClassGraph.G_p/G_a 一致性校验（docs/optimization.md §4 A-3）。
"""
from __future__ import annotations

import json
from typing import Any

import networkx as nx

from src.data.dual_graph import EDGE_LEG_ASSOCIATIVE, EDGE_LEG_PRAGMATIC, normalize_edge_leg
from src.graph.dual.hyperedge import oriented_ep_pairs_from_record, star_oriented_pairs_from_connections


def edge_sets_from_entity_graph_dict(eg: dict[str, Any]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """从 EntityGraph 导出 dict 得到 E_P 有向边集与 E_A 规范无向边键集。"""
    ep: set[tuple[str, str]] = set()
    for e in eg.get("edges_p") or []:
        s, t = e.get("source"), e.get("target")
        if s and t:
            ep.add((str(s), str(t)))
    ea: set[tuple[str, str]] = set()
    for e in eg.get("edges_a") or []:
        u, v = e.get("u"), e.get("v")
        if u is None or v is None:
            continue
        u, v = str(u), str(v)
        ea.add((u, v) if u <= v else (v, u))
    return ep, ea


def expected_edge_sets_from_edge_records(records: list[dict[str, Any]]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """与 ``entity_graph_from_class_graph`` 相同的星形展开规则下的期望边集。"""
    ep: set[tuple[str, str]] = set()
    ea: set[tuple[str, str]] = set()
    for rec in records or []:
        leg = normalize_edge_leg(rec.get("edge_leg"))
        if leg == EDGE_LEG_PRAGMATIC:
            pairs = oriented_ep_pairs_from_record(rec)
        else:
            pairs = star_oriented_pairs_from_connections(rec.get("connections") or [])
        if not pairs:
            continue
        if leg == EDGE_LEG_PRAGMATIC:
            ep.update(pairs)
        elif leg == EDGE_LEG_ASSOCIATIVE:
            for u, v in pairs:
                a, b = (u, v) if u <= v else (v, u)
                ea.add((a, b))
    return ep, ea


def verify_entity_json_matches_graph_edge_json(
    entity_graph: dict[str, Any],
    edge_records: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    """核对 ``entity_graph_*.json`` 与 ``graph_edge_*.json`` 展开边集一致（无需 ClassGraph pickle）。"""
    got_p, got_a = edge_sets_from_entity_graph_dict(entity_graph)
    exp_p, exp_a = expected_edge_sets_from_edge_records(edge_records)
    if got_p != exp_p:
        return False, (
            f"E_P 不一致: entity_json |E|={len(got_p)} 期望 |E|={len(exp_p)} "
            f"仅json={got_p - exp_p} 仅期望={exp_p - got_p}"
        )
    if got_a != exp_a:
        return False, (
            f"E_A 不一致: entity_json |E|={len(got_a)} 期望 |E|={len(exp_a)}"
        )
    return True, None


def verify_classgraph_nx_vs_edges(cg: Any) -> tuple[bool, str | None]:
    """``self.edges`` 星形展开与 ``G_p``/``G_a`` 边集一致。"""
    ok, msg = cg._dual_nx_matches_edge_records()
    return ok, msg


def verify_classgraph_nx_vs_entity_export(cg: Any) -> tuple[bool, str | None]:
    """内存 ``entity_graph_from_class_graph`` 导出边集与 ``G_p``/``G_a`` 一致。"""
    from src.graph.dual.entity_graph_store import entity_graph_from_class_graph

    eg = entity_graph_from_class_graph(cg).export()
    got_p, got_a = edge_sets_from_entity_graph_dict(eg)
    gp = set(cg.G_p.edges())
    ga = {(u, v) if u <= v else (v, u) for u, v in cg.G_a.edges()}
    if got_p != gp:
        return False, (
            f"EntityGraph E_P 与 G_p 不一致: |json|={len(got_p)} |G_p|={len(gp)} "
            f"仅json={got_p - gp} 仅G_p={gp - got_p}"
        )
    if got_a != ga:
        return False, f"EntityGraph E_A 与 G_a 不一致: |json|={len(got_a)} |G_a|={len(ga)}"
    return True, None


def verify_classgraph_full(cg: Any) -> tuple[bool, list[str]]:
    """A-3 组合：边记录↔nx、导出↔nx、DAG（G_p）。"""
    errs: list[str] = []
    ok, msg = verify_classgraph_nx_vs_edges(cg)
    if not ok and msg:
        errs.append(msg)
    ok2, msg2 = verify_classgraph_nx_vs_entity_export(cg)
    if not ok2 and msg2:
        errs.append(msg2)
    if cg.G_p.number_of_nodes() > 0 and not nx.is_directed_acyclic_graph(cg.G_p):
        errs.append("G_p 非 DAG")
    return len(errs) == 0, errs


def load_json_path(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
