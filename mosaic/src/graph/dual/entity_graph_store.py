"""
A-2：手稿级 EntityGraph 内存存储与导出（docs/optimization.md §3.1、§4）。

从现有 ClassGraph + self.edges 做 **适配器式** 导出：实体 id 规则为 ``{class_id}:{instance_id}``；
共现超边按字典序选 hub，向其余节点连有向边（星形定向），保证 G_P 无环以便验收 DAG 校验。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import networkx as nx

from src.assist import build_instance_fragments
from src.data.dual_graph import EDGE_LEG_ASSOCIATIVE, EDGE_LEG_PRAGMATIC, normalize_edge_leg
from src.graph.dual.hyperedge import oriented_ep_pairs_from_record, star_oriented_pairs_from_connections

ENTITY_GRAPH_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EntityGraphStore:
    """内存中的 EntityGraph；``export()`` 产出与 ``entity_graph.schema.json`` 一致的 dict。"""

    def __init__(
        self,
        *,
        version: str = "1",
        build_mode: str = "hybrid",
        conversation_id: str = "unknown",
    ) -> None:
        self.version = version
        self.build_mode = build_mode if build_mode in ("hybrid", "hash_only") else "hybrid"
        self.conversation_id = conversation_id
        self._entities: dict[str, dict[str, Any]] = {}
        self._edges_p: list[dict[str, Any]] = []
        self._edges_a: list[dict[str, Any]] = []
        self._communities: dict[str, str] = {}
        self._legacy_graph_info: dict[str, Any] | None = None

    def add_entity(
        self,
        entity_id: str,
        *,
        canonical_name: str,
        description: str,
        belief: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._entities[entity_id] = {
            "entity_id": entity_id,
            "canonical_name": canonical_name,
            "description": description,
            "belief": belief if belief is not None else {"state": "unknown", "entropy": None},
            "metadata": metadata if metadata is not None else {},
        }

    def add_edge_p(
        self,
        source: str,
        target: str,
        *,
        confidence: float | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        rec: dict[str, Any] = {
            "source": source,
            "target": target,
            "leg": EDGE_LEG_PRAGMATIC,
            "confidence": confidence,
            "provenance": provenance,
        }
        self._edges_p.append(rec)

    def add_edge_a(
        self,
        u: str,
        v: str,
        *,
        weight: float,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        a, b = (u, v) if u <= v else (v, u)
        self._edges_a.append(
            {
                "u": a,
                "v": b,
                "leg": EDGE_LEG_ASSOCIATIVE,
                "weight": float(weight),
                "provenance": provenance,
            }
        )

    def set_communities(self, mapping: dict[str, str]) -> None:
        self._communities = dict(mapping)

    def set_legacy_graph_info(self, info: dict[str, Any] | None) -> None:
        self._legacy_graph_info = info

    def validate_dag(self) -> tuple[bool, str | None]:
        """仅在 ``edges_p`` 上构造 DiGraph，检查是否为 DAG。"""
        g = nx.DiGraph()
        for e in self._edges_p:
            s, t = e.get("source"), e.get("target")
            if not s or not t:
                continue
            g.add_edge(s, t)
        if g.number_of_nodes() == 0:
            return True, None
        if nx.is_directed_acyclic_graph(g):
            return True, None
        try:
            cyc = nx.find_cycle(g, orientation="original")
            return False, repr(cyc)
        except Exception as err:
            return False, str(err)

    def export(self) -> dict[str, Any]:
        dag_ok, dag_detail = self.validate_dag()
        out: dict[str, Any] = {
            "schema_version": ENTITY_GRAPH_SCHEMA_VERSION,
            "version": self.version,
            "build_mode": self.build_mode,
            "conversation_id": self.conversation_id,
            "exported_at": _utc_now_iso(),
            "entities": dict(self._entities),
            "edges_p": list(self._edges_p),
            "edges_a": list(self._edges_a),
            "communities": dict(self._communities),
            "dag_valid": dag_ok,
        }
        if not dag_ok and dag_detail:
            out["dag_violation_detail"] = dag_detail
        if self._legacy_graph_info is not None:
            out["legacy_graph_info"] = self._legacy_graph_info
        return out

    def write_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export(), f, indent=2, ensure_ascii=False)


def _instance_description(inst: dict[str, Any]) -> str:
    parts: list[str] = []
    for _ty, text in build_instance_fragments(inst):
        t = (text or "").strip()
        if t:
            parts.append(t)
    if parts:
        return "\n".join(parts)
    raw = str(inst)
    return raw
    # return raw if len(raw) <= 8000 else raw[:8000] + "…"


def entity_graph_from_class_graph(cg: Any) -> EntityGraphStore:
    """
    从 ClassGraph（networkx 存 ClassNode）与 ``cg.edges`` 构建 EntityGraphStore。

    不 import ClassGraph 类型，避免循环依赖；运行时传入 ClassGraph 实例即可。
    """
    build_mode = os.environ.get("MOSAIC_BUILD_EFFECTIVE", "hybrid")
    if build_mode not in ("hybrid", "hash_only"):
        build_mode = "hybrid"
    conv = getattr(cg, "filepath", None) or "unknown"
    store = EntityGraphStore(version="1", build_mode=build_mode, conversation_id=str(conv))

    dual_counts: dict[str, int] | None = None
    try:
        from src.data.dual_graph import count_edge_legs

        dual_counts = count_edge_legs(getattr(cg, "edges", []) or [])
    except Exception:
        dual_counts = None

    legacy: dict[str, Any] = {
        "total_classes": len(getattr(cg.graph, "nodes", []) or []),
        "total_edge_records": len(getattr(cg, "edges", []) or []),
        "dual_graph_edge_counts": dual_counts,
    }
    tel = getattr(cg, "sense_class_telemetry_cumulative", None)
    if isinstance(tel, dict):
        legacy["construction_telemetry"] = dict(tel)
    store.set_legacy_graph_info(legacy)

    for class_node in cg.graph.nodes:
        cid = getattr(class_node, "class_id", None) or "unknown_class"
        cname = getattr(class_node, "class_name", "") or "Unclassified"
        instances = getattr(class_node, "_instances", None) or []
        for idx, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue
            iid = inst.get("instance_id") or f"instance_{idx + 1}"
            eid = f"{cid}:{iid}"
            canon = str(inst.get("instance_name") or cname)
            store.add_entity(
                eid,
                canonical_name=canon,
                description=_instance_description(inst),
                metadata={
                    "class_id": cid,
                    "class_name": cname,
                    "message_labels": inst.get("message_labels", []),
                },
            )

    comms = getattr(cg, "_entity_communities", None)
    if isinstance(comms, dict) and comms:
        store.set_communities(comms)

    for rec in getattr(cg, "edges", []) or []:
        leg = normalize_edge_leg(rec.get("edge_leg"))
        prov_base: dict[str, Any] = {
            "kind": "cooccurrence_message",
            "message_label": rec.get("label"),
            "content_preview": (rec.get("content") or "")[:240],
        }
        extra = rec.get("provenance")
        if isinstance(extra, dict):
            prov_base = {**prov_base, **extra}
        if leg == EDGE_LEG_PRAGMATIC:
            pairs = oriented_ep_pairs_from_record(rec)
            if not pairs:
                continue
            for u, v in pairs:
                store.add_edge_p(u, v, provenance={**prov_base, "edge_leg": EDGE_LEG_PRAGMATIC})
        else:
            pairs = star_oriented_pairs_from_connections(rec.get("connections") or [])
            if not pairs:
                continue
            w = float(rec.get("weight", 1.0))
            for u, v in pairs:
                store.add_edge_a(u, v, weight=w, provenance={**prov_base, "edge_leg": EDGE_LEG_ASSOCIATIVE})

    return store
