"""
P-1 双图边类型与边记录约定（与 docs/optimization.md、Manuscript DualGraph 对齐）。

**E_P（前提图 \(G_P\)）** — ``EDGE_LEG_PRAGMATIC`` = ``"P"``：

- **共现星形**：同一条对话消息标签上多个实例共现；``connections``≥2，展开为字典序 hub→leaf（``hyperedge.oriented_ep_pairs_from_record``）。
- **显式有向先决**：``ep_oriented_pairs``: ``[[u, v], ...]`` 其中 ``entity_id`` 为 ``class_id:instance_id``；用于 LLM/BGE 后处理得到的非对称先决，须经 DAG 校验。

**E_A（关联图 \(G_A\)）** — ``EDGE_LEG_ASSOCIATIVE`` = ``"A"``：

- 无向加权边；``connections`` 通常为两实例；``weight``∈[0,1]（如 BGE 余弦）。
- ``provenance.kind`` 建议 ``semantic_bge_cosine`` 等，便于导出与消融实验。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

EDGE_LEG_PRAGMATIC = "P"
EDGE_LEG_ASSOCIATIVE = "A"

ALL_EDGE_LEGS = frozenset((EDGE_LEG_PRAGMATIC, EDGE_LEG_ASSOCIATIVE))

# provenance.kind（EntityGraph / graph_edge 诊断）
EDGE_KIND_MESSAGE_COOCCURRENCE = "cooccurrence_message"
EDGE_KIND_SEMANTIC_BGE = "semantic_bge_cosine"
EDGE_KIND_PREREQUISITE_LLM = "llm_prerequisite_asymmetric"


def entity_id_to_connection(entity_id: str) -> dict[str, str] | None:
    """``class_3:instance_1`` → {class_id, instance_id}。"""
    if not entity_id or ":" not in entity_id:
        return None
    cid, iid = entity_id.split(":", 1)
    if not cid.strip() or not iid.strip():
        return None
    return {"class_id": cid.strip(), "instance_id": iid.strip()}


def edge_record_associative_pair(
    entity_u: str,
    entity_v: str,
    *,
    weight: float,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """二元 E_A 边记录（写入 ``ClassGraph.edges``）。"""
    cu = entity_id_to_connection(entity_u)
    cv = entity_id_to_connection(entity_v)
    if not cu or not cv:
        raise ValueError(f"invalid entity ids: {entity_u!r} {entity_v!r}")
    prov = {"kind": EDGE_KIND_SEMANTIC_BGE, **(provenance or {})}
    return {
        "edge_leg": EDGE_LEG_ASSOCIATIVE,
        "weight": float(weight),
        "content": "",
        "label": None,
        "connections": [cu, cv],
        "provenance": prov,
    }


def edge_record_prerequisite_oriented(
    source_entity: str,
    target_entity: str,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """单弧 E_P 边记录（``ep_oriented_pairs``），用于 LLM 先决等。"""
    cs = entity_id_to_connection(source_entity)
    ct = entity_id_to_connection(target_entity)
    if not cs or not ct:
        raise ValueError(f"invalid entity ids: {source_entity!r} {target_entity!r}")
    prov = {"kind": EDGE_KIND_PREREQUISITE_LLM, **(provenance or {})}
    return {
        "edge_leg": EDGE_LEG_PRAGMATIC,
        "content": "",
        "label": "llm_prerequisite",
        "connections": [cs, ct],
        "ep_oriented_pairs": [[source_entity, target_entity]],
        "provenance": prov,
    }


def normalize_edge_leg(raw: str | None) -> str:
    """旧 JSON 无 `edge_leg` 时视为 P，保证向后兼容。"""
    if raw == EDGE_LEG_ASSOCIATIVE:
        return EDGE_LEG_ASSOCIATIVE
    return EDGE_LEG_PRAGMATIC


def count_edge_legs(edge_records: list[dict[str, Any]]) -> dict[str, int]:
    """按 `edge_leg` 统计条数；始终包含 P / A 键（无记录时为 0）。"""
    c: Counter[str] = Counter({EDGE_LEG_PRAGMATIC: 0, EDGE_LEG_ASSOCIATIVE: 0})
    for rec in edge_records:
        c[normalize_edge_leg(rec.get("edge_leg"))] += 1
    return dict(c)
