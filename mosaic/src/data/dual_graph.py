"""
P-1 双图边类型（与 docs/optimization.md 对齐）。

- **E_P（过程/共现）**：`EDGE_LEG_PRAGMATIC`，记为 ``"P"``。当前构图里「同对话消息标签」关联的实例边归此类。
- **E_A（语义/联想）**：`EDGE_LEG_ASSOCIATIVE`，记为 ``"A"``。预留；后续 TF-IDF/LLM 类间语义边等写入同一 `edge_record` 结构并带 `edge_leg`。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

EDGE_LEG_PRAGMATIC = "P"
EDGE_LEG_ASSOCIATIVE = "A"

ALL_EDGE_LEGS = frozenset((EDGE_LEG_PRAGMATIC, EDGE_LEG_ASSOCIATIVE))


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
