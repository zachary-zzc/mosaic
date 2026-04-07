"""
信念占位更新（docs/optimization.md §1 Bayes 式扩展前先三态）。

可与 ``EntityGraphStore.add_entity`` 的 ``belief`` 字段衔接。
"""
from __future__ import annotations

from typing import Any


def belief_unknown() -> dict[str, Any]:
    return {"state": "unknown", "entropy": None, "confidence": None}


def belief_transition_simple(prior: dict[str, Any] | None, evidence_strength: float) -> dict[str, Any]:
    """evidence_strength∈[0,1] 越大越趋向 confirmed。"""
    s = max(0.0, min(1.0, float(evidence_strength)))
    if s >= 0.75:
        st = "confirmed"
    elif s >= 0.35:
        st = "partial"
    else:
        st = "unknown"
    return {
        "state": st,
        "entropy": max(0.0, 1.0 - s),
        "confidence": s,
    }
