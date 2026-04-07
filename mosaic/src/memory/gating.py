"""
置信门控 \(\rho_{\min}\) 占位（docs/optimization.md §1、§3）。

与 ``EntityNode.belief`` 绑定的状态机待实验 Runner 接入；此处仅保留 API 形状。
"""
from __future__ import annotations

from typing import Any


def should_commit_to_ltm(belief: dict[str, Any] | None, rho_min: float = 0.5) -> bool:
    """belief 含 ``confidence`` 或 ``state`` 时做简单阈值判断。"""
    if not belief:
        return False
    c = belief.get("confidence")
    if c is not None:
        try:
            return float(c) >= rho_min
        except (TypeError, ValueError):
            pass
    st = str(belief.get("state") or "").lower()
    return st in ("confirmed", "partial")
