"""
轨 C：对话控制状态机骨架（docs/optimization.md §6）。

离线 LoCoMo 构图路径不经过本模块；主动对话 / NCS 实验时由 Runner 逐步调用。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.config_loader import get_ncs_trace_path
from src.control.scoring import neighbors_union_gp_ga
from src.telemetry.ncs_trace import append_ncs_trace


@dataclass
class TurnTrace:
    """单步遥测（写入 JSONL 见 ``src/telemetry/ncs_trace``）。"""

    turn_index: int
    delta_entities: list[str] = field(default_factory=list)
    predicted_frontier: list[str] = field(default_factory=list)
    actual_rescore_entities: list[str] = field(default_factory=list)
    mode: str = "ncs"
    timing_ms: dict[str, float] = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "delta_entities": self.delta_entities,
            "predicted_frontier": self.predicted_frontier,
            "actual_rescore_entities": self.actual_rescore_entities,
            "mode": self.mode,
            "timing_ms": self.timing_ms,
        }


class DialogController:
    """持有 ClassGraph（或 EntityGraphStore 适配器），提供 NCS 前沿计算。"""

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def predicted_ncs_frontier(self, delta_entity_ids: set[str]) -> set[str]:
        G_p = getattr(self.memory, "G_p", None)
        G_a = getattr(self.memory, "G_a", None)
        if G_p is None or G_a is None:
            return set(delta_entity_ids)
        return neighbors_union_gp_ga(G_p, G_a, delta_entity_ids)

    def step_trace(
        self,
        turn_index: int,
        delta_entity_ids: set[str],
        *,
        mode: str = "ncs",
        actual_rescore: set[str] | None = None,
        timing_ms: dict[str, float] | None = None,
    ) -> TurnTrace:
        t0 = time.perf_counter()
        front = self.predicted_ncs_frontier(delta_entity_ids)
        pred_ms = (time.perf_counter() - t0) * 1000.0
        tm = dict(timing_ms or {})
        tm.setdefault("predicted_frontier_ms", round(pred_ms, 3))
        return TurnTrace(
            turn_index=turn_index,
            delta_entities=sorted(delta_entity_ids),
            predicted_frontier=sorted(front),
            actual_rescore_entities=sorted(actual_rescore) if actual_rescore else [],
            mode=mode,
            timing_ms=tm,
        )

    def emit_ncs_trace(self, trace: TurnTrace) -> None:
        """若配置了 ``[NCS] trace_jsonl`` 或 ``MOSAIC_NCS_TRACE_JSONL`` 则追加一行 JSON。"""
        p = get_ncs_trace_path()
        if p:
            append_ncs_trace(p, trace.to_json_dict())
