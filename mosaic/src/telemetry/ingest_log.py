"""
轨 E-3：构图 ingest 单行 JSONL（docs/optimization.md §8）。

路径：``[TELEMETRY] ingest_jsonl`` 或环境变量 ``MOSAIC_INGEST_JSONL``。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any


def _max_rss_mb() -> float | None:
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if ru > 0 and ru < 1 << 40:
            return round(ru / 1024.0 / 1024.0, 3) if ru > 10**6 else round(ru / 1024.0, 3)
    except Exception:
        pass
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 3)
    except Exception:
        return None


def append_ingest_record(
    *,
    conversation_id: str,
    wall_s: float,
    memory: Any,
    llm_calls: int | None = None,
    json_failures: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    cp_path = ""
    try:
        from src.config_loader import load_api_config

        cp = load_api_config()
        if cp.has_section("TELEMETRY"):
            cp_path = cp.get("TELEMETRY", "ingest_jsonl", fallback="").strip()
    except Exception:
        pass
    path = os.environ.get("MOSAIC_INGEST_JSONL", "").strip() or cp_path
    if not path:
        return

    n_inst = sum(len(getattr(n, "_instances", []) or []) for n in getattr(memory.graph, "nodes", []) or [])
    try:
        from src.graph.dual.hyperedge import unique_directed_star_pairs_p, unique_undirected_star_pairs_a

        ep = len(unique_directed_star_pairs_p(getattr(memory, "edges", []) or []))
        ea = len(unique_undirected_star_pairs_a(getattr(memory, "edges", []) or []))
    except Exception:
        ep, ea = 0, 0

    row: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conversation_id": conversation_id,
        "wall_s": round(float(wall_s), 3),
        "max_rss_mb": _max_rss_mb(),
        "|V|": n_inst,
        "|E_P|": ep,
        "|E_A|": ea,
        "llm_calls": llm_calls,
        "json_failures": json_failures,
    }
    if extra:
        row.update(extra)
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
