"""
LLM 调用遥测：阶段（构图/查询）、步骤名、耗时统计、可选 JSONL 全量 prompt/响应。

环境变量：
- MOSAIC_LLM_IO_LOG：JSONL 路径（与 --log-prompt 一致），记录 messages + response + duration_ms
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

_llm_step: ContextVar[str] = ContextVar("mosaic_llm_step", default="unknown")
_llm_phase: ContextVar[str] = ContextVar("mosaic_llm_phase", default="other")

_lock = threading.Lock()
_counters: dict[str, float | int] = {
    "build_calls": 0,
    "build_ms": 0.0,
    "total_calls": 0,
    "total_ms": 0.0,
}


def reset_build_llm_counters() -> None:
    with _lock:
        _counters["build_calls"] = 0
        _counters["build_ms"] = 0.0


def get_llm_counters() -> dict[str, float | int]:
    with _lock:
        return dict(_counters)


def _record_llm_completion_locked(*, duration_ms: float) -> None:
    _counters["total_calls"] = int(_counters["total_calls"]) + 1
    _counters["total_ms"] = float(_counters["total_ms"]) + duration_ms
    if _llm_phase.get() == "build":
        _counters["build_calls"] = int(_counters["build_calls"]) + 1
        _counters["build_ms"] = float(_counters["build_ms"]) + duration_ms


def record_llm_http_roundtrip(
    *,
    duration_ms: float,
    messages: list[dict[str, Any]],
    response_text: str,
    model_name: str = "",
) -> None:
    """在 Chat 完成一次 HTTP 往返后调用（Qwen / Custom 等）。"""
    with _lock:
        _record_llm_completion_locked(duration_ms=duration_ms)

    path = os.environ.get("MOSAIC_LLM_IO_LOG", "").strip()
    if not path:
        return
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "step": _llm_step.get(),
        "phase": _llm_phase.get(),
        "model": model_name,
        "duration_ms": round(duration_ms, 3),
        "messages": messages,
        "response": response_text,
    }
    try:
        d = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(d, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


@contextmanager
def llm_call_scope(step: str):
    tok = _llm_step.set(step)
    try:
        yield
    finally:
        _llm_step.reset(tok)


@contextmanager
def llm_phase_scope(phase: str):
    tok = _llm_phase.set(phase)
    try:
        yield
    finally:
        _llm_phase.reset(tok)


def dump_build_metrics_file(path: str, extra: dict[str, Any] | None = None) -> None:
    """写入构图阶段 LLM 统计（及可选元数据）。"""
    payload: dict[str, Any] = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "llm_http": get_llm_counters(),
    }
    if extra:
        payload.update(extra)
    try:
        d = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Ingest-level telemetry (merged from src/telemetry/ingest_log.py)
# ---------------------------------------------------------------------------

import time as _time


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
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
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
