"""
NCS / DialogController 遥测 JSONL（docs/optimization.md §6 C-3、§8 E-3）。

每行一个 JSON 对象；路径由环境变量 ``MOSAIC_NCS_TRACE_JSONL`` 或调用方传入。
"""
from __future__ import annotations

import json
import os
from typing import Any


def append_ncs_trace(path: str | None, record: dict[str, Any]) -> None:
    p = (path or "").strip() or os.environ.get("MOSAIC_NCS_TRACE_JSONL", "").strip()
    if not p:
        try:
            from src.config_loader import get_ncs_trace_path

            p = get_ncs_trace_path().strip()
        except Exception:
            p = ""
    if not p:
        return
    line = json.dumps(record, ensure_ascii=False)
    d = os.path.dirname(os.path.abspath(p)) or "."
    os.makedirs(d, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")
