"""Shared I/O helpers for JSON and pickle."""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any


def parse_llm_json_object(text: str | None) -> dict[str, Any] | None:
    """
    从 LLM 回复中解析单个 JSON 对象。会去除 ```json 围栏、截取首尾花括号，
    并对常见错误做修复（如 JSON 中非法的 \\' 转义），再 json.loads。

    解析失败返回 None；成功则返回 dict（不要求一定含特定键）。
    """
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start : end + 1]

    def _loads(s: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    for candidate in (raw,):
        out = _loads(candidate)
        if out is not None:
            return out

    # LLM 常在英文撇号处输出 \'，在 JSON 字符串中非法（合法转义为 \" \\ / 等）
    repaired = re.sub(r"\\(?=')", "", raw)
    out = _loads(repaired)
    if out is not None:
        return out

    repaired2 = raw.replace("\\'", "'")
    out = _loads(repaired2)
    if out is not None:
        return out

    # 尾随逗号
    repaired3 = re.sub(r",\s*([}\]])", r"\1", raw)
    out = _loads(repaired3)
    if out is not None:
        return out
    repaired3b = re.sub(r",\s*([}\]])", r"\1", repaired2)
    out = _loads(repaired3b)
    if out is not None:
        return out

    return None


def parse_llm_json_value(text: str | None) -> Any | None:
    """
    从 LLM 回复中解析单个 JSON 值（对象或数组）。流程与 parse_llm_json_object 类似，
    但成功时返回 dict 或 list；标量或其它类型返回 None。

    用于 PROMPT_TAGS_QUERY 等要求输出 JSON 数组的接口。
    """
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()

    start_obj, end_obj = raw.find("{"), raw.rfind("}")
    start_arr, end_arr = raw.find("["), raw.rfind("]")
    if start_arr != -1 and end_arr > start_arr and (start_obj == -1 or start_arr < start_obj):
        raw = raw[start_arr : end_arr + 1]
    elif start_obj != -1 and end_obj > start_obj:
        raw = raw[start_obj : end_obj + 1]

    def _loads(s: str) -> Any | None:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, (dict, list)) else None
        except json.JSONDecodeError:
            return None

    for candidate in (raw,):
        out = _loads(candidate)
        if out is not None:
            return out

    repaired = re.sub(r"\\(?=')", "", raw)
    out = _loads(repaired)
    if out is not None:
        return out

    repaired2 = raw.replace("\\'", "'")
    out = _loads(repaired2)
    if out is not None:
        return out

    repaired3 = re.sub(r",\s*([}\]])", r"\1", raw)
    out = _loads(repaired3)
    if out is not None:
        return out
    repaired3b = re.sub(r",\s*([}\]])", r"\1", repaired2)
    out = _loads(repaired3b)
    if out is not None:
        return out

    return None


def read_json(path: str | Path) -> Any:
    """Load JSON from file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write JSON to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_pickle(path: str | Path) -> Any:
    """Load object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(path: str | Path, obj: Any) -> None:
    """Write object to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
