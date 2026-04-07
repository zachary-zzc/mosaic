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
    并对常见错误做修复（如 JSON 中非法的 \\' 转义、尾随逗号、单引号键值、
    控制字符、截断 JSON 等），再 json.loads。

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

    # 控制字符清理（LLM 可能输出 \x00-\x1f 中非法字符）
    repaired4 = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    out = _loads(repaired4)
    if out is not None:
        return out

    # 字符串值内的非法换行（不在引号外的 \n）
    repaired5 = _fix_unescaped_newlines(raw)
    out = _loads(repaired5)
    if out is not None:
        return out

    # 单引号 → 双引号（简单情况：无嵌套引号的键值）
    repaired6 = _single_to_double_quotes(raw)
    out = _loads(repaired6)
    if out is not None:
        return out

    # 截断 JSON 修复：尝试关闭未闭合的引号和括号
    repaired7 = _fix_truncated_json(raw)
    if repaired7 != raw:
        out = _loads(repaired7)
        if out is not None:
            return out

    # 组合修复：尾随逗号 + 控制字符 + 截断
    combined = re.sub(r",\s*([}\]])", r"\1", repaired4)
    combined = _fix_truncated_json(combined)
    out = _loads(combined)
    if out is not None:
        return out

    return None


def _fix_unescaped_newlines(s: str) -> str:
    """修复 JSON 字符串值内的裸换行符。"""
    result = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == '\\':
            escape = True
            result.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch == '\n':
            result.append('\\n')
            continue
        if in_string and ch == '\r':
            result.append('\\r')
            continue
        result.append(ch)
    return ''.join(result)


def _single_to_double_quotes(s: str) -> str:
    """将单引号风格的 JSON 转为双引号（仅处理键和简单字符串值）。"""
    # 仅在整体看起来像单引号 JSON 时尝试
    if "'" not in s or s.count('"') > s.count("'"):
        return s
    result = []
    in_single = False
    in_double = False
    escape = False
    for ch in s:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == '\\':
            escape = True
            result.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            result.append(ch)
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            result.append('"')
            continue
        result.append(ch)
    return ''.join(result)


def _fix_truncated_json(s: str) -> str:
    """尝试关闭被截断的 JSON（未闭合的引号、数组、对象）。"""
    s = s.rstrip()
    # 移除末尾不完整的键值对（如 "key": 无值、"key 无冒号）
    s = re.sub(r',\s*"[^"]*"\s*:\s*$', '', s)
    s = re.sub(r',\s*"[^"]*"\s*$', '', s)
    # 计算未闭合的括号
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
    # 如果有未闭合的字符串引号，先关闭
    if in_str:
        s += '"'
    # 关闭未闭合的括号（按逆序）
    for bracket in reversed(stack):
        s += '}' if bracket == '{' else ']'
    return s


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


def llm_response_text(message: Any) -> str:
    """
    从 LangChain ChatMessage（或类似对象）提取用于 JSON 解析的字符串。
    兼容 content 为 str / 多段列表、以及部分适配器写在 additional_kwargs 里的正文。
    """
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        s = content.strip()
        if s:
            return s
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
        joined = "".join(parts).strip()
        if joined:
            return joined

    ak = getattr(message, "additional_kwargs", None) or {}
    if isinstance(ak, dict):
        for key in ("content", "text", "reasoning_content"):
            v = ak.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

    fallback = str(message).strip()
    return fallback


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
