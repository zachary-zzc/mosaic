#!/usr/bin/env python3
"""
最小化联调：验证 DashScope 兼容模式 + QwenChatModel + response_format=json_object
能否返回可 json.loads 的内容（与 instance 构图路径一致）。

用法（在 mosaic 目录下）:
  cd mosaic && PYTHONPATH=. python scripts/test_dashscope_json_mode.py

依赖: config/config.cfg 中 [API_KEYS] 或环境变量 DASHSCOPE_API_KEY / DASHSCOPE_API_BASE。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 保证可从任意 cwd 导入 src（与 __main__ 运行方式一致）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config_loader import get_api_key_and_base_url  # noqa: E402
from src.llm.llm import QwenChatModel  # noqa: E402

_JSON_OBJECT = {"type": "json_object"}


def main() -> int:
    key, base = get_api_key_and_base_url()
    if not key or not base:
        print("错误: 未配置 API Key 或 base_url（config.cfg 或 DASHSCOPE_* 环境变量）", file=sys.stderr)
        return 2

    llm = QwenChatModel(api_key=key, base_url=base, model_name="qwen3.5-plus", temperature=0.0)
    # 百炼要求 messages 中含 json 字样；输出须为单个 JSON 对象
    prompt = """请用 JSON 返回一个对象，包含两个键：
- "hello": 字符串 "ok"
- "n": 整数 42
不要输出 markdown，不要代码块，只输出 JSON 对象。"""

    print("请求中（json_object 模式）…")
    try:
        msg = llm.invoke(prompt, response_format=_JSON_OBJECT)
    except Exception as e:
        print(f"API 调用失败: {e}", file=sys.stderr)
        return 1

    raw = (msg.content or "").strip()
    print("--- 原始 content（前 500 字符）---")
    print(raw[:500] + ("…" if len(raw) > 500 else ""))

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"json.loads 失败: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print(f"期望根类型为 object(dict)，得到: {type(data).__name__}", file=sys.stderr)
        return 1

    if data.get("hello") != "ok" or data.get("n") != 42:
        print("警告: 结构已解析为 JSON，但字段与预期不完全一致（模型仍返回了合法 JSON）:", data)

    print("--- 校验 ---")
    print("json.loads: 通过")
    print("根节点为 JSON 对象: 通过")
    print("对接 response_format + QwenChatModel: 正常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
