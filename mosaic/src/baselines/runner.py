"""
轨 F：外部基线统一接口（docs/optimization.md §9）。

Mosaic 主路径不继承此类；Mem0/Letta 等适配器可实现 ``ingest`` / ``answer``。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaselineRunner(ABC):
    @abstractmethod
    def ingest(self, dialogue: Any) -> None:
        """摄入一轮或多轮对话（域相关结构）。"""

    @abstractmethod
    def answer(self, question: str) -> str:
        """仅返回答案文本；评测侧与 Mosaic 共用 judge。"""


class MemoryOnlyStubRunner(BaselineRunner):
    """memory_only 占位：不读图，答案由子类或空实现。"""

    def ingest(self, dialogue: Any) -> None:
        return None

    def answer(self, question: str) -> str:
        return ""
