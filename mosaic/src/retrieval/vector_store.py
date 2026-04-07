"""
实体级向量索引（docs/optimization.md §1 LTM 检索占位）。

无 HNSW 时提供 **暴力余弦 Top-m**（``BruteForceEntityIndex``），与手稿 Limitation「小 m、暴力」一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def _l2n(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return mat / n


@dataclass
class BruteForceEntityIndex:
    """entity_id → 行索引；查询时全量余弦，返回 Top-m。"""

    model_path: str
    _ids: list[str] = field(default_factory=list)
    _mat: np.ndarray | None = None

    def build_from_texts(self, entity_ids: Sequence[str], texts: Sequence[str]) -> None:
        from sentence_transformers import SentenceTransformer

        self._ids = list(entity_ids)
        safe = [t if (t or "").strip() else " " for t in texts]
        model = SentenceTransformer(self.model_path)
        emb = model.encode(list(safe), convert_to_numpy=True, show_progress_bar=len(safe) > 64)
        self._mat = _l2n(np.asarray(emb, dtype=np.float32))

    def search(self, query: str, top_m: int = 16) -> list[tuple[str, float]]:
        if not self._ids or self._mat is None:
            return []
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_path)
        q = model.encode([query], convert_to_numpy=True)
        q = _l2n(np.asarray(q, dtype=np.float32))
        sims = (q @ self._mat.T)[0]
        k = min(top_m, len(self._ids))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._ids[int(i)], float(sims[int(i)])) for i in idx]
