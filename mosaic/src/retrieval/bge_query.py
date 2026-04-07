"""
查询阶段 BGE 编码与余弦相似度（docs/optimization.md §7 D-2）。

与构图阶段共用 ``get_embedding_model_path()``；大批量实例编码时显示 ``show_progress_bar``。
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from src.logger import setup_logger

_logger = setup_logger("bge_query")


def _l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def query_instance_cosine_similarities(
    query: str,
    instance_texts: Sequence[str],
    model_path: str,
) -> tuple[list[float], float]:
    """
    返回与 ``instance_texts`` 等长的余弦相似度列表（已 L2 归一化嵌入），及 wall_ms。
    """
    texts = [t if (t or "").strip() else " " for t in instance_texts]
    if not texts:
        return [], 0.0
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("需要 sentence-transformers 以启用查询 BGE") from e
    t0 = time.perf_counter()
    model = SentenceTransformer(model_path)
    q_emb = model.encode([query], convert_to_numpy=True)
    doc_emb = model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 48,
    )
    q_emb = _l2n(np.asarray(q_emb, dtype=np.float32))
    doc_emb = _l2n(np.asarray(doc_emb, dtype=np.float32))
    sims = (q_emb @ doc_emb.T)[0].astype(np.float64)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return sims.tolist(), wall_ms


def minmax_01(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.5] * len(values)
    return [(float(v) - lo) / (hi - lo) for v in values]
