"""
轨 D：检索融合工具（docs/optimization.md §7 D-2）。

TF-IDF 分数与 BGE 余弦分数线性融合：\(\lambda \cdot s_{\mathrm{bge}} + (1-\lambda) \cdot s_{\mathrm{tfidf}}\)。
查询路径接入由 ``query.py`` / ``ClassGraph._search_*`` 在后续迭代中调用 ``fuse_retrieval_scores``。
"""
from __future__ import annotations

from typing import Sequence


def fuse_retrieval_scores(
    tfidf: Sequence[float],
    bge: Sequence[float],
    *,
    lambda_bge: float = 0.35,
) -> list[float]:
    if len(tfidf) != len(bge):
        raise ValueError("tfidf 与 bge 长度须一致")
    lam = max(0.0, min(1.0, float(lambda_bge)))
    return [lam * float(b) + (1.0 - lam) * float(t) for t, b in zip(tfidf, bge)]
