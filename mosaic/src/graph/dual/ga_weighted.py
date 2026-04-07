"""
手稿 \(G_A\)：无向加权语义关联（docs/optimization.md §1、§5 B-2）。

使用本地 SentenceTransformer（BGE/MiniLM 等，路径见 ``get_embedding_model_path``）对实体描述编码，
按余弦相似度生成候选 (u,v,w)，供 ``edge_construction`` 写入 ``EDGE_LEG_ASSOCIATIVE``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from src.logger import setup_logger

_logger = setup_logger("ga_weighted")


@dataclass(frozen=True)
class EntityText:
    entity_id: str
    text: str


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return mat / n


def encode_descriptions(
    texts: Sequence[str],
    model_path: str,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """返回 (n, d) float32 已 L2 归一化的嵌入矩阵。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("需要 sentence-transformers 以构建 E_A（语义边）") from e
    model = SentenceTransformer(model_path)
    n = len(texts)
    if progress_callback:
        progress_callback(0, n)
    emb = model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    emb = np.asarray(emb, dtype=np.float32)
    emb = _l2_normalize(emb)
    if progress_callback:
        progress_callback(n, n)
    return emb


def pairwise_cosine_top_pairs(
    entity_ids: Sequence[str],
    embeddings: np.ndarray,
    *,
    min_similarity: float,
    max_pairs: int,
    min_text_len: int,
    texts_for_filter: Sequence[str] | None = None,
) -> list[tuple[str, str, float]]:
    """
    在上三角上枚举相似度，取超过阈值且全局最高的至多 ``max_pairs`` 条无向对。
    ``texts_for_filter`` 与 entity 对齐时，过短文本不参与（减少噪声）。
    """
    n = len(entity_ids)
    if n < 2 or embeddings.shape[0] != n:
        return []
    sim_m = embeddings @ embeddings.T
    candidates: list[tuple[float, int, int]] = []
    for i in range(n):
        tlen = len((texts_for_filter or [""] * n)[i].strip()) if texts_for_filter else 999
        if texts_for_filter is not None and tlen < min_text_len:
            continue
        for j in range(i + 1, n):
            tlen_j = len((texts_for_filter or [""] * n)[j].strip()) if texts_for_filter else 999
            if texts_for_filter is not None and tlen_j < min_text_len:
                continue
            s = float(sim_m[i, j])
            if s >= min_similarity:
                candidates.append((s, i, j))
    candidates.sort(key=lambda x: -x[0])
    out: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for s, i, j in candidates:
        if len(out) >= max_pairs:
            break
        a, b = entity_ids[i], entity_ids[j]
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b, s))
    return out


def collect_entity_texts_from_class_graph(cg: Any) -> list[EntityText]:
    """从 ClassGraph 收集 entity_id 与描述文本（与 EntityGraph 导出一致用 build_instance_fragments）。"""
    from src.assist import build_instance_fragments

    rows: list[EntityText] = []
    for class_node in cg.graph.nodes:
        cid = getattr(class_node, "class_id", None) or ""
        if not cid:
            continue
        for inst in getattr(class_node, "_instances", []) or []:
            if not isinstance(inst, dict):
                continue
            iid = inst.get("instance_id")
            if not iid:
                continue
            eid = f"{cid}:{iid}"
            parts: list[str] = []
            for _ty, text in build_instance_fragments(inst):
                t = (text or "").strip()
                if t:
                    parts.append(t)
            desc = "\n".join(parts) if parts else str(inst.get("instance_name", eid))
            rows.append(EntityText(entity_id=eid, text=desc))
    return rows
