"""
构图完成后的双图增强：E_A（BGE 语义）与可选 E_P（LLM 非对称先决，DAG 安全）。

docs/optimization.md §4 A-2、§5 B-2；由 ``ClassGraph.enrich_dual_graph_edges_post_build`` / ``save.save`` 调用。
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x

from src.config_loader import get_edge_construction_config, get_embedding_model_path
from src.data.dual_graph import edge_record_associative_pair, edge_record_prerequisite_oriented
from src.graph.dual.dag_gp import trial_add_edge_preserves_dag
from src.graph.dual.ga_weighted import collect_entity_texts_from_class_graph, encode_descriptions, pairwise_cosine_top_pairs
from src.graph.dual.hyperedge import unique_undirected_star_pairs_a
from src.llm.telemetry import llm_call_scope
from src.logger import setup_logger
from src.prompts_entity_graph_en import PROMPT_PREREQUISITE_BATCH
from src.utils.io_utils import parse_llm_json_object

_logger = setup_logger("edge_construction")


def _existing_a_pairs(cg: Any) -> set[tuple[str, str]]:
    ua = unique_undirected_star_pairs_a(getattr(cg, "edges", []) or [])
    return ua


def _append_edge_record(cg: Any, rec: dict[str, Any]) -> None:
    if not hasattr(cg, "edges") or cg.edges is None:
        cg.edges = []
    cg.edges.append(rec)
    cg._apply_edge_record_to_dual_nx(rec)


def add_semantic_association_edges_bge(cg: Any, cfg: Any) -> dict[str, int]:
    stats = {"pairs_considered": 0, "edges_added": 0, "skipped": 0}
    rows = collect_entity_texts_from_class_graph(cg)
    if len(rows) < 2:
        _logger.info("E_A BGE: 实体少于 2，跳过")
        return stats
    path = get_embedding_model_path()
    if not os.path.isdir(path):
        _logger.warning("E_A BGE: 嵌入模型目录不存在 %s，跳过", path)
        return stats

    ids = [r.entity_id for r in rows]
    texts = [r.text for r in rows]
    stats["pairs_considered"] = len(rows) * (len(rows) - 1) // 2

    try:
        emb = encode_descriptions(texts, path)
    except Exception as e:
        _logger.warning("E_A BGE: 编码失败，跳过: %s", e)
        return stats

    # 缓存实体嵌入到 ClassGraph，查询阶段直接复用，避免重复编码
    cache = {eid: emb[i] for i, eid in enumerate(ids)}
    if not hasattr(cg, "_bge_embedding_cache") or cg._bge_embedding_cache is None:
        cg._bge_embedding_cache = {}
    cg._bge_embedding_cache.update(cache)
    _logger.info("E_A BGE: 已缓存 %d 条实体嵌入供查询复用", len(cache))

    existing_a = _existing_a_pairs(cg)
    pairs = pairwise_cosine_top_pairs(
        ids,
        emb,
        min_similarity=cfg.semantic_min_similarity,
        max_pairs=cfg.semantic_max_pairs,
        min_text_len=cfg.semantic_min_text_len,
        texts_for_filter=texts,
    )

    for u, v, w in tqdm(pairs, desc="E_A 语义边", unit="edge"):
        a, b = (u, v) if u <= v else (v, u)
        if (a, b) in existing_a:
            stats["skipped"] += 1
            continue
        try:
            rec = edge_record_associative_pair(
                u,
                v,
                weight=w,
                provenance={"similarity": w, "embedding_model_path": path},
            )
            _append_edge_record(cg, rec)
            existing_a.add((a, b))
            stats["edges_added"] += 1
        except Exception as e:
            _logger.debug("E_A 跳过一对 %s %s: %s", u, v, e)
            stats["skipped"] += 1
    return stats


def ensure_minimum_connectivity(cg: Any, cfg: Any) -> dict[str, int]:
    """
    Post-construction pass: connect isolated entities (degree 0 in both G_p
    and G_a) to their nearest neighbor by embedding similarity, ignoring the
    ``semantic_min_similarity`` threshold.

    This prevents retrieval dead-ends where neighbor expansion can never reach
    an isolated entity.  Uses the BGE embedding cache populated by
    ``add_semantic_association_edges_bge``.
    """
    import numpy as np

    stats: dict[str, int] = {"isolated_found": 0, "edges_added": 0}

    emb_cache: dict[str, np.ndarray] | None = getattr(cg, "_bge_embedding_cache", None)
    if not emb_cache or len(emb_cache) < 2:
        return stats

    # Nodes present in either dual graph → connected
    connected: set[str] = set()
    gp = getattr(cg, "G_p", None)
    ga = getattr(cg, "G_a", None)
    if gp is not None:
        connected.update(gp.nodes())
    if ga is not None:
        connected.update(ga.nodes())

    # Isolated = in embedding cache but absent from both dual graphs
    all_ids = list(emb_cache.keys())
    isolated = [eid for eid in all_ids if eid not in connected]
    if not isolated:
        return stats

    stats["isolated_found"] = len(isolated)
    _logger.info("Minimum connectivity: %d isolated entities found", len(isolated))

    # Build target pool: entities that ARE connected
    targets = [eid for eid in all_ids if eid in connected]
    if not targets:
        return stats

    target_emb = np.stack([emb_cache[eid] for eid in targets])
    existing_a = _existing_a_pairs(cg)

    for iso_eid in isolated:
        iso_emb = emb_cache[iso_eid].reshape(1, -1)
        sims = (iso_emb @ target_emb.T).flatten()
        best_idx = int(np.argmax(sims))
        best_eid = targets[best_idx]
        best_sim = float(sims[best_idx])

        a, b = (iso_eid, best_eid) if iso_eid <= best_eid else (best_eid, iso_eid)
        if (a, b) in existing_a:
            continue

        try:
            rec = edge_record_associative_pair(
                iso_eid,
                best_eid,
                weight=best_sim,
                provenance={"similarity": best_sim, "kind": "minimum_connectivity"},
            )
            _append_edge_record(cg, rec)
            existing_a.add((a, b))
            stats["edges_added"] += 1
        except Exception as e:
            _logger.debug("Minimum connectivity skip %s: %s", iso_eid, e)

    _logger.info(
        "Minimum connectivity: added %d edges for %d isolated entities",
        stats["edges_added"],
        stats["isolated_found"],
    )
    return stats


def _prerequisite_candidate_pairs(
    cg: Any,
    cfg: Any,
    entity_ids: list[str],
    texts: list[str],
    emb: Any,
) -> list[tuple[str, str, float]]:
    """与 E_A 类似但阈值略低，并排除已有无向 A 边，供 LLM 判向。"""
    existing_a = _existing_a_pairs(cg)
    pairs = pairwise_cosine_top_pairs(
        entity_ids,
        emb,
        min_similarity=cfg.prerequisite_min_similarity,
        max_pairs=max(cfg.prerequisite_max_pairs * 4, 32),
        min_text_len=cfg.semantic_min_text_len,
        texts_for_filter=texts,
    )
    out: list[tuple[str, str, float]] = []
    for u, v, s in pairs:
        a, b = (u, v) if u <= v else (v, u)
        if (a, b) in existing_a:
            continue
        out.append((u, v, s))
        if len(out) >= cfg.prerequisite_max_pairs:
            break
    return out


def add_llm_prerequisite_edges(cg: Any, llm: Any, cfg: Any) -> dict[str, int]:
    stats = {"batches": 0, "edges_added": 0, "skipped_cycle": 0, "skipped_parse": 0}
    if cfg.prerequisite_max_pairs <= 0 or llm is None:
        return stats
    rows = collect_entity_texts_from_class_graph(cg)
    if len(rows) < 2:
        return stats
    path = get_embedding_model_path()
    if not os.path.isdir(path):
        _logger.warning("LLM 先决: 无嵌入模型，跳过候选生成")
        return stats
    ids = [r.entity_id for r in rows]
    texts = [r.text for r in rows]
    try:
        emb = encode_descriptions(texts, path)
    except Exception as e:
        _logger.warning("LLM 先决: 编码失败 %s", e)
        return stats

    candidates = _prerequisite_candidate_pairs(cg, cfg, ids, texts, emb)
    if not candidates:
        return stats

    id_to_text = dict(zip(ids, texts, strict=False))
    batch_size = max(1, cfg.prerequisite_batch_size)
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        payload = []
        for i, (u, v, _s) in enumerate(batch):
            payload.append(
                {
                    "index": i,
                    "u_id": u,
                    "v_id": v,
                    "u_text": (id_to_text.get(u) or "")[:1200],
                    "v_text": (id_to_text.get(v) or "")[:1200],
                }
            )
        prompt = PROMPT_PREREQUISITE_BATCH.substitute(PAIRS_JSON=json.dumps(payload, ensure_ascii=False))
        try:
            with llm_call_scope("build.prerequisite_batch"):
                resp = llm.invoke(prompt)
            raw = getattr(resp, "content", None) or str(resp)
            parsed = parse_llm_json_object(raw)
            if not isinstance(parsed, dict):
                stats["skipped_parse"] += len(batch)
                continue
            decisions = parsed.get("decisions")
            if not isinstance(decisions, list):
                stats["skipped_parse"] += len(batch)
                continue
            by_idx = {}
            for d in decisions:
                if isinstance(d, dict) and "index" in d:
                    by_idx[int(d["index"])] = d.get("relation", "none")
            stats["batches"] += 1
            for i, (u, v, _s) in enumerate(batch):
                rel = by_idx.get(i, "none")
                src, tgt = None, None
                if rel == "u_before_v":
                    src, tgt = u, v
                elif rel == "v_before_u":
                    src, tgt = v, u
                if not src or not tgt:
                    continue
                if not trial_add_edge_preserves_dag(cg.G_p, src, tgt):
                    stats["skipped_cycle"] += 1
                    continue
                rec = edge_record_prerequisite_oriented(
                    src,
                    tgt,
                    provenance={"batch_start": start, "relation": rel},
                )
                _append_edge_record(cg, rec)
                stats["edges_added"] += 1
        except Exception as e:
            _logger.warning("LLM 先决 batch 失败: %s", e)
            stats["skipped_parse"] += len(batch)
    return stats


def enrich_class_graph_dual_edges(cg: Any, *, llm: Any | None = None) -> dict[str, Any]:
    """
    在 **全部分批构图完成之后** 调用；会修改 ``cg.edges``、``G_p``/``G_a`` 并触发一次快照（由调用方 ``save_graph_snapshot`` 或本函数末尾）。

    Returns:
        遥测字典，写入日志与可选 ``cg.sense_class_telemetry_cumulative``。
    """
    cfg = get_edge_construction_config()
    if os.environ.get("MOSAIC_BUILD_EFFECTIVE", "").strip().lower() == "hash_only":
        if cfg.prerequisite_llm_enabled:
            _logger.info("MOSAIC_BUILD_EFFECTIVE=hash_only：跳过 LLM 先决边（E_A BGE 仍可按 [EDGE] 执行）")
        cfg = dataclasses.replace(cfg, prerequisite_llm_enabled=False)
    report: dict[str, Any] = {"edge_construction": {"enabled": cfg.enabled_summary()}}
    if not cfg.semantic_a_enabled and not cfg.prerequisite_llm_enabled:
        _logger.info("双图后处理：未启用 E_A BGE / LLM 先决（见 [EDGE]）")
        return report

    if cfg.semantic_a_enabled:
        report["edge_construction"]["semantic_bge"] = add_semantic_association_edges_bge(cg, cfg)
        report["edge_construction"]["minimum_connectivity"] = ensure_minimum_connectivity(cg, cfg)
    if cfg.prerequisite_llm_enabled:
        report["edge_construction"]["prerequisite_llm"] = add_llm_prerequisite_edges(
            cg, llm or getattr(cg, "_llm", None), cfg
        )

    try:
        from src.control.scoring import communities_louvain_ga

        cg._entity_communities = communities_louvain_ga(cg.G_a)
    except Exception:
        cg._entity_communities = {}

    tel = getattr(cg, "sense_class_telemetry_cumulative", None)
    if isinstance(tel, dict):
        tel["edge_construction"] = report["edge_construction"]

    try:
        cg.save_graph_snapshot()
    except Exception as e:
        _logger.warning("双图增强后快照失败: %s", e)

    return report
