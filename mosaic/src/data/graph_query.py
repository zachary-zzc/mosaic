"""ClassGraph query mixin: retrieval, neighbor expansion, keyword search, tag generation."""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict, deque
from string import Template
from typing import Any, Dict, List

import networkx as nx

from src.assist import (
    build_class_fragments,
    build_instance_fragments,
    calculate_tfidf_similarity,
    fetch_default_llm_model,
    keywords_process,
    serialize_instance,
    serialize_instance_kw,
    serialize_query,
)
from src.config_loader import (
    get_embedding_model_path,
    get_query_neighbor_mmr_lambda,
    get_query_neighbor_traversal_config,
    get_query_retrieval_config,
)
from src.data.dual_graph import (
    EDGE_LEG_ASSOCIATIVE,
    EDGE_LEG_PRAGMATIC,
    normalize_edge_leg,
)
from src.data.graph_base import _truncate_context
from src.llm.telemetry import llm_call_scope
from src.logger import setup_logger
from src.prompts_ch import PROMPT_TAGS_QUERY
from src.prompts_en import PROMPT_QUERY_CLASSES
from src.utils.constants import DEFAULT_TFIDF_VECTORIZER_PARAMS, TFIDF_KEYWORD_MAX_DF
from src.utils.io_utils import parse_llm_json_object, parse_llm_json_value

_logger = setup_logger("graph_query")


class ClassGraphQueryMixin:
    """Query-time methods: retrieval, neighbor expansion, keyword search."""

    def _neighbor_expansion_key_list(
        self, seeds: set[str], query: str | None = None
    ) -> list[str]:
        max_hops, max_extra, legs = get_query_neighbor_traversal_config()
        if max_hops <= 0 or max_extra <= 0 or not seeds:
            return []
        if query is not None:
            return self._neighbor_bfs_ranked(
                seeds, max_hops, max_extra, legs, query
            )
        return self._neighbor_bfs_keys(seeds, max_hops, max_extra, legs)

    def _build_instance_adjacency(self, allowed_legs: frozenset[str]) -> dict[str, set[str]]:
        """
        P-2：实例级无向邻接。与 P-1 一致：``self.edges`` 中 ``edge_leg`` 区分 E_P / E_A；
        ``functions`` 来自构图时共现边，视为 E_P。
        结果按 allowed_legs 缓存，避免同一查询中重复构建。
        """
        cache = getattr(self, "_adj_cache", None)
        if cache is not None and cache[0] == allowed_legs:
            return cache[1]
        adj: dict[str, set[str]] = defaultdict(set)

        def add_undirected(a: str, b: str) -> None:
            if not a or not b or a == b:
                return
            adj[a].add(b)
            adj[b].add(a)

        if EDGE_LEG_PRAGMATIC in allowed_legs:
            for class_node in self.graph.nodes:
                cid = getattr(class_node, "class_id", None) or ""
                if not cid:
                    continue
                for inst in getattr(class_node, "_instances", []) or []:
                    iid = inst.get("instance_id")
                    if iid is None:
                        continue
                    sk = self._instance_key(cid, iid)
                    for fn in inst.get("functions") or []:
                        ocid = fn.get("class_id")
                        oid = fn.get("instance_id")
                        if not ocid or oid is None:
                            continue
                        add_undirected(sk, self._instance_key(str(ocid), oid))

        for rec in self.edges or []:
            leg = normalize_edge_leg(rec.get("edge_leg"))
            if leg not in allowed_legs:
                continue
            conns = rec.get("connections") or []
            keys: list[str] = []
            for c in conns:
                ocid = c.get("class_id")
                oid = c.get("instance_id")
                if ocid and oid is not None:
                    keys.append(self._instance_key(str(ocid), oid))
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    add_undirected(keys[i], keys[j])

        result = dict(adj)
        self._adj_cache = (allowed_legs, result)
        return result

    def _neighbor_bfs_keys(
        self,
        seeds: set[str],
        max_hops: int,
        max_extra: int,
        allowed_legs: frozenset[str],
    ) -> list[str]:
        if max_hops <= 0 or max_extra <= 0 or not seeds:
            return []
        adj = self._build_instance_adjacency(allowed_legs)
        if not adj:
            return []

        visited: dict[str, int] = {s: 0 for s in seeds}
        q: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
        collected: list[str] = []

        while q and len(collected) < max_extra:
            u, d = q.popleft()
            if d >= max_hops:
                continue
            for v in adj.get(u, ()):
                if v in visited:
                    continue
                nd = d + 1
                visited[v] = nd
                q.append((v, nd))
                if v not in seeds and nd <= max_hops:
                    collected.append(v)
                    if len(collected) >= max_extra:
                        break

        return collected

    def _neighbor_bfs_ranked(
        self,
        seeds: set[str],
        max_hops: int,
        max_extra: int,
        allowed_legs: frozenset[str],
        query: str,
    ) -> list[str]:
        """
        Coverage-aware neighbor expansion for multi-hop retrieval.

        The old MMR strategy ranked neighbors by TF-IDF similarity to the full
        query, which re-introduced the same bias that already failed to retrieve
        these entities.  This method instead uses a two-signal score:

        1. **Query relevance** via edge content: the connecting edge's dialog
           text often bridges the semantic gap between seed and answer entity.
           Score += keyword overlap between the query and edges touching this
           candidate.
        2. **Content novelty**: neighbors whose text is *dissimilar* from
           already-selected seeds bring genuinely new information — exactly
           what multi-hop needs.  Score += (1 - max_sim_to_any_seed).

        At hop 2+, only "bridge" nodes (those matching >= 1 query keyword in
        either their own text or connecting edge text) are expanded further,
        keeping the search focused.
        """
        if max_hops <= 0 or max_extra <= 0 or not seeds:
            return []
        adj = self._build_instance_adjacency(allowed_legs)
        if not adj:
            return []

        # ── 0. Build instance_key → instance lookup ──
        key_to_inst: dict[str, dict] = {}
        for class_node in self.graph.nodes:
            cid = getattr(class_node, "class_id", None) or ""
            if not cid:
                continue
            for inst in getattr(class_node, "_instances", []) or []:
                iid = inst.get("instance_id")
                if iid is None:
                    continue
                key_to_inst[self._instance_key(cid, iid)] = inst

        def _inst_text(key: str) -> str:
            inst = key_to_inst.get(key)
            if inst is None:
                return ""
            parts = [tx for _ty, tx in build_instance_fragments(inst) if (tx or "").strip()]
            return "\n".join(parts) if parts else ""

        # ── 1. Extract query keywords ──
        _POSS_RE = re.compile(r"['\u2019]s$")
        _POSS_TEXT_RE = re.compile(r"['\u2019]s\b")
        q_lower = query.lower()
        q_words: set[str] = set()
        for w in q_lower.split():
            w = w.strip(".,!?()[]\"':;")
            w = _POSS_RE.sub("", w)  # "audrey's" → "audrey"
            if len(w) > 2:
                q_words.add(w)
        _stop = {"the", "and", "for", "are", "was", "were", "has", "have", "had",
                 "that", "this", "with", "from", "what", "which", "who", "how",
                 "does", "did", "not", "been", "but", "they", "them", "than",
                 "can", "her", "his", "she", "you", "all", "any", "some", "will"}
        q_words -= _stop

        def _norm_poss(text: str) -> str:
            """Strip possessive suffixes for consistent keyword matching."""
            return _POSS_TEXT_RE.sub("", text)

        # ── 2. Build edge content index (instance_key → list of edge texts) ──
        edge_texts_by_key: dict[str, list[str]] = defaultdict(list)
        for rec in self.edges or []:
            leg = normalize_edge_leg(rec.get("edge_leg"))
            if leg not in allowed_legs:
                continue
            content = (rec.get("content") or "").lower()
            if not content:
                continue
            conns = rec.get("connections") or []
            for c in conns:
                ocid = c.get("class_id")
                oid = c.get("instance_id")
                if ocid and oid is not None:
                    edge_texts_by_key[self._instance_key(str(ocid), oid)].append(content)

        def _kw_hits(key: str) -> int:
            """Count query keyword hits in instance text + edge texts."""
            txt = _norm_poss(_inst_text(key).lower())
            hits = sum(1 for kw in q_words if kw in txt)
            for etxt in edge_texts_by_key.get(key, []):
                hits += sum(1 for kw in q_words if kw in _norm_poss(etxt))
            return hits

        # ── 3. Query-guided DFS with similarity scoring ──
        # DFS prioritizes depth for multi-hop: follow high-similarity
        # chains deeply before exploring breadth, then backfill with
        # remaining candidates sorted by score.
        visited: dict[str, int] = {s: 0 for s in seeds}
        candidates: list[str] = []
        candidate_hop: dict[str, int] = {}
        candidate_score: dict[str, float] = {}

        def _score(ck: str) -> float:
            """Keyword overlap on edge text + instance text vs query."""
            edge_hits = 0
            for etxt in edge_texts_by_key.get(ck, []):
                edge_hits += sum(1 for kw in q_words if kw in _norm_poss(etxt))
            edge_sc = min(edge_hits / max(len(q_words), 1), 1.0)
            inst_txt = _norm_poss(_inst_text(ck).lower())
            inst_hits = sum(1 for kw in q_words if kw in inst_txt)
            inst_sc = min(inst_hits / max(len(q_words), 1), 1.0)
            return 0.5 * inst_sc + 0.5 * edge_sc

        # DFS stack: (node, depth). Use stack (LIFO) for depth-first.
        stack: list[tuple[str, int]] = [(s, 0) for s in seeds]
        while stack:
            u, d = stack.pop()
            if d >= max_hops:
                continue
            nd = d + 1
            # Score neighbors and sort so highest-score is pushed last (popped first)
            neighbors = []
            for v in adj.get(u, ()):
                if v in visited:
                    continue
                visited[v] = nd
                sc = _score(v)
                if v not in seeds:
                    candidates.append(v)
                    candidate_hop[v] = nd
                    candidate_score[v] = sc
                neighbors.append((v, nd, sc))
            # Push in ascending score order so highest is on top of stack
            neighbors.sort(key=lambda x: x[2])
            for v, vd, sc in neighbors:
                # Continue DFS if within hop limit and node has keyword relevance
                if vd < max_hops and (vd == 1 or sc > 0):
                    stack.append((v, vd))

        # ── 4. Score-ranked selection with depth bonus ──
        # DFS already explored deep chains; now select best candidates
        # with a depth bonus to favor multi-hop discoveries.
        if not candidates:
            return []
        if len(candidates) <= max_extra:
            # All fit — sort by score descending for context ordering
            candidates.sort(key=lambda ck: -candidate_score.get(ck, 0))
            _logger.debug(
                "DFS expansion: %d candidates (all selected, max_hops=%d)",
                len(candidates), max_hops,
            )
            return candidates

        # Rank: score + small depth bonus (deeper = explored for a reason)
        def _rank(ck: str) -> float:
            sc = candidate_score.get(ck, 0)
            hop = candidate_hop.get(ck, 1)
            depth_bonus = 0.05 * (hop - 1) if sc > 0 else 0
            return sc + depth_bonus

        scored = [(ck, _rank(ck)) for ck in candidates]
        scored.sort(key=lambda x: -x[1])
        result = [ck for ck, _ in scored[:max_extra]]

        hop_counts: dict[int, int] = {}
        for ck in result:
            h = candidate_hop.get(ck, 1)
            hop_counts[h] = hop_counts.get(h, 0) + 1
        hop_str = " ".join(f"hop{h}={n}" for h, n in sorted(hop_counts.items()))
        _logger.debug(
            "DFS expansion: %d candidates -> %d selected (%s)",
            len(candidates), len(result), hop_str,
        )
        return result

    def _query_neighbor_context_string(self, seeds: set[str], *, _precomputed_keys: list[str] | None = None) -> str:
        """在种子实例集合上按配置做图邻域扩展，序列化为补充检索片段。"""
        max_hops, max_extra, legs = get_query_neighbor_traversal_config()
        if max_hops <= 0 or max_extra <= 0:
            return ""
        expanded_keys = _precomputed_keys if _precomputed_keys is not None else self._neighbor_bfs_keys(seeds, max_hops, max_extra, legs)
        if not expanded_keys:
            return ""

        key_to_inst: dict[str, dict] = {}
        for class_node in self.graph.nodes:
            cid = getattr(class_node, "class_id", None) or ""
            if not cid:
                continue
            for inst in getattr(class_node, "_instances", []) or []:
                iid = inst.get("instance_id")
                if iid is None:
                    continue
                key_to_inst[self._instance_key(cid, iid)] = inst

        block: list[dict] = []
        for ek in expanded_keys:
            inst = key_to_inst.get(ek)
            if inst is not None:
                block.append(inst)

        if not block:
            return ""

        legs_note = ",".join(sorted(legs)) if legs else "P,A"
        header = (
            f"\n─── Graph neighbor context (DualGraph traversal, legs={legs_note}, "
            f"hops≤{max_hops}, max {len(block)} instances) ───\n"
        )
        _logger.debug(
            "P-2 neighbor expansion: seeds=%d expanded=%d (hops=%d max_extra=%d legs=%s)",
            len(seeds),
            len(block),
            max_hops,
            max_extra,
            legs_note,
        )
        return header + serialize_instance(block)

    #用llm的方式去搜索
    def _search_by_sub_llm(self,
                           query,
                           llm,
                           top_k_class=3,
                           top_k_instances=7,
                           ):
        self.selected_instance_keys.clear()
        self._retrieval_bge_accum = []
        keyword_matched_str = self.find_keyword_relevant_instance_tags(query)
        sensed_classes = self._sense_classes_by_llm(query, llm, top_k_class)
        class_ids = [
            str(c.get("class_id"))
            for c in sensed_classes.get("selected_classes", [])
            if c.get("class_id")
        ]
        _, instances_in_classes_str, retrieved_eids = self._fetch_instances_by_tfidf(
            query, top_k_instances, threshold=0.5, classes=sensed_classes
        )
        seeds = set(self.selected_instance_keys)
        expanded_keys = self._neighbor_expansion_key_list(seeds, query=query)
        neighbor_str = self._query_neighbor_context_string(seeds, _precomputed_keys=expanded_keys)
        combined_instances = "\n\n".join(
            s for s in (keyword_matched_str, instances_in_classes_str, neighbor_str) if (s or "").strip()
        )
        max_ctx = get_query_retrieval_config().max_context_chars
        combined_instances = _truncate_context(combined_instances, max_ctx)
        self.selected_instance_keys.clear()
        self._adj_cache = None
        trace: dict[str, Any] = {
            "schema_version": 2,
            "retrieval_method": "llm",
            "llm_selected_class_ids": class_ids,
            "retrieved_entity_ids": retrieved_eids,
            "tfidf_hits": {
                "count": len(retrieved_eids),
                "entity_ids": retrieved_eids,
            },
            "neighbor_expansion": {
                "count": len(expanded_keys),
                "entity_ids": [self._instance_key_to_entity_id(k) for k in expanded_keys],
            },
            "retrieval_fusion": {"stages": list(self._retrieval_bge_accum)},
            "prompt_chars": len(combined_instances),
        }
        return combined_instances, trace

    #用hash的方式去搜索
    def _search_by_sub_hash(self,
                            query,
                            top_k_class=10,
                            top_k_instances=15,
                            ):
        self.selected_instance_keys.clear()
        self._retrieval_bge_accum = []

        sensed_classes = self._sense_classes_by_tfidf(query, top_k_class, threshold=0.6, allow_below_threshold=True)
        tfidf_class_ids = [
            str(c.get("class_id"))
            for c in sensed_classes.get("selected_classes", [])
            if c.get("class_id")
        ]
        count_in_classes, instances_in_classes_str, e_class = self._fetch_instances_by_tfidf(
            query, top_k_instances, threshold=0.5, classes=sensed_classes
        )
        _logger.debug(f"第一阶段选择实例数量: {count_in_classes}")

        count_global, instances_global_str, e_global = self._fetch_instances_by_tfidf(
            query, top_k_instances, threshold=0.1
        )

        seeds = set(self.selected_instance_keys)
        expanded_keys = self._neighbor_expansion_key_list(seeds, query=query)
        neighbor_str = self._query_neighbor_context_string(seeds, _precomputed_keys=expanded_keys)
        combined_instances = "\n\n".join(
            s for s in (instances_global_str, instances_in_classes_str, neighbor_str) if (s or "").strip()
        )
        max_ctx = get_query_retrieval_config().max_context_chars
        combined_instances = _truncate_context(combined_instances, max_ctx)
        merged_eids: list[str] = []
        seen_e: set[str] = set()
        for e in e_global + e_class:
            if e not in seen_e:
                seen_e.add(e)
                merged_eids.append(e)
        self.selected_instance_keys.clear()
        self._adj_cache = None
        trace: dict[str, Any] = {
            "schema_version": 2,
            "retrieval_method": "hash",
            "tfidf_class_ids_stage1": tfidf_class_ids,
            "retrieved_entity_ids": merged_eids,
            "tfidf_hits": {
                "count": len(merged_eids),
                "entity_ids": merged_eids,
            },
            "neighbor_expansion": {
                "count": len(expanded_keys),
                "entity_ids": [self._instance_key_to_entity_id(k) for k in expanded_keys],
            },
            "retrieval_fusion": {"stages": list(self._retrieval_bge_accum)},
            "prompt_chars": len(combined_instances),
        }
        return combined_instances, trace

    def _sense_classes_by_llm(self, query, llm, top_k_class, *, _max_retries: int = 2):
        """
        查询最相关的多个类

        :param question: 问题文本
        :param classes_info: 类信息列表
        :param top_k: 返回的类数量
        :return: 选择的类列表
        """

        class_info = serialize_query(self.graph.nodes)
        # class_info=serialize(list(self.graph.nodes))
        query_classes_prompt = Template(PROMPT_QUERY_CLASSES).substitute(
            question=query,
            classes=class_info,
            top_k=top_k_class
        )

        for attempt in range(1, _max_retries + 1):
            #_logger.debug(f"QUERY_CLASSES_PROMPT: {query_classes_prompt}")
            with llm_call_scope("query.class_pick"):
                response = llm.invoke(query_classes_prompt)
            content = getattr(response, "content", None) or str(response)
            _logger.debug(f"QUERY_CLASSES_RESPONSE: %s", content)

            query_results = parse_llm_json_object(content)
            if query_results is not None:
                if "selected_classes" not in query_results:
                    query_results["selected_classes"] = []
                return query_results
            _logger.warning("解析类查询结果失败 (尝试 %d/%d): 非 JSON 或无法解析为对象", attempt, _max_retries)

        _logger.error("解析类查询结果失败: 经过 %d 次尝试仍无法解析", _max_retries)
        return {"error": "解析类查询结果失败", "selected_classes": []}

    def _fuse_class_scores_with_bge(
        self,
        query: str,
        class_max_scores: dict,
        class_nodes: list,
        class_info_map: dict,
    ) -> dict:
        """Fuse TF-IDF class scores with BGE embedding similarity (Option 3).

        Computes per-class BGE score by averaging cached instance embeddings,
        then applies ``(1-λ)*tfidf + λ*bge`` fusion.  Falls back to pure
        TF-IDF on any error.
        """
        import numpy as np
        import time

        cfg = get_query_retrieval_config()
        if cfg.bge_lambda <= 0:
            return class_max_scores
        cache = getattr(self, "_bge_embedding_cache", None)
        if not cache:
            _logger.debug("BGE class sensing: no embedding cache, skipping")
            return class_max_scores
        path = get_embedding_model_path()
        if not os.path.isdir(path):
            _logger.debug("BGE class sensing: embedding model dir missing, skipping")
            return class_max_scores

        try:
            t0 = time.perf_counter()
            from sentence_transformers import SentenceTransformer
            from src.retrieval.bge_query import minmax_01
            model = SentenceTransformer(path)

            q_emb = model.encode([query], convert_to_numpy=True)
            q_emb = q_emb / np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12)
            q_emb = np.asarray(q_emb, dtype=np.float32)

            class_bge_scores: dict[int, float] = {}
            for class_idx in class_max_scores:
                info = class_info_map.get(class_idx, {})
                class_id = info.get("class_id", "")
                node = info.get("node")
                if node is None:
                    continue
                # Collect cached embeddings for all instances in this class
                instances = getattr(node, "_instances", [])
                embs = []
                for inst in instances:
                    eid = f"{class_id}:{inst.get('instance_id', '')}"
                    if eid in cache:
                        embs.append(cache[eid])
                if not embs:
                    # Fallback: try class_id alone or skip
                    continue
                class_emb = np.mean(np.stack(embs, axis=0), axis=0, keepdims=True).astype(np.float32)
                class_emb = class_emb / np.maximum(np.linalg.norm(class_emb, axis=1, keepdims=True), 1e-12)
                sim = float((q_emb @ class_emb.T)[0, 0])
                class_bge_scores[class_idx] = sim

            if not class_bge_scores:
                _logger.debug("BGE class sensing: no class embeddings found in cache")
                return class_max_scores

            # Fuse only for classes that have BGE scores
            scored_idxs = sorted(class_bge_scores.keys())
            tfidf_vals = [class_max_scores[i] for i in scored_idxs]
            bge_vals = [class_bge_scores[i] for i in scored_idxs]
            t01 = minmax_01(tfidf_vals)
            b01 = minmax_01(bge_vals)
            lam = cfg.bge_lambda
            fused = dict(class_max_scores)
            for j, idx in enumerate(scored_idxs):
                fused[idx] = (1.0 - lam) * t01[j] + lam * b01[j]

            ms = (time.perf_counter() - t0) * 1000.0
            _logger.debug(
                "BGE class sensing: fused %d/%d classes (λ=%.2f) in %.1f ms",
                len(class_bge_scores), len(class_max_scores), lam, ms,
            )
            return fused
        except Exception as e:
            _logger.warning("BGE class sensing failed, using pure TF-IDF: %s", e)
            return class_max_scores

    def _sense_classes_by_tfidf(self, query, top_k_class, threshold,allow_below_threshold=True):
        """
        基于TF-IDF向量化的类感知方法

        Args:
            query: 查询字符串
            top_k_class: 返回的类数量上限
            threshold: 相似度阈值

        Returns:
            dict: 包含相关类信息的字典
        """
        # 1. 构建所有类的片段文本表示
        class_fragments = []  # 存储每个类片段的文本表示
        fragment_class_map = {}  # 存储片段索引到类索引的映射
        class_nodes = []  # 存储类节点
        class_info_map = {}  # 存储类索引到类信息的映射

        _logger.debug("开始构建类片段文本表示...")

        # 遍历图中的所有类节点，构建每个类的片段文本表示
        class_idx = 0
        fragment_count = 0

        for node in self.graph.nodes:
            # 检查节点是否为ClassNode
            if hasattr(node, 'class_id') and hasattr(node, 'class_name'):
                class_id = getattr(node, 'class_id', '')
                if not class_id:
                    continue

                class_name = getattr(node, 'class_name', '')
                class_node = node

                # 记录类节点和信息
                current_class_idx = len(class_nodes)
                class_nodes.append(class_node)

                class_info_map[current_class_idx] = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "node": class_node
                }

                # 构建类的片段
                fragments = build_class_fragments(class_node)

                for fragment_type, fragment_text in fragments:
                    if fragment_text.strip():  # 只添加非空片段
                        fragment_idx = len(class_fragments)
                        class_fragments.append(fragment_text)

                        # 记录片段到类的映射
                        fragment_class_map[fragment_idx] = {
                            'class_idx': current_class_idx,
                            'fragment_type': fragment_type,
                            'fragment_text': fragment_text
                        }

                        fragment_count += 1

                class_idx += 1

        if not class_fragments or not class_nodes:
            _logger.warning("未找到任何类节点或类片段")
            return {"selected_classes": []}

        _logger.debug(f"TF-IDF类感知: 总共收集到 {len(class_nodes)} 个类，{len(class_fragments)} 个片段")

        # 2. 使用TF-IDF进行向量化和相似度计算
        similarities, vectorizer, tfidf_matrix = calculate_tfidf_similarity(
            query,
            class_fragments,
            vectorizer_params=dict(DEFAULT_TFIDF_VECTORIZER_PARAMS)
        )

        # 3. 为每个类找出最高相似度的片段
        class_max_scores = {}  # 类索引 -> 最高相似度
        class_best_fragments = {}  # 类索引 -> 最佳片段信息

        for fragment_idx, score in enumerate(similarities):
            fragment_info = fragment_class_map.get(fragment_idx)
            if not fragment_info:
                continue

            class_idx = fragment_info['class_idx']

            # 如果这是该类的更高相似度片段，更新记录
            if class_idx not in class_max_scores or score > class_max_scores[class_idx]:
                class_max_scores[class_idx] = score
                class_best_fragments[class_idx] = {
                    'fragment_type': fragment_info['fragment_type'],
                    'fragment_text': fragment_info['fragment_text']
                }

        if not class_max_scores:
            _logger.warning("没有找到任何匹配的类片段")
            return {"selected_classes": []}

        # 3b. BGE embedding fusion for class sensing (Option 3)
        class_max_scores = self._fuse_class_scores_with_bge(
            query, class_max_scores, class_nodes, class_info_map,
        )

        # 4. 将类按最高相似度排序
        class_scores = list(class_max_scores.items())
        class_scores.sort(key=lambda x: x[1], reverse=True)

        _logger.debug(f"类相似度统计: 共{len(class_scores)}个类有匹配片段")

        # 5. 按阈值过滤并选择类
        above_threshold = [(idx, score) for idx, score in class_scores if score >= threshold]
        below_threshold = [(idx, score) for idx, score in class_scores if score < threshold]

        _logger.debug(
            f"TF-IDF类感知: 高于阈值({threshold})的类数量: {len(above_threshold)}, 低于阈值的类数量: {len(below_threshold)}")

        # 6. 选择策略
        if top_k_class <= 0:
            return {"selected_classes": []}

        if len(above_threshold) >= top_k_class:
            # 情况1: 高于阈值的类足够
            selected_class_scores = above_threshold[:top_k_class]
            _logger.debug(f"选择前{top_k_class}个高于阈值的类")
        else:
            if allow_below_threshold:
                # 情况2: 高于阈值的类不足，用低于阈值但分数最高的补足
                above_threshold.sort(key=lambda x: x[1], reverse=True)
                below_threshold.sort(key=lambda x: x[1], reverse=True)

                remaining_slots = top_k_class - len(above_threshold)
                actual_slots = min(remaining_slots, len(below_threshold))
                selected_class_scores = above_threshold + below_threshold[:actual_slots]

                _logger.debug(f"补充{actual_slots}个低于阈值但分数最高的类")
            else:
                # 情况3: 不允许使用低于阈值的类，只返回高于阈值的类
                selected_class_scores = above_threshold
                _logger.debug(f"不允许使用低于阈值类，选择{len(selected_class_scores)}个高于阈值的类")

        # 7. 构建返回结果
        selected_classes_info = []
        for class_idx, score in selected_class_scores:
            class_info = class_info_map.get(class_idx, {})
            best_fragment = class_best_fragments.get(class_idx, {})

            selected_class_info = {
                "class_id": class_info.get('class_id', ''),
                "class_name": class_info.get('class_name', ''),
                "score": float(score),
                "matched_fragment": {
                    'type': best_fragment.get('fragment_type', 'unknown'),
                    'content': best_fragment.get('fragment_text', ''),
                    'similarity_score': float(score)
                }
            }
            selected_classes_info.append(selected_class_info)

        # 记录最终选择结果
        class_names = [info['class_name'] for info in selected_classes_info]
        scores = [info['score'] for info in selected_classes_info]
        _logger.debug(f"TF-IDF最终选择的{len(selected_classes_info)}个类: {class_names}")
        _logger.debug(f"对应TF-IDF相似度分数: {scores}")

        return {"selected_classes": selected_classes_info}

    def _maybe_fuse_instance_scores_with_bge(
        self,
        query: str,
        all_instances: list,
        instance_max_scores: dict[int, float],
        instance_class_map: dict[int, dict] | None = None,
    ) -> tuple[dict[int, float], dict[str, Any]]:
        from src.retrieval.bge_query import minmax_01, query_instance_cosine_similarities
        import numpy as np
        import time

        cfg = get_query_retrieval_config()
        aux: dict[str, Any] = {
            "bge_applied": False,
            "bge_lambda": cfg.bge_lambda,
            "bge_ms": 0.0,
            "bge_pool_size": 0,
            "bge_cache_used": False,
        }
        if cfg.bge_lambda <= 0 or not instance_max_scores:
            return instance_max_scores, aux
        path = get_embedding_model_path()
        if not os.path.isdir(path):
            _logger.debug("query BGE: 嵌入模型目录不存在，跳过融合")
            return instance_max_scores, aux
        idxs = sorted(instance_max_scores.keys(), key=lambda i: instance_max_scores[i], reverse=True)
        cap = min(cfg.bge_max_encode_instances, len(idxs))
        pool_list = idxs[:cap]

        # 尝试使用构图阶段的预计算嵌入缓存
        cache = getattr(self, "_bge_embedding_cache", None)
        if cache and instance_class_map is not None:
            try:
                t0 = time.perf_counter()
                # 查找缓存命中的实例
                cached_embs: list[np.ndarray] = []
                cached_indices: list[int] = []
                uncached_indices: list[int] = []
                for inst_idx in pool_list:
                    inst = all_instances[inst_idx]
                    cls_info = instance_class_map.get(inst_idx)
                    if cls_info:
                        eid = f"{cls_info['class_id']}:{inst.get('instance_id', '')}"
                    else:
                        eid = None
                    if eid and eid in cache:
                        cached_embs.append(cache[eid])
                        cached_indices.append(inst_idx)
                    else:
                        uncached_indices.append(inst_idx)

                if cached_indices:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(path)
                    q_emb = model.encode([query], convert_to_numpy=True)
                    q_emb = q_emb / np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12)
                    q_emb = np.asarray(q_emb, dtype=np.float32)

                    # 对缓存命中的实例直接用矩阵乘法
                    doc_mat = np.stack(cached_embs, axis=0).astype(np.float32)
                    cached_sims = (q_emb @ doc_mat.T)[0].tolist()

                    # 对缓存未命中的实例退回到编码
                    uncached_sims: list[float] = []
                    if uncached_indices:
                        uncached_texts: list[str] = []
                        for inst_idx in uncached_indices:
                            inst = all_instances[inst_idx]
                            parts = [tx for _ty, tx in build_instance_fragments(inst) if (tx or "").strip()]
                            uncached_texts.append("\n".join(parts) if parts else " ")
                        uc_emb = model.encode(list(uncached_texts), convert_to_numpy=True)
                        uc_emb = uc_emb / np.maximum(np.linalg.norm(uc_emb, axis=1, keepdims=True), 1e-12)
                        uc_emb = np.asarray(uc_emb, dtype=np.float32)
                        uncached_sims = (q_emb @ uc_emb.T)[0].tolist()

                    # 按 pool_list 原始顺序重组相似度
                    bge_raw: list[float] = []
                    ci, ui = 0, 0
                    for inst_idx in pool_list:
                        if ci < len(cached_indices) and inst_idx == cached_indices[ci]:
                            bge_raw.append(cached_sims[ci])
                            ci += 1
                        else:
                            bge_raw.append(uncached_sims[ui])
                            ui += 1

                    ms = (time.perf_counter() - t0) * 1000.0
                    aux["bge_ms"] = round(ms, 3)
                    aux["bge_applied"] = True
                    aux["bge_pool_size"] = len(pool_list)
                    aux["bge_cache_used"] = True
                    aux["bge_cache_hits"] = len(cached_indices)
                    aux["bge_cache_misses"] = len(uncached_indices)
                    _logger.debug(
                        "query BGE: 缓存命中 %d/%d 实例",
                        len(cached_indices), len(pool_list),
                    )
                    tfidf_pool = [instance_max_scores[i] for i in pool_list]
                    t01 = minmax_01(tfidf_pool)
                    b01 = minmax_01(bge_raw)
                    lam = cfg.bge_lambda
                    fused_pool = [(1.0 - lam) * t01[j] + lam * b01[j] for j in range(len(pool_list))]
                    new_scores = dict(instance_max_scores)
                    for j, inst_idx in enumerate(pool_list):
                        new_scores[inst_idx] = fused_pool[j]
                    return new_scores, aux
            except Exception as e:
                _logger.warning("query BGE 缓存路径失败，退回完整编码: %s", e)

        # 退回路径：无缓存或缓存查找失败，使用原始完整编码
        texts: list[str] = []
        for inst_idx in pool_list:
            inst = all_instances[inst_idx]
            parts = [tx for _ty, tx in build_instance_fragments(inst) if (tx or "").strip()]
            texts.append("\n".join(parts) if parts else " ")
        try:
            bge_raw, ms = query_instance_cosine_similarities(query, texts, path)
            aux["bge_ms"] = round(ms, 3)
            aux["bge_applied"] = True
            aux["bge_pool_size"] = len(pool_list)
            tfidf_pool = [instance_max_scores[i] for i in pool_list]
            t01 = minmax_01(tfidf_pool)
            b01 = minmax_01(bge_raw)
            lam = cfg.bge_lambda
            fused_pool = [(1.0 - lam) * t01[j] + lam * b01[j] for j in range(len(pool_list))]
            new_scores = dict(instance_max_scores)
            for j, inst_idx in enumerate(pool_list):
                new_scores[inst_idx] = fused_pool[j]
            return new_scores, aux
        except Exception as e:
            _logger.warning("query BGE 融合失败，使用纯 TF-IDF: %s", e)
            return instance_max_scores, aux

    def _fetch_instances_by_tfidf(self, query, top_k_instances, threshold, classes=None) :
        """
        基于TF-IDF向量化的实例检索方法

        Args:
            query: 查询字符串
            top_k_instances: 返回的实例数量上限
            threshold: 相似度阈值
            classes: 从_sense_classes_by_tfidf返回的相关类列表，如果为None则从所有类中收集

        Returns:
            tuple[int, str]: (选中实例数量, 序列化后的相关实例文本)
        """
        # 1. 从相关类中收集所有实例及其片段
        all_instances = []  # 存储实例对象
        instance_documents = []  # 存储每个片段的文本表示
        fragment_instance_map = {}  # 存储片段索引到实例索引的映射
        instance_class_map = {}  # 存储实例索引到所属类的映射
        instance_keys_map = {}  # 存储实例索引到唯一标识的映射

        if classes is None:
            _logger.debug(f"开始从所有类中收集实例片段...")
            class_count = 0
            instance_count = 0
            fragment_count = 0

            # 遍历图谱中的所有节点，找到ClassNode
            for node in self.graph.nodes:
                # 检查节点是否为ClassNode
                if hasattr(node, 'class_id') and hasattr(node, '_instances'):
                    class_id = getattr(node, 'class_id', None)
                    if class_id is None:
                        continue

                    class_count += 1
                    target_class_node = node

                    # 获取该类的所有实例
                    class_instances = getattr(target_class_node, '_instances', [])

                    for instance in class_instances:
                        instance_id = instance.get('instance_id', '')
                        instance_key = f"{class_id}_{instance_id}"

                        # 如果实例已在第一步中被选中，则跳过
                        if instance_key in self.selected_instance_keys:
                            _logger.debug(f"跳过已选中的实例: {instance_key}")
                            continue

                        instance_idx = len(all_instances)
                        all_instances.append(instance)

                        # 记录实例的类信息
                        instance_class_map[instance_idx] = {
                            'class_id': class_id,
                            'class_name': getattr(target_class_node, 'name', ''),
                            'class_node': target_class_node
                        }
                        # 记录实例唯一标识
                        instance_keys_map[instance_idx] = instance_key

                        instance_count += 1

                        # 为实例的每个片段构建文本表示
                        fragments = build_instance_fragments(instance)
                       # _logger.debug(f"实例的片段文本表示{fragments}")

                        for fragment_type, fragment_text in fragments:
                            if fragment_text.strip():  # 只添加非空片段
                                fragment_idx = len(instance_documents)
                                instance_documents.append(fragment_text)

                                # 记录片段到实例的映射
                                fragment_instance_map[fragment_idx] = {
                                    'instance_idx': instance_idx,
                                    'fragment_type': fragment_type,
                                    'fragment_text': fragment_text
                                }

                                fragment_count += 1

            _logger.debug(
                f"从 {class_count} 个类中收集到 {instance_count} 个实例，共 {fragment_count} 个片段，"
                f"跳过了 {len(self.selected_instance_keys)} 个已选中的实例")
        else:
            _logger.debug(f"开始从指定类中收集实例片段...")

            # 遍历选中的类，从graph中查找对应的ClassNode并收集其实例
            for class_info in classes.get("selected_classes", []):
                class_id = class_info.get("class_id")
                target_class_node = None

                # 查找类节点
                for class_node in self.graph.nodes:
                    if getattr(class_node, 'class_id', None) == class_id:
                        target_class_node = class_node
                        break

                if target_class_node is None:
                    _logger.warning(f"未找到类ID为 {class_id} 的类节点")
                    continue

                # 获取该类的所有实例
                class_instances = getattr(target_class_node, '_instances', [])
                _logger.debug("当前类下面的实例: %s", class_instances)

                for instance in class_instances:
                    instance_idx = len(all_instances)
                    all_instances.append(instance)

                    # 获取实例的唯一标识
                    instance_id = instance.get('instance_id', '')
                    instance_key = f"{class_id}_{instance_id}"

                    # 如果实例已在第一步中被选中，则跳过
                    if instance_key in self.selected_instance_keys:
                        _logger.debug(f"跳过已选中的实例: {instance_key}")
                        continue

                    # 记录实例的类信息
                    instance_class_map[instance_idx] = {
                        'class_id': class_id,
                        'class_name': getattr(target_class_node, 'name', ''),
                        'class_node': target_class_node
                    }

                    # 记录实例唯一标识
                    instance_keys_map[instance_idx] = instance_key

                    _logger.debug("用于去构建片段的实例: %s", instance)
                    # 为实例的每个片段构建文本表示
                    fragments = build_instance_fragments(instance)

                    _logger.debug("当前实例的片段: %s", fragments)
                    for fragment_type, fragment_text in fragments:
                       # if fragment_text.strip():  # 只添加非空片段
                        fragment_idx = len(instance_documents)
                        instance_documents.append(fragment_text)

                        # 记录片段到实例的映射
                        fragment_instance_map[fragment_idx] = {
                            'instance_idx': instance_idx,
                            'fragment_type': fragment_type,
                            'fragment_text': fragment_text
                        }

                _logger.debug("从类 %s 获取到 %d 个实例，共 %d 个片段", class_id, len(class_instances), len(instance_documents))

        if not all_instances or not instance_documents:
            _logger.warning("未找到任何实例或实例片段")
            return 0, "", []

        _logger.debug(f"TF-IDF实例检索: 总共收集到 {len(all_instances)} 个实例，{len(instance_documents)} 个片段")

        # 2. 使用TF-IDF进行向量化和相似度计算
        similarities, vectorizer, tfidf_matrix = calculate_tfidf_similarity(query, instance_documents)

        # 3. 为每个实例找出最高相似度的片段
        instance_max_scores = {}  # 实例索引 -> 最高相似度
        instance_best_fragments = {}  # 实例索引 -> 最佳片段信息

        for fragment_idx, score in enumerate(similarities):
            fragment_info = fragment_instance_map.get(fragment_idx)
            if not fragment_info:
                continue

            instance_idx = fragment_info['instance_idx']

            # 如果这是该实例的更高相似度片段，更新记录
            if instance_idx not in instance_max_scores or score > instance_max_scores[instance_idx]:
                instance_max_scores[instance_idx] = score
                instance_best_fragments[instance_idx] = {
                    'fragment_type': fragment_info['fragment_type'],
                    'fragment_text': fragment_info['fragment_text']
                }

        if not instance_max_scores:
            _logger.warning("没有找到任何匹配的实例片段")
            return 0, "", []

        instance_max_scores, bge_aux = self._maybe_fuse_instance_scores_with_bge(
            query, all_instances, instance_max_scores, instance_class_map
        )
        self._retrieval_bge_accum.append(bge_aux)

        # 4. 将实例按最高相似度排序
        instance_scores = list(instance_max_scores.items())
        instance_scores.sort(key=lambda x: x[1], reverse=True)

        _logger.debug(f"实例相似度统计: 共{len(instance_scores)}个实例有匹配片段")

        # 5. 按阈值过滤并选择实例
        above_threshold = [(idx, score) for idx, score in instance_scores if score >= threshold]
        below_threshold = [(idx, score) for idx, score in instance_scores if score < threshold]

        log_msg = f"相似度统计: 高于阈值({threshold})实例数: {len(above_threshold)}, 低于阈值实例数: {len(below_threshold)}"
        _logger.debug(log_msg)

        # 6. 根据新逻辑选择实例
        if top_k_instances <= 0:
            return 0, "", []

        above_count = len(above_threshold)
        selected_instance_scores = []
        if len(above_threshold) >= top_k_instances:
            # 情况1: 高于阈值的实例足够
            selected_instance_scores = above_threshold[:top_k_instances]
            _logger.debug(f"高于阈值实例充足，直接返回前{len(selected_instance_scores)}个")
        else:
            # 情况2: 高于阈值的实例不足
            num_needed = top_k_instances - above_count
            _logger.debug(f"高于阈值实例只有{above_count}个，需要从低于阈值实例中补充{num_needed}个")

            # 计算可以从低于阈值实例中获取的最大数量
            available_below = len(below_threshold)

            if available_below >= num_needed:
                # 有足够的低于阈值实例来补足
                supplementary_instances = below_threshold[:num_needed]
                selected_instance_scores = above_threshold + supplementary_instances
                _logger.debug(f"低于阈值实例充足，用{len(supplementary_instances)}个补足到{top_k_instances}个")
            else:
                # 低于阈值实例也不足，只能返回尽可能多的实例
                supplementary_instances = below_threshold[:available_below]
                selected_instance_scores = above_threshold + supplementary_instances
                _logger.warning(
                    f"低于阈值实例也不足，只能返回{len(selected_instance_scores)}个实例，而不是期望的{top_k_instances}个")

        _logger.debug(f"最终选中的实例数量: {len(selected_instance_scores)}")

        # 7. 创建清理后的实例副本
        cleaned_instances = []
        for instance_idx, score in selected_instance_scores:
            original_instance = all_instances[instance_idx]

            # 获取实例的类信息
            class_info = instance_class_map.get(instance_idx, {})

            # 获取最佳片段信息
            best_fragment = instance_best_fragments.get(instance_idx, {})

            # 获取实例的唯一标识并记录到全局集合
            instance_key = instance_keys_map.get(instance_idx)
            if instance_key:
                self.selected_instance_keys.add(instance_key)

            # 创建不包含敏感字段的实例副本
            cleaned_instance = {key: value for key, value in original_instance.items()
                                if key not in ['message_labels']}

            # 添加TF-IDF相似度分数和匹配片段信息
            cleaned_instance['similarity_score'] = float(score)
            cleaned_instance['matched_fragment'] = {
                'type': best_fragment.get('fragment_type', 'unknown'),
                'content': best_fragment.get('fragment_text', ''),
                'similarity_score': float(score)
            }

            # 添加实例所属的类信息
            cleaned_instance['class_info'] = {
                'class_id': class_info.get('class_id', ''),
                'class_name': class_info.get('class_name', ''),
                'fully_covering': False,
                'stage': 0
            }

            cleaned_instances.append(cleaned_instance)

            # 记录日志
            if classes is None:
                _logger.debug(f"选择实例 {instance_idx}: 最高相似度={score:.4f}, "
                             f"匹配片段类型={best_fragment.get('fragment_type', 'unknown')}, "
                             f"类={class_info.get('class_id', 'unknown')}")
            else:
                _logger.debug(f"选择实例 {instance_idx}: 相似度={score:.4f}, "
                             f"类={class_info.get('class_id', 'unknown')}, "
                             f"匹配片段类型={best_fragment.get('fragment_type', 'unknown')}")

        ordered_entity_ids: list[str] = []
        for instance_idx, _score in selected_instance_scores:
            ik = instance_keys_map.get(instance_idx)
            if ik:
                ordered_entity_ids.append(self._instance_key_to_entity_id(ik))

        return len(cleaned_instances), serialize_instance(cleaned_instances), ordered_entity_ids

    def find_keyword_relevant_instance_tags(self, query: str) -> str:
        tags_data = self.tags
        excluded_count = 0
        all_instance_info = []  # 存储所有实例的信息

        for tag_item in tags_data:
            class_id = tag_item.get('class_id')
            instance_id = tag_item.get('instance_id')
            instance_key = f"{class_id}_{instance_id}"  # 创建唯一键

            # 检查是否需要排除此实例
            if instance_key in self.selected_instance_keys:
                excluded_count += 1
                _logger.debug(f"排除实例（已在前两步选中）: {instance_key}")
                continue

            instance_info = {
                'class_id': class_id,
                'instance_id': instance_id,
                'keywords': tag_item.get('keywords', [])
            }
            all_instance_info.append(instance_info)

        find_keywords_prompt = Template(PROMPT_TAGS_QUERY).substitute({
            "QUESTION": query,
            "TAGS": all_instance_info
        })
        _logger.debug(f"keywords_prompt: %s", find_keywords_prompt)
        with llm_call_scope("query.keyword_tags"):
            response = self._llm.invoke(find_keywords_prompt)
        kw_content = getattr(response, "content", None) or str(response)
        _logger.debug(f"keywords_prompt response: %s", kw_content)
        parsed_kw = parse_llm_json_value(kw_content)
        if isinstance(parsed_kw, list):
            instances_data = parsed_kw
        elif isinstance(parsed_kw, dict) and isinstance(parsed_kw.get("instances"), list):
            instances_data = parsed_kw["instances"]
        else:
            _logger.warning(
                "PROMPT_TAGS_QUERY 回复无法解析为 JSON 数组，跳过关键词实例补充。原文前 500 字: %r",
                kw_content[:500],
            )
            instances_data = []

        # 将选中的实例添加到self.selected_instance_keys中
        added_instance_count = 0
        for instance_info in instances_data:
            class_id = instance_info.get('class_id')
            instance_id = instance_info.get('instance_id')

            if class_id and instance_id:
                instance_key = f"{class_id}_{instance_id}"

                # 检查是否已经存在于selected_instance_keys中
                if instance_key not in self.selected_instance_keys:
                    self.selected_instance_keys.add(instance_key)
                    added_instance_count += 1
                    _logger.debug(f"添加实例到已选集合: {instance_key}")
                else:
                    _logger.debug(f"实例已存在于已选集合: {instance_key}")

        _logger.debug(f"已将 {added_instance_count} 个实例添加到已选实例集合中")

        relevant_instances = []
        for instance_info in instances_data:
            class_id = instance_info.get('class_id')
            instance_id = instance_info.get('instance_id')
            found_instance = self._find_instance_by_ids(class_id, instance_id)
            if found_instance:
                relevant_instances.append(found_instance)
            else:
                _logger.warning(f"Instance not found: class_id={class_id}, instance_id={instance_id}")

        _logger.debug(f"Found {len(relevant_instances)} relevant instances")
        return serialize_instance(relevant_instances)

    def find_keyword_coverage_instances_with_tfidf(self, query_keywords, similarity_threshold=0.1,
                                              single_top_k=7, combo_top_k=3) -> str:
        """
        基于TF-IDF相似度的关键词覆盖实例检索方法

        Args:
            query_keywords: 查询关键词列表
            similarity_threshold: 关键词相似度阈值，默认0.1
            single_top_k: 第一阶段返回的单个实例数量，默认7
            combo_top_k: 第二阶段返回的组合数量，默认3

        Returns:
            str: 序列化的相关实例列表
        """
        # 1. 读取tags文件数据
        tags_data = self.tags
        _logger.debug(f"成功加载 {len(tags_data)} 个tags数据")

        # 2. 处理查询关键词
        if isinstance(query_keywords, str):
            query_keywords = [kw.strip() for kw in query_keywords.split(',') if kw.strip()]
        elif isinstance(query_keywords, list):
            query_keywords = [kw.strip() for kw in query_keywords if kw]
        else:
            _logger.error("query_keywords应为关键词列表或逗号分隔的字符串")
            return serialize_instance([])

        if not query_keywords:
            _logger.warning("查询关键词列表为空")
            return serialize_instance([])

        _logger.debug(f"查询关键词: {query_keywords}")
        _logger.debug(f"相似度阈值: {similarity_threshold}")
        _logger.debug(f"单个实例返回数量: {single_top_k}")
        _logger.debug(f"组合返回数量: {combo_top_k}")

        # 3. 准备数据结构和TF-IDF向量化
        instance_keywords_map = {}  # 组合键 -> 关键词列表
        all_keywords_set = set()  # 所有实例中出现的关键词集合
        instance_info_map = {}  # 组合键 -> 实例完整信息
        instance_key_to_ids = {}  # 存储实例键到class_id和instance_id的映射

        # 使用class_id和instance_id的组合作为唯一键
        excluded_count = 0  # 记录排除的实例数量

        for tag_item in tags_data:
            class_id = tag_item.get('class_id')
            instance_id = tag_item.get('instance_id')
            instance_key = f"{class_id}_{instance_id}"  # 创建唯一键

            # 检查是否需要排除此实例
            if instance_key in self.selected_instance_keys:
                excluded_count += 1
                _logger.debug(f"排除实例（已在前两步选中）: {instance_key}")
                continue

            instance_info = {
                'class_id': class_id,
                'instance_id': instance_id,
                'keywords': tag_item.get('keywords', [])
            }

            keywords = tag_item.get('keywords', [])
            instance_keywords_map[instance_key] = keywords
            instance_info_map[instance_key] = instance_info
            instance_key_to_ids[instance_key] = {'class_id': class_id, 'instance_id': instance_id}

            for keyword in keywords:
                all_keywords_set.add(keyword)

        _logger.debug(f"总计 {len(all_keywords_set)} 个唯一关键词，排除 {excluded_count} 个已选实例")

        if not all_keywords_set:
            _logger.warning("没有找到任何关键词")
            return serialize_instance([])

        # 4. 计算TF-IDF相似度矩阵
        # 准备文档：每个关键词作为一个文档
        keyword_docs = list(all_keywords_set)
        query_docs = query_keywords

        vectorizer_params = dict(DEFAULT_TFIDF_VECTORIZER_PARAMS)
        vectorizer_params["max_df"] = TFIDF_KEYWORD_MAX_DF

        # 计算每个查询关键词与所有关键词的相似度
        similarity_matrix = []
        for query in query_docs:
            # 注意：这里我们使用calculate_tfidf_similarity函数，但该函数是计算单个查询与多个文档的相似度
            # 这里我们需要计算多个查询，所以需要为每个查询调用一次
            similarities, vectorizer, tfidf_matrix = calculate_tfidf_similarity(
                query,
                keyword_docs,
                vectorizer_params=vectorizer_params
            )
            similarity_matrix.append(similarities)

        # 转换为numpy数组以便后续计算
        import numpy as np
        similarity_matrix = np.array(similarity_matrix)
        _logger.debug(f"TF-IDF向量维度: {tfidf_matrix.shape}")
        _logger.debug(f"相似度矩阵形状: {similarity_matrix.shape}")

        # 6. 构建关键词到索引的映射
        keyword_to_index = {keyword: idx for idx, keyword in enumerate(keyword_docs)}

        # 7. 为每个查询关键词找到相似度高于阈值的关键词
        query_to_similar_keywords = {}

        for query_idx, query_keyword in enumerate(query_keywords):
            similar_keywords = []

            # 找到相似度高于阈值的关键词
            for keyword_idx, similarity in enumerate(similarity_matrix[query_idx]):
                if similarity >= similarity_threshold:
                    similar_keywords.append({
                        'keyword': keyword_docs[keyword_idx],
                        'similarity': similarity
                    })

            # 按相似度降序排序
            similar_keywords.sort(key=lambda x: x['similarity'], reverse=True)

            query_to_similar_keywords[query_keyword] = similar_keywords

            _logger.debug(f"查询关键词 '{query_keyword}' 找到 {len(similar_keywords)} 个相似关键词")
            if similar_keywords:
                top_similar = similar_keywords[:3]
                similarity_str = ', '.join([f"{item['keyword']}:{item['similarity']:.3f}" for item in top_similar])
                _logger.debug(f"  Top 3: {similarity_str}")

        # 8. 为每个查询关键词找到相关的实例
        query_to_instances = {}

        for query_keyword, similar_items in query_to_similar_keywords.items():
            instances_for_query = set()

            for similar_item in similar_items:
                keyword = similar_item['keyword']
                similarity = similar_item['similarity']

                # 找到包含这个关键词的实例
                for instance_key, keywords in instance_keywords_map.items():
                    if keyword in keywords:
                        instances_for_query.add((instance_key, keyword, similarity))

            query_to_instances[query_keyword] = instances_for_query
            _logger.debug(f"查询关键词 '{query_keyword}': 找到 {len(instances_for_query)} 个相关实例")

        # 9. 构建覆盖映射：每个实例覆盖哪些查询关键词
        instance_to_queries = {}
        instance_to_queries_details = {}

        for query_keyword, instance_set in query_to_instances.items():
            for instance_key, matched_keyword, similarity in instance_set:
                if instance_key not in instance_to_queries:
                    instance_to_queries[instance_key] = set()
                    instance_to_queries_details[instance_key] = []

                instance_to_queries[instance_key].add(query_keyword)
                instance_to_queries_details[instance_key].append({
                    'query_keyword': query_keyword,
                    'matched_keyword': matched_keyword,
                    'similarity': similarity
                })

        _logger.debug(f"构建完成: {len(instance_to_queries)} 个实例至少匹配到一个查询关键词")

        # 10. 跳过第一阶段，直接进入第二阶段
        _logger.debug("跳过第一阶段，直接进入第二阶段：贪心算法寻找最小覆盖组合")

        # 11. 第二阶段：贪心算法寻找覆盖所有查询关键词的实例组合
        all_combinations = []  # 存储所有找到的组合
        stage1_instance_keys_set = set()  # 第一阶段为空，因为我们跳过了第一阶段

        # 为了找到多个组合，我们需要多次运行贪心算法
        for combo_idx in range(combo_top_k):
            _logger.debug(f"开始寻找第 {combo_idx + 1} 个组合...")

            # 复制原始数据，用于当前组合的构建
            current_remaining_queries = set(query_keywords)
            current_selected_instances = []
            current_covered_queries = set()

            # 预处理：计算每个实例能覆盖的新查询数量
            current_instance_coverage_map = {}

            for instance_key, queries in instance_to_queries.items():
                # 如果实例已经被之前的组合选中，跳过
                if any(instance_key in combo['instances'] for combo in all_combinations):
                    continue

                # 排除第一阶段用到的实例
                if instance_key in stage1_instance_keys_set:
                    continue

                new_coverage = queries.intersection(current_remaining_queries)
                if new_coverage:
                    current_instance_coverage_map[instance_key] = {
                        'coverage_count': len(new_coverage),
                        'queries': new_coverage
                    }

            iteration = 0
            max_iterations = 5

            while current_remaining_queries and current_instance_coverage_map and iteration < max_iterations:
                iteration += 1

                # 找出能覆盖最多剩余查询的实例
                best_instance = None
                best_coverage = 0
                best_queries = set()

                for instance_key, coverage_data in current_instance_coverage_map.items():
                    if coverage_data['coverage_count'] > best_coverage:
                        best_coverage = coverage_data['coverage_count']
                        best_instance = instance_key
                        best_queries = coverage_data['queries']
                    # 如果覆盖数量相同，选择总关键词数更少的实例
                    elif coverage_data['coverage_count'] == best_coverage and best_instance:
                        if len(instance_keywords_map[instance_key]) < len(instance_keywords_map[best_instance]):
                            best_instance = instance_key
                            best_queries = coverage_data['queries']

                if not best_instance or best_coverage == 0:
                    break

                # 选择最佳实例
                current_selected_instances.append(best_instance)
                _logger.debug(f"第{iteration}轮贪心选择: 选择实例 {best_instance}, 覆盖 {best_coverage} 个新查询关键词")

                # 记录覆盖详情
                coverage_details = []
                for query in best_queries:
                    for detail in instance_to_queries_details[best_instance]:
                        if detail['query_keyword'] == query:
                            coverage_details.append(f"{query}->{detail['matched_keyword']}:{detail['similarity']:.3f}")
                            break

                if coverage_details:
                    _logger.debug(f"  覆盖详情: {coverage_details}")

                # 更新已覆盖的查询
                current_covered_queries.update(best_queries)
                current_remaining_queries.difference_update(best_queries)

                # 更新其他实例的覆盖率
                for instance_key in list(current_instance_coverage_map.keys()):
                    if instance_key == best_instance:
                        current_instance_coverage_map.pop(instance_key)
                        continue

                    instance_queries = instance_to_queries[instance_key]
                    new_coverage = instance_queries.intersection(current_remaining_queries)

                    if new_coverage:
                        current_instance_coverage_map[instance_key] = {
                            'coverage_count': len(new_coverage),
                            'queries': new_coverage
                        }
                    else:
                        current_instance_coverage_map.pop(instance_key)

            # 检查是否达到最大迭代次数
            if iteration >= max_iterations and current_remaining_queries:
                _logger.warning(
                    f"达到最大迭代次数({max_iterations})，仍有未覆盖的查询关键词: {list(current_remaining_queries)}")
            elif not current_remaining_queries:
                _logger.debug(f"第 {combo_idx + 1} 个组合已完全覆盖所有查询关键词")
            elif not current_instance_coverage_map:
                _logger.warning(f"第 {combo_idx + 1} 个组合: 没有更多可选的实例来覆盖剩余查询关键词")

            _logger.debug(
                f"第 {combo_idx + 1} 个组合结果: 选择 {len(current_selected_instances)} 个实例, 覆盖了 {len(current_covered_queries)}/{len(query_keywords)} 个查询关键词")

            if current_remaining_queries:
                _logger.warning(f"第 {combo_idx + 1} 个组合剩余未覆盖的查询关键词: {list(current_remaining_queries)}")

            # 如果没有找到足够的实例，继续下一个组合
            if not current_selected_instances:
                _logger.debug(f"第 {combo_idx + 1} 个组合为空，停止寻找更多组合")
                break

            # 记录当前组合
            all_combinations.append({
                'instances': current_selected_instances[:],  # 创建副本
                'covered_queries': len(current_covered_queries),
                'total_queries': len(query_keywords),
                'remaining_queries': list(current_remaining_queries)
            })

        # 如果没有找到任何组合，返回空列表
        if not all_combinations:
            _logger.warning("没有找到任何有效的组合")
            return serialize_instance([])

        # 对组合进行排序：先按覆盖数量降序，再按实例数量升序
        all_combinations.sort(key=lambda x: (-x['covered_queries'], len(x['instances'])))

        # 限制返回的组合数量
        if len(all_combinations) > combo_top_k:
            all_combinations = all_combinations[:combo_top_k]

        _logger.debug(f"第二阶段: 找到 {len(all_combinations)} 个有效组合")

        # 12. 获取所有组合的完整实例数据
        all_instances = []
        instance_processed = set()  # 记录已处理的实例，避免重复

        for combo_idx, combo in enumerate(all_combinations):
            _logger.debug(f"处理第 {combo_idx + 1} 个组合，包含 {len(combo['instances'])} 个实例")

            combo_instances = []

            for instance_key in combo['instances']:
                # 避免重复处理同一个实例
                if instance_key in instance_processed:
                    _logger.warning(f"实例 {instance_key} 已在其他组合中处理，跳过")
                    continue

                instance_info = instance_info_map[instance_key]
                class_id = instance_info['class_id']
                instance_id = instance_info['instance_id']

                # 在知识图谱中查找对应的完整实例
                found_instance = self._find_instance_by_ids(class_id, instance_id)
                if found_instance:
                    # 获取这个实例覆盖的查询关键词详情
                    coverage_details = []
                    covered_by_instance = set()

                    for detail in instance_to_queries_details[instance_key]:
                        coverage_details.append({
                            'query_keyword': detail['query_keyword'],
                            'matched_keyword': detail['matched_keyword'],
                            'similarity': float(detail['similarity'])
                        })
                        covered_by_instance.add(detail['query_keyword'])

                    # 计算平均相似度
                    if coverage_details:
                        avg_similarity = sum(detail['similarity'] for detail in coverage_details) / len(
                            coverage_details)
                    else:
                        avg_similarity = 0.0

                    # 添加元数据
                    found_instance['coverage_details'] = coverage_details
                    found_instance['covered_queries'] = [detail['query_keyword'] for detail in coverage_details]
                    found_instance['coverage_count'] = len(coverage_details)
                    found_instance['avg_similarity'] = float(avg_similarity)
                    found_instance['original_keywords'] = instance_keywords_map[instance_key]
                    found_instance['fully_covering'] = False
                    found_instance['stage'] = 2
                    found_instance['combo_index'] = combo_idx + 1
                    found_instance['combo_instance_count'] = len(combo['instances'])

                    # 添加实例唯一标识到实例中，便于后续使用
                    found_instance['instance_key'] = instance_key
                    found_instance['class_id'] = class_id
                    found_instance['instance_id'] = instance_id

                    combo_instances.append(found_instance)
                    instance_processed.add(instance_key)

                    _logger.debug(
                        f"组合 {combo_idx + 1} 实例 {instance_key}: 覆盖 {len(coverage_details)} 个查询, 平均相似度: {avg_similarity:.4f}")
                else:
                    _logger.warning(f"未找到实例: {instance_key}")

            # 将组合中的实例添加到总列表
            all_instances.extend(combo_instances)

        # 13. 如果仍有未覆盖的查询关键词，尝试添加额外实例
        if all_combinations and len(all_instances) < 10:
            # 计算所有已覆盖的查询关键词
            all_covered_queries = set()
            for inst in all_instances:
                if 'covered_queries' in inst:
                    all_covered_queries.update(inst['covered_queries'])

            # 查找未覆盖的查询关键词
            all_query_set = set(query_keywords)
            remaining_queries = all_query_set - all_covered_queries

            if remaining_queries:
                _logger.debug(f"尝试为剩余 {len(remaining_queries)} 个查询关键词寻找额外实例...")

                # 为每个未覆盖的查询关键词找到最相关的实例
                additional_instances = []

                for query in remaining_queries:
                    best_instance_for_query = None
                    best_similarity = 0

                    for instance_key, details in instance_to_queries_details.items():
                        if instance_key in instance_processed:
                            continue

                        # 找到这个查询关键词的最佳匹配
                        for detail in details:
                            if detail['query_keyword'] == query and detail['similarity'] > best_similarity:
                                best_similarity = detail['similarity']
                                best_instance_for_query = instance_key

                    if best_instance_for_query and best_similarity >= similarity_threshold:
                        additional_instances.append((best_instance_for_query, best_similarity))

                # 按相似度排序，选择前几个
                additional_instances.sort(key=lambda x: x[1], reverse=True)
                additional_instances = additional_instances[:single_top_k]

                for instance_key, similarity in additional_instances:
                    instance_info = instance_info_map[instance_key]
                    found_instance = self._find_instance_by_ids(instance_info['class_id'],
                                                                instance_info['instance_id'])

                    if found_instance:
                        # 获取这个实例的覆盖详情
                        coverage_details = []
                        for detail in instance_to_queries_details[instance_key]:
                            if detail['query_keyword'] in remaining_queries:
                                coverage_details.append({
                                    'query_keyword': detail['query_keyword'],
                                    'matched_keyword': detail['matched_keyword'],
                                    'similarity': float(detail['similarity'])
                                })

                        found_instance['coverage_details'] = coverage_details
                        found_instance['covered_queries'] = [detail['query_keyword'] for detail in coverage_details]
                        found_instance['coverage_count'] = len(coverage_details)

                        if coverage_details:
                            found_instance['avg_similarity'] = float(
                                sum(detail['similarity'] for detail in coverage_details) / len(coverage_details))
                        else:
                            found_instance['avg_similarity'] = 0.0

                        found_instance['original_keywords'] = instance_keywords_map[instance_key]
                        found_instance['is_additional'] = True
                        found_instance['fully_covering'] = False
                        found_instance['stage'] = 2
                        found_instance['combo_index'] = 0  # 标记为额外实例

                        # 添加实例唯一标识到实例中，便于后续使用
                        found_instance['instance_key'] = instance_key
                        found_instance['class_id'] = instance_info['class_id']
                        found_instance['instance_id'] = instance_info['instance_id']

                        all_instances.append(found_instance)
                        instance_processed.add(instance_key)
                        _logger.debug(f"添加额外实例 {instance_key}, 最高相似度: {similarity:.4f}")

        # 14. 将选中的实例添加到self.selected_instance_keys中
        added_instance_keys = []

        for instance in all_instances:
            # 从实例的instance_key字段中获取
            instance_key = instance.get('instance_key')

            if not instance_key:
                # 如果实例中没有instance_key，尝试从其他字段构造
                class_id = instance.get('class_id')
                instance_id = instance.get('instance_id')

                if class_id and instance_id:
                    instance_key = f"{class_id}_{instance_id}"
                else:
                    # 如果都没有，跳过这个实例
                    _logger.warning(f"无法从实例中提取class_id和instance_id: {instance}")
                    continue

            if instance_key not in self.selected_instance_keys:
                self.selected_instance_keys.add(instance_key)
                added_instance_keys.append(instance_key)
                _logger.debug(f"添加实例到已选集合: {instance_key}")

        _logger.debug(f"已将 {len(added_instance_keys)} 个实例添加到已选实例集合中")

        # 15. 按阶段、组合索引、覆盖数量和相似度排序
        all_instances.sort(key=lambda x: (
            x.get('stage', 2),  # 第一阶段在前
            x.get('combo_index', 0),  # 按组合排序
            -x.get('coverage_count', 0),  # 覆盖数量降序
            -x.get('avg_similarity', 0)  # 相似度降序
        ))

        _logger.debug(f"最终返回 {len(all_instances)} 个相关实例")

        return serialize_instance(all_instances)

