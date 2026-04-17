"""ClassGraph base: initialization, serialization, dual-graph sync, state tracking."""
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import networkx as nx

from src.assist import build_instance_fragments, fetch_default_llm_model
from src.data.classnode import ClassNode
from src.data.dual_graph import (
    EDGE_LEG_ASSOCIATIVE,
    EDGE_LEG_PRAGMATIC,
    count_edge_legs,
    normalize_edge_leg,
)
from src.graph.dual.hyperedge import (
    oriented_ep_pairs_from_record,
    star_oriented_pairs_from_connections,
    unique_directed_star_pairs_p,
    unique_undirected_star_pairs_a,
)
from src.logger import setup_logger

_logger = setup_logger("graph_base")


def _truncate_context(text: str, max_chars: int) -> str:
    """按实例边界截断上下文，保留前 max_chars 字符内完整的实例块。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    # Split on instance headers to truncate at instance boundaries
    sep = "\n─── instance "
    parts = text.split(sep)
    result = parts[0]
    for part in parts[1:]:
        candidate = result + sep + part
        if len(candidate) > max_chars:
            break
        result = candidate
    if not result.strip():
        # Fallback: hard truncate if no instance boundary found
        return text[:max_chars]
    return result


def _trim_build_context(context: Any, max_messages: int | None = None) -> Any:
    """限制构图时传入 LLM 的上下文条数，降低 prompt 体积（MOSAIC_BUILD_CONTEXT_MAX_MESSAGES，默认 12）。"""
    if max_messages is None:
        raw = os.environ.get("MOSAIC_BUILD_CONTEXT_MAX_MESSAGES", "12").strip()
        max_messages = int(raw) if raw.isdigit() else 12
    if max_messages <= 0 or not isinstance(context, list):
        return context
    if len(context) <= max_messages:
        return context
    return context[-max_messages:]


def _message_label_key(lab: Any) -> str:
    if lab is None:
        return ""
    return str(lab).strip()


def _instance_has_message_label(instance_dict: dict, message_label: Any) -> bool:
    """与 ``data_item['label']`` 对齐：兼容 int/str（如 3 与 \"3\"）。"""
    labs = instance_dict.get("message_labels")
    if not labs:
        return False
    key = _message_label_key(message_label)
    return any(_message_label_key(x) == key for x in labs)



class ClassGraphBase:
    def __init__(self,
                 instance_sense_threshold: float = 0.7,
                 llm=fetch_default_llm_model()):
        self._llm = llm
        self.graph: nx.Graph = nx.Graph()
        # A-3：与 self.edges（及 EntityGraph 星形展开）同步的 networkx 双图
        self.G_p: nx.DiGraph = nx.DiGraph()
        self.G_a: nx.Graph = nx.Graph()
        self._instance_sense_threshold = 0.7
        self._built_in = ["unclassified", "attributes", "operations"]
        self.name = ""
        self.description = ""
        self.edges = []  #存储边
        self.message_labels = []  # 每条信息被分配的标签，供 consistency_valid_dynamic 使用
        self.warning_items = []  #存储警告信息

        self.tags = []  #用来存储每个实例的tags
        self.filepath = "" #存储构图过程中的图文件
        # 图持久化目录：优先实例属性，其次环境变量 GRAPH_SAVE_DIR，否则当前目录下 results/graph
        self._graph_save_dir = os.environ.get("GRAPH_SAVE_DIR") or os.path.join(os.getcwd(), "results", "graph")

        self.selected_instance_keys = set()  # 用于存储已选中实例的唯一标识
        self.all_instances = []  # 存储所有实例对象
        self._retrieval_bge_accum: list[dict[str, Any]] = []  # 单次查询内多次 _fetch 的 BGE 遥测
        self._bge_embedding_cache: dict[str, Any] | None = None  # 构图阶段 BGE 嵌入缓存，供查询复用

        # HaluMem eval: 用于跟踪会话间新增的实例状态
        self.last_session_instance_states = set()
        # B 轨遥测（docs/optimization.md §5 B-3）：跨批次累计，供日志与快照/EntityGraph legacy
        self.sense_class_telemetry_cumulative: dict[str, int] = {
            "tfidf_matched_fragment_labels": 0,
            "tfidf_unmatched_fragments": 0,
            "llm_new_class_invocations": 0,
            "llm_new_class_json_failures": 0,
        }

    def reset_sense_class_telemetry(self) -> None:
        for k in self.sense_class_telemetry_cumulative:
            self.sense_class_telemetry_cumulative[k] = 0

    def __getstate__(self):
        """构图断点 pickle：LangChain LLM 不可稳定序列化，落盘时剥离，加载后重建。"""
        state = self.__dict__.copy()
        state.pop("_llm", None)
        return state

    def __setstate__(self, state):
        state.pop("_llm", None)
        self.__dict__.update(state)
        self._llm = fetch_default_llm_model()
        # 旧断点无 G_p/G_a：从 self.edges 重建，保证 A-3 与 EntityGraph 导出一致
        if not hasattr(self, "G_p") or self.G_p is None:
            self.G_p = nx.DiGraph()
        if not hasattr(self, "G_a") or self.G_a is None:
            self.G_a = nx.Graph()
        if not hasattr(self, "_bge_embedding_cache"):
            self._bge_embedding_cache = None
        self._rebuild_dual_nx_from_edges()

    def __str__(self):
        return f"ClassGraph<{self.name}>"

    def __repr__(self):
        return f"{self.__str__()}, description: {self.description}, classes: {len(self.graph.nodes)}"

    #根据类id 去查找指定的类
    def get_classnode(self, class_id: str) -> ClassNode | None:
        for node in self.graph.nodes:
            if node.class_id == class_id:
                return node
        return None

    def list_classes(self) -> List[ClassNode]:
        return list(self.graph.nodes)

    def _apply_edge_record_to_dual_nx(self, rec: dict) -> None:
        """A-3：每条 ``self.edges`` 记录同时写入 G_p（E_P 有向）或 G_a（E_A 无向加权）。"""
        leg = normalize_edge_leg(rec.get("edge_leg"))
        if leg == EDGE_LEG_PRAGMATIC:
            pairs = oriented_ep_pairs_from_record(rec)
        else:
            pairs = star_oriented_pairs_from_connections(rec.get("connections") or [])
        if not pairs:
            return
        prov = {
            "message_label": rec.get("label"),
            "content_preview": (str(rec.get("content") or ""))[:200],
        }
        pextra = rec.get("provenance")
        if isinstance(pextra, dict):
            prov = {**prov, **pextra}
        if leg == EDGE_LEG_PRAGMATIC:
            for u, v in pairs:
                self.G_p.add_edge(u, v, **prov)
        elif leg == EDGE_LEG_ASSOCIATIVE:
            w = float(rec.get("weight", 1.0))
            for u, v in pairs:
                a, b = (u, v) if u <= v else (v, u)
                if self.G_a.has_edge(a, b):
                    self.G_a[a][b]["weight"] = max(
                        float(self.G_a[a][b].get("weight", w)), w
                    )
                else:
                    self.G_a.add_edge(a, b, weight=w, **prov)

    def _rebuild_dual_nx_from_edges(self) -> None:
        """从 ``self.edges`` 全量重建 G_p/G_a（用于校验失败后的自愈）。"""
        self.G_p.clear()
        self.G_a.clear()
        for rec in self.edges or []:
            self._apply_edge_record_to_dual_nx(rec)

    def _dual_nx_matches_edge_records(self) -> tuple[bool, str | None]:
        """唯一星形展开边集应与当前 nx 图一致（去重后与 DiGraph/Graph 边数对齐）。"""
        exp_p = unique_directed_star_pairs_p(self.edges or [])
        got_p = set(self.G_p.edges())
        if exp_p != got_p:
            return (
                False,
                f"G_p 边集不一致: |期望|={len(exp_p)} |nx|={len(got_p)} "
                f"仅nx={got_p - exp_p} 仅期望={exp_p - got_p}",
            )
        exp_a = unique_undirected_star_pairs_a(self.edges or [])
        got_a = {(u, v) if u <= v else (v, u) for u, v in self.G_a.edges()}
        if exp_a != got_a:
            return (
                False,
                f"G_a 边集不一致: |期望|={len(exp_a)} |nx|={len(got_a)}",
            )
        return True, None

    ###感知信息和现有类，看哪些类需要更新，哪些类需要新增。对于需要更新的类，保留更新的类id和对应这个类的信息片段和背景片段；对于需要新增的类，保留这个新增类的类名和新增类的信息片段和背景片段
    def _instance_key(self, class_id: str, instance_id) -> str:
        return f"{class_id}_{instance_id}"

    @staticmethod
    def _instance_key_to_entity_id(instance_key: str) -> str:
        """``class_1_instance_1`` → ``class_1:instance_1``（与 EntityGraph entity_id 对齐）。"""
        sep = "_instance_"
        if sep in instance_key:
            cid, tail = instance_key.split(sep, 1)
            return f"{cid}:instance_{tail}"
        return instance_key

    def _instance_keys_to_entity_ids(self, keys: set[str] | list[str]) -> list[str]:
        out = [self._instance_key_to_entity_id(k) for k in keys]
        return sorted(set(out))

    def graph_stats_for_qa(self) -> dict[str, Any]:
        """E-1 / optimization §7：当前图统计，供每条 QA 记录附着。"""
        n_inst = sum(len(getattr(n, "_instances", []) or []) for n in self.graph.nodes)
        n_class = len(self.graph.nodes)
        up = unique_directed_star_pairs_p(self.edges or [])
        ua = unique_undirected_star_pairs_a(self.edges or [])
        np_e, na_e = len(up), len(ua)
        density_a = (2.0 * na_e / (n_inst * (n_inst - 1))) if n_inst > 1 else 0.0
        gp_dag = True
        if self.G_p.number_of_nodes() > 0:
            gp_dag = nx.is_directed_acyclic_graph(self.G_p)
        return {
            "schema_version": 1,
            "|V|_instances": n_inst,
            "|V|_classes": n_class,
            "|E_P|_unique": np_e,
            "|E_A|_unique": na_e,
            "edge_records_count": len(self.edges or []),
            "density_G_a": round(density_a, 8),
            "G_p_is_dag": gp_dag,
            "G_p_edges": self.G_p.number_of_edges(),
            "G_a_edges": self.G_a.number_of_edges(),
        }

    def _find_instance_by_ids(self, class_id, instance_id):
        """
        根据类ID和实例ID在知识图谱中查找具体实例

        Args:
            class_id: 类ID
            instance_id: 实例ID

        Returns:
            Dict: 找到的实例数据，未找到返回None
        """
        for class_node in self.graph.nodes:
            if getattr(class_node, 'class_id', None) == class_id:
                class_instances = getattr(class_node, '_instances', [])

                # 遍历该类的所有实例，查找匹配的实例ID
                for instance in class_instances:
                    if instance.get('instance_id') == instance_id:
                        return instance
        return None

    # ── HaluMem eval helpers ───────────────────────────────────────
    def get_all_instances_state(self):
        """
        获取当前所有实例的状态（唯一标识符）
        :return: 包含所有实例(class_id, instance_id)的集合
        """
        instance_states = set()
        for node in self.graph.nodes:
            if hasattr(node, 'class_id') and hasattr(node, '_instances'):
                class_id = getattr(node, 'class_id', None)
                if class_id is None:
                    continue
                class_instances = getattr(node, '_instances', [])
                for instance in class_instances:
                    instance_id = instance.get('instance_id')
                    if instance_id:
                        instance_states.add((class_id, instance_id))
        return instance_states

    def record_current_state(self):
        """
        记录当前实例状态
        """
        self.last_session_instance_states = self.get_all_instances_state()

    def get_new_instances_since_state(self, old_state):
        """
        获取自指定状态以来新增的实例
        :param old_state: 之前记录的实例状态集合
        :return: 新增的实例列表
        """
        current_state = self.get_all_instances_state()
        new_instance_keys = current_state - old_state

        _logger.info(f"新增的实例有 {len(new_instance_keys)}个, 具体{new_instance_keys}")

        new_instances = []
        for class_id, instance_id in new_instance_keys:
            instance = self._find_instance_by_ids(class_id, instance_id)
            if instance:
                if isinstance(instance, dict):
                    instance_with_class = instance.copy()
                    instance_with_class['class_id'] = class_id
                else:
                    try:
                        instance_with_class = dict(instance.__dict__) if hasattr(instance, '__dict__') else {}
                        instance_with_class['class_id'] = class_id
                    except Exception:
                        instance_with_class = {
                            'original_instance': instance,
                            'class_id': class_id
                        }
                new_instances.append(instance_with_class)

        return new_instances
