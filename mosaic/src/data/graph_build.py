"""ClassGraph build mixin: sensing, instance CRUD, edge building, conflict detection, snapshots."""
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from string import Template
from typing import Any, Dict, List

import networkx as nx

from src.assist import fetch_default_llm_model, keywords_process, read_to_file_json, serialize_instance_kw
from src.data.classnode import ClassNode
from src.data.dual_graph import (
    EDGE_LEG_ASSOCIATIVE,
    EDGE_LEG_PRAGMATIC,
    count_edge_legs,
    normalize_edge_leg,
)
from src.data.graph_base import _message_label_key, _instance_has_message_label, _trim_build_context
from src.graph.dual.hyperedge import (
    unique_directed_star_pairs_p,
    unique_undirected_star_pairs_a,
)
from src.llm.telemetry import llm_call_scope
from src.logger import setup_logger
from src.prompts_ch import PROMPT_CONFLICT
from src.prompts_en import PROMPT_NEW_CLASS_SENSE
from src.utils.io_utils import parse_llm_json_object

_logger = setup_logger("graph_build")


class ClassGraphBuildMixin:
    """Build-time methods: sensing, instance processing, edges, conflicts, snapshots."""

    def sense_classes(self,
                      data: list,
                      context: list,
                      tfidf_threshold: float = 0.5,  # TF-IDF相似度阈值
                      top_k_class: int = 3,  # 每个片段最多匹配的类数
                      use_llm_for_new: bool = True,  # 未匹配片段是否用 LLM 创建新类；False 时归入 Unclassified
                      ) -> tuple[dict[Any, dict[str, Any]], dict[Any, dict[str, Any]]]:
        """
        混合感知方法：先用TF-IDF匹配现有类，再用LLM创建新类（或 use_llm_for_new=False 时归入 Unclassified）

        Args:
            data: 信息片段列表，每个元素包含'message'和'label'
            context: 背景片段列表
            tfidf_threshold: TF-IDF相似度阈值
            top_k_class: 每个信息片段最多匹配的类数量
            use_llm_for_new: 未匹配片段是否用 LLM 创建新类；False 时全部归入 "Unclassified"
        """
        # 第一阶段：使用TF-IDF匹配现有类
        relevant_class_messages = {}  # 需要更新的已有类: class_id -> {related_message, dependent_context}
        unmatched_data = []  # 未匹配到任何类的信息片段
        processed_labels = set()  # 已处理的信息片段标签

        for item in data:
            message = item.get("message", "")
            label = item.get("label", "")

            if label in processed_labels:
                continue

            tfidf_result = self._sense_classes_by_tfidf(
                query=message,
                top_k_class=top_k_class,
                threshold=tfidf_threshold,
                allow_below_threshold=False
            )

            selected_classes = tfidf_result.get("selected_classes", [])

            if selected_classes:
                for class_info in selected_classes:
                    class_id = class_info.get("class_id")
                    if not class_id:
                        continue

                    if class_id not in relevant_class_messages:
                        relevant_class_messages[class_id] = {
                            "related_message": [],
                            "dependent_context": []
                        }

                    relevant_class_messages[class_id]["related_message"].append({
                        "message": message,
                        "label": label
                    })

                processed_labels.add(label)
            else:
                unmatched_data.append(item)

        _logger.debug(
            f"TF-IDF匹配结果: 匹配到{len(relevant_class_messages)}个类，{len(processed_labels)}个信息片段，{len(unmatched_data)}个片段未匹配")

        self.sense_class_telemetry_cumulative["tfidf_matched_fragment_labels"] += len(processed_labels)
        self.sense_class_telemetry_cumulative["tfidf_unmatched_fragments"] += len(unmatched_data)

        # 第二阶段：对未匹配的片段使用LLM创建新类，或归入 Unclassified（hash 模式）
        new_class_messages = {}  # 需要新增的类: class_name -> {related_message, dependent_context}

        if unmatched_data:
            if use_llm_for_new:
                _logger.debug(f"对{len(unmatched_data)}个未匹配片段使用LLM创建新类")
                self.sense_class_telemetry_cumulative["llm_new_class_invocations"] += 1

                ctx_trim = _trim_build_context(context)
                # 构建LLM提示词，只处理未匹配的片段
                sense_prompt = Template(PROMPT_NEW_CLASS_SENSE).substitute(
                    {
                        "DATA": unmatched_data,  # 只传入未匹配的片段
                        "CONTEXT": ctx_trim,
                    }
                )

                _logger.debug(f"LLM感知提示词: %s", sense_prompt)

                result = None
                for _retry in range(2):
                    with llm_call_scope("build.sense_new_classes"):
                        response = self._llm.invoke(sense_prompt)
                    _logger.debug(f"LLM感知响应: %s", response.content)
                    result = parse_llm_json_object(response.content)
                    if result is not None:
                        break
                    _logger.warning("LLM 新类感知 JSON 解析失败 (尝试 %d/2)", _retry + 1)

                if result is None:
                    self.sense_class_telemetry_cumulative["llm_new_class_json_failures"] += 1
                    _logger.warning(
                        "LLM 新类感知返回无法解析为 JSON，未匹配片段已归入 Unclassified（见 prompt STRICT JSON 与 parse_llm_json_object）"
                    )
                    related = [
                        {"message": item.get("message", ""), "label": item.get("label", "")}
                        for item in unmatched_data
                    ]
                    new_class_messages["Unclassified"] = {
                        "related_message": related,
                        "dependent_context": context
                        if isinstance(context, list)
                        else ([context] if context else []),
                    }
                else:
                    # 处理LLM返回的新类
                    new_classes = result.get("new_classes", [])
                    if not isinstance(new_classes, list):
                        new_classes = []

                    for item in new_classes:
                        class_name = item.get("class_name")
                        if class_name:
                            new_class_messages[class_name] = {
                                "related_message": item.get("related_message", []),
                                "dependent_context": item.get("dependent_context", []),
                            }
            else:
                # Hash 模式：全部归入 Unclassified，不调用 LLM
                _logger.debug(f"Hash 模式: 将{len(unmatched_data)}个未匹配片段归入 Unclassified")
                related = [{"message": item.get("message", ""), "label": item.get("label", "")} for item in unmatched_data]
                new_class_messages["Unclassified"] = {
                    "related_message": related,
                    "dependent_context": context if isinstance(context, list) else ([context] if context else []),
                }
        _logger.debug(f"最终结果: 需要更新的相关类: {len(relevant_class_messages)}个, 需要新增的新类: {len(new_class_messages)}个")
        return relevant_class_messages, new_class_messages

    def process_relevant_class_instances(self,
                                         relevant_class_messages: Dict[str, Dict[str, List[str]]],
                                         threshold: float = 0.9,
                                         use_hash: bool = False,
                                         ) -> list[ClassNode]:
        """
        处理相关类中的实例，根据阈值筛选相关实例和未知实例。

        Args:
            relevant_class_messages: 从 sense_classes 返回的相关类字典，class_id -> {related_message, dependent_context}
            threshold: 相似度阈值，用于判断实例相关性
            use_hash: 若 True 使用 hash 路径更新/新增实例，不调用 LLM
        """
        processed_classes = []

        for class_id, messages_dict in relevant_class_messages.items():
            try:
                class_node = self.get_classnode(class_id)
                if class_node is None:
                    _logger.warning(f"找不到对应的类节点: {class_id}")
                    continue

                related_messages = messages_dict.get("related_message", [])
                context_messages = messages_dict.get("dependent_context", [])

                relevant_instance_messages, new_instance_messages = class_node.get_message_allocation_from_instance(
                    related_messages, context_messages, threshold)

                class_node.update_relevant_instances(relevant_instance_messages, use_hash=use_hash)
                class_node.add_instances(new_instance_messages, use_hash=use_hash)

                processed_classes.append(class_node)

                _logger.debug("Processed class %s: found %d relevant instances, %d new instances",
                             class_id, len(relevant_instance_messages), len(new_instance_messages))
            except Exception as e:
                _logger.exception(
                    "处理 class_id=%s 的相关实例失败，已跳过该类并继续其它类: %s",
                    class_id,
                    e,
                )
                continue
        self.save_graph_snapshot()
        return processed_classes

    ##对于第一步感知出来的需要新创建的类，去对这些类进行实例的创建和类节点创建和类id的分配。
    def add_classnodes(self, new_class_messages: Dict[str, Dict[str, List[str]]], use_hash: bool = False) -> list[ClassNode]:
        """
        对 sense_classes 得到的新类创建类节点与实例，并分配 class_id。

        Args:
            new_class_messages: 新类字典，class_name -> {related_message, dependent_context}
            use_hash: 若 True 使用 hash 路径创建实例，不调用 LLM
        """
        current_class_count = len(self.graph.nodes)
        added_class_nodes = []
        for class_name, content in new_class_messages.items():
            try:
                class_node = ClassNode.new_classnode(class_name)
                class_id = f"class_{current_class_count + 1}"
                class_node.class_id = class_id

                class_node.process_classnode_initialization(
                    content.get("related_message", []),
                    content.get("dependent_context", []),
                    threshold=0,
                    use_hash=use_hash,
                )
                self.graph.add_node(class_node)
                _logger.debug("Added new class %s: %s", class_id, class_name)
                current_class_count += 1
                added_class_nodes.append(class_node)
            except Exception as e:
                _logger.exception(
                    "新增类 %r 失败，已跳过并继续其它新类: %s",
                    class_name,
                    e,
                )
                continue

        self.save_graph_snapshot()
        return added_class_nodes

    def save_graph_snapshot(self) -> None:
        """
        实时保存图节点的快照信息，用于调试和验证存储是否正确

        Args:
            operation: 操作描述，用于标识保存时机
        """
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.now().strftime("%Y%m%d")
        operation = os.path.basename(self.filepath or "snapshot")
        os.makedirs(self._graph_save_dir, exist_ok=True)
        filename = os.path.join(self._graph_save_dir, f"graph_snapshot_{operation}_{timestamp}.json")

        dual_counts = count_edge_legs(self.edges)
        ok_nx, nx_warn = self._dual_nx_matches_edge_records()
        if not ok_nx:
            _logger.warning("G_p/G_a 与 self.edges 不同步，执行重建: %s", nx_warn)
            self._rebuild_dual_nx_from_edges()
            ok_nx, nx_warn = self._dual_nx_matches_edge_records()
            if not ok_nx:
                _logger.warning("重建后 G_p/G_a 仍异常: %s", nx_warn)

        gp_dag = True
        if self.G_p.number_of_nodes() > 0:
            gp_dag = nx.is_directed_acyclic_graph(self.G_p)
        if not gp_dag:
            _logger.warning("G_p 非 DAG（星形定向下通常不应出现），请检查边数据")

        unique_p = unique_directed_star_pairs_p(self.edges or [])
        unique_a = unique_undirected_star_pairs_a(self.edges or [])
        snapshot_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "graph_info": {
                "total_classes": len(self.graph.nodes),
                "total_edges": len(self.graph.edges),
                "dual_graph_edge_counts": dual_counts,
                "dual_graph_legend": {
                    "P": "E_P 前提/过程：共现星形或 ep_oriented_pairs（LLM 先决）",
                    "A": "E_A 关联：BGE 语义相似等（edge_leg=A，无向加权）",
                },
                "dual_nx": {
                    "G_p": {
                        "nodes": self.G_p.number_of_nodes(),
                        "edges": self.G_p.number_of_edges(),
                        "is_dag": gp_dag,
                        "unique_star_pairs_p": len(unique_p),
                    },
                    "G_a": {
                        "nodes": self.G_a.number_of_nodes(),
                        "edges": self.G_a.number_of_edges(),
                        "unique_star_pairs_a": len(unique_a),
                    },
                    "matches_edge_records": ok_nx,
                },
                "construction_telemetry": dict(self.sense_class_telemetry_cumulative),
            },
            "classes": []
        }

        # 收集所有类的详细信息，根据ClassNode的实际数据结构
        for class_node in self.graph.nodes:
            # 关键修复：将set转换为list以确保JSON可序列化
            raw_attributes = getattr(class_node, 'attributes', set())
            raw_operations = getattr(class_node, 'operations', set())
            raw_unclassified = getattr(class_node, 'unclassified', set())

            class_info = {
                "class_id": getattr(class_node, 'class_id', 'unknown'),
                "class_name": getattr(class_node, 'class_name', 'unknown'),
                "attributes": list(raw_attributes),  # 将set转为list
                "operations": list(raw_operations),  # 将set转为list
                "unclassified": list(raw_unclassified),  # 将set转为list
                "instance_count": len(getattr(class_node, '_instances', [])),
                "instances": [str(instance) for instance in getattr(class_node, '_instances', [])]
            }
            snapshot_data["classes"].append(class_info)

        # 保存到JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)

        _logger.debug(f"图快照已保存: {filename}")

        # 2. 新增：保存整个图结构
        self._save_complete_graph(operation, timestamp)

        #3.保存边
        filename = os.path.join(self._graph_save_dir, f"graph_edge_{operation}_{timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.edges, f, indent=2, ensure_ascii=False)
        _logger.debug(f"图边已保存: {filename}")

        # 4. A-2：手稿级 EntityGraph JSON（与 graph_snapshot / graph_edge 同目录、同日戳）
        try:
            from src.graph.dual.entity_graph_store import entity_graph_from_class_graph

            eg_path = os.path.join(self._graph_save_dir, f"entity_graph_{operation}_{timestamp}.json")
            eg_store = entity_graph_from_class_graph(self)
            eg_store.write_json(eg_path)
            _logger.debug("EntityGraph 已保存: %s", eg_path)
            dag_ok, dag_detail = eg_store.validate_dag()
            if not dag_ok:
                _logger.warning("EntityGraph G_P 校验非 DAG: %s", dag_detail)
            from src.graph.dual.verify_exports import verify_classgraph_nx_vs_entity_export

            vok, vmsg = verify_classgraph_nx_vs_entity_export(self)
            if not vok:
                _logger.warning("A-3: EntityGraph 导出与 G_p/G_a 边集不一致: %s", vmsg)
        except Exception as exc:
            _logger.warning("EntityGraph 导出失败（不影响类图快照）: %s", exc)

    def _save_complete_graph(self, operation: str, timestamp: str) -> None:
        """
        保存完整的图结构为多种格式

        Args:
            operation: 操作描述
            timestamp: 时间戳，用于文件名
            """
        filename = os.path.join(self._graph_save_dir, f"graph_network_{operation}_{timestamp}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self.graph, f)
        _logger.debug(f"图完整结构已经保存 {filename}")

    #为新增的类之间和更新的类之间加边
    def update_class_relationships(self,
                                   data: list,
                                   processed_classes: list[ClassNode] | None,
                                   new_classes: list[ClassNode] | None):
        """
        为类节点下的实例建立关联关系

        Args:
            data: 包含消息和标签的数据列表
            processed_classes: 已处理的类节点列表
            new_classes: 新增的类节点列表
        """
        # 安全处理可能的None值
        processed_classes_safe = processed_classes if processed_classes is not None else []
        new_classes_safe = new_classes if new_classes is not None else []

        # 遍历data中的每条信息
        for data_item in data:
            message_content = data_item['message']
            message_label = data_item['label']

            # 存储与当前消息标签匹配的实例
            matching_instances = []
            _logger.debug(f"message_label: %s; data_item: %s", message_label, data_item)

            # 在全图类节点上查找（不仅本批 touched 类），避免漏连历史实例
            for class_node in self.graph.nodes:
                if not hasattr(class_node, '_instances') or not class_node._instances:
                    continue

                for instance_dict in class_node._instances:  # 重命名变量，明确它是字典
                    #_logger.debug(f"instance_dict: {instance_dict}")
                    #_logger.debug(f"instance_message_label: {instance_dict['message_labels']}")

                    # 检查实例的message_labels是否包含当前标签
                    if _instance_has_message_label(instance_dict, message_label):
                        # _logger.debug(f"message_label matched: {message_label}")
                        matching_instances.append({
                            'class_node': class_node,
                            'instance': instance_dict
                        })

            #_logger.debug(f"matching_instances: {matching_instances}")

            # 如果找到至少2个匹配的实例，则建立关联
            if len(matching_instances) >= 2:
                # 创建edge记录
                edge_record = {
                    "edge_leg": EDGE_LEG_PRAGMATIC,
                    "content": message_content,
                    "label": message_label,
                    "connections": [],
                }

                # 为每个匹配的实例构建连接信息
                for match in matching_instances:
                    connection_info = {
                        "class_id": match['class_node'].class_id,
                        "instance_id": match['instance']['instance_id']  # 使用正确的变量名
                    }
                    edge_record["connections"].append(connection_info)

                # 添加到edges列表
                self.edges.append(edge_record)
                self._apply_edge_record_to_dual_nx(edge_record)

                # 为每个实例添加functions字段（双向关联）
                for i, current_match in enumerate(matching_instances):
                    current_instance = current_match['instance']  # 使用正确的变量名
                    # 初始化functions字段
                    if 'functions' not in current_instance:
                        current_instance['functions'] = []

                    # 为当前实例添加与其他实例的关联
                    for j, other_match in enumerate(matching_instances):
                        if i != j:  # 不与自己关联
                            function_info = {
                                "class_id": other_match['class_node'].class_id,
                                "instance_id": other_match['instance']['instance_id'],  # 使用正确的变量名
                                "content": message_content
                            }

                            # 避免重复添加相同的关联
                            if function_info not in current_instance['functions']:
                                current_instance['functions'].append(function_info)

                    # 更新实例回类节点  classnode是指定的引用对象，会自动更新
                    #self._update_instance_in_class_node(current_class, current_instance)

                _logger.debug(f"为消息标签 %s 建立了 %d 个实例间的关联", message_label, len(matching_instances))

        # 记录处理结果
        _logger.debug(f"关系更新完成：建立了 {len(self.edges)} 个关联边")
        _logger.debug(
            "处理了 %d 条数据；本批 touched 类 %d 个，全图类节点 %d 个",
            len(data),
            len(processed_classes_safe) + len(new_classes_safe),
            len(self.graph.nodes),
        )

        self.save_graph_snapshot()

    def sweep_cross_class_cooccurrence_edges(
        self, *, min_shared_labels: int = 2
    ) -> dict[str, int]:
        """
        Post-build sweep: create E_P edges for cross-class instances that share
        message_labels but were never linked during per-batch
        ``update_class_relationships`` (because the other instance didn't exist
        yet when the batch ran).

        ``min_shared_labels`` (default 2) controls how many distinct message
        labels two instances must share before an edge is created.  A value
        of 1 reverts to the original behaviour (link on any single
        co-occurrence) but tends to produce too many spurious edges.

        Also patches the ``functions`` field so that neighbor expansion in
        ``_build_instance_adjacency`` sees the new links.

        Returns stats dict with ``edges_added``, ``pairs_linked``, ``skipped``.
        """
        from collections import defaultdict

        stats = {"edges_added": 0, "pairs_linked": 0, "skipped": 0}

        # 1. Build label → [(class_node, instance_dict)] mapping across the
        #    full graph (all batches completed).
        label_to_instances: dict[Any, list[tuple]] = defaultdict(list)
        for class_node in self.graph.nodes:
            if not hasattr(class_node, "_instances") or not class_node._instances:
                continue
            for inst in class_node._instances:
                for lab in inst.get("message_labels") or []:
                    label_to_instances[_message_label_key(lab)].append(
                        (class_node, inst)
                    )

        # 2. Build set of already-linked (instance_key, instance_key) pairs to
        #    avoid duplicate edges.
        existing_pairs: set[tuple[str, str]] = set()
        for rec in self.edges or []:
            if rec.get("edge_leg") != EDGE_LEG_PRAGMATIC:
                continue
            conns = rec.get("connections") or []
            keys = []
            for c in conns:
                cid = c.get("class_id")
                iid = c.get("instance_id")
                if cid and iid is not None:
                    keys.append(self._instance_key(str(cid), iid))
            for a in range(len(keys)):
                for b in range(a + 1, len(keys)):
                    pair = tuple(sorted([keys[a], keys[b]]))
                    existing_pairs.add(pair)

        # 3. Build instance-pair → shared-label-count mapping for the
        #    min_shared_labels filter.
        #    instance key tuple (sorted) → set of label keys they co-occur on
        inst_key_for = {}  # id(inst) → instance_key  (cache)
        pair_shared: dict[tuple[str, str], set] = defaultdict(set)

        for lab_key, entries in label_to_instances.items():
            if len(entries) < 2:
                continue
            # group by class to detect cross-class
            by_class: dict[str, list] = defaultdict(list)
            for cn, inst in entries:
                by_class[getattr(cn, "class_id", "")].append((cn, inst))
            if len(by_class) < 2:
                continue  # all intra-class — skip

            keys_in_label: list[tuple[str, Any]] = []  # (inst_key, (cn, inst))
            for cn, inst in entries:
                cid = getattr(cn, "class_id", "")
                iid = inst.get("instance_id")
                if not cid or iid is None:
                    continue
                ik = self._instance_key(cid, iid)
                inst_key_for[id(inst)] = ik
                keys_in_label.append((ik, (cn, inst)))

            # record cross-class co-occurrence
            for a in range(len(keys_in_label)):
                ka, (cn_a, _) = keys_in_label[a]
                cid_a = getattr(cn_a, "class_id", "")
                for b in range(a + 1, len(keys_in_label)):
                    kb, (cn_b, _) = keys_in_label[b]
                    cid_b = getattr(cn_b, "class_id", "")
                    if cid_a == cid_b:
                        continue
                    pair = tuple(sorted([ka, kb]))
                    pair_shared[pair].add(lab_key)

        # 4. For each cross-class pair meeting the min_shared_labels
        #    threshold, create an E_P edge record if not already present.
        #    First, collect qualified instance keys grouped by their shared
        #    labels for batching edge records.

        # Build a fast lookup: instance_key → (class_node, inst_dict)
        key_to_entry: dict[str, tuple] = {}
        for class_node in self.graph.nodes:
            if not hasattr(class_node, "_instances") or not class_node._instances:
                continue
            cid = getattr(class_node, "class_id", "")
            for inst in class_node._instances:
                iid = inst.get("instance_id")
                if iid is None:
                    continue
                key_to_entry[self._instance_key(cid, iid)] = (class_node, inst)

        for pair, shared_labs in pair_shared.items():
            if len(shared_labs) < min_shared_labels:
                stats["skipped"] += 1
                continue
            if pair in existing_pairs:
                stats["skipped"] += 1
                continue

            ka, kb = pair
            entry_a = key_to_entry.get(ka)
            entry_b = key_to_entry.get(kb)
            if entry_a is None or entry_b is None:
                continue

            cn_a, inst_a = entry_a
            cn_b, inst_b = entry_b
            cid_a = getattr(cn_a, "class_id", "")
            cid_b = getattr(cn_b, "class_id", "")
            iid_a = inst_a.get("instance_id")
            iid_b = inst_b.get("instance_id")

            lab_desc = ",".join(sorted(shared_labs)[:3])
            edge_record = {
                "edge_leg": EDGE_LEG_PRAGMATIC,
                "content": f"[post-build sweep] co-occurrence on {len(shared_labs)} labels: {lab_desc}",
                "label": lab_desc,
                "connections": [
                    {"class_id": cid_a, "instance_id": iid_a},
                    {"class_id": cid_b, "instance_id": iid_b},
                ],
            }
            self.edges.append(edge_record)
            self._apply_edge_record_to_dual_nx(edge_record)
            stats["edges_added"] += 1
            existing_pairs.add(pair)

            # Patch functions field for neighbor expansion
            all_entries = [entry_a, entry_b]
            for a in range(len(all_entries)):
                cn_a_f, inst_a_f = all_entries[a]
                if "functions" not in inst_a_f:
                    inst_a_f["functions"] = []
                for b in range(len(all_entries)):
                    if a == b:
                        continue
                    cn_b_f, inst_b_f = all_entries[b]
                    cid_a_f = getattr(cn_a_f, "class_id", "")
                    cid_b_f = getattr(cn_b_f, "class_id", "")
                    if cid_a_f == cid_b_f:
                        continue  # skip intra-class (already linked)
                    fn_info = {
                        "class_id": cid_b_f,
                        "instance_id": inst_b_f.get("instance_id"),
                        "content": f"[sweep] co-occurrence {len(shared_labs)} labels",
                    }
                    if fn_info not in inst_a_f["functions"]:
                        inst_a_f["functions"].append(fn_info)
                        stats["pairs_linked"] += 1

        _logger.info(
            "Post-build cross-class edge sweep: %d edges added, %d function pairs linked, %d skipped",
            stats["edges_added"],
            stats["pairs_linked"],
            stats["skipped"],
        )
        return stats

    def sweep_uncovered_messages(
        self, all_batches: list[tuple[list, list]], *, min_text_len: int = 30
    ) -> dict[str, int]:
        """Post-build sweep: create hash instances for messages dropped during construction.

        Some messages are lost when ``sense_classes`` matches them to a class
        but the LLM instance-creation step omits their content (e.g. a message
        starting with a greeting that also contains factual information like
        "went fishing").  This sweep catches those orphaned messages and adds
        them as hash-style instances in the best-matching existing class,
        ensuring they become searchable by TF-IDF at query time.

        Args:
            all_batches: The ``(data, context)`` pairs from ``conv_message_splitter``.
            min_text_len: Minimum message text length to consider (filters greetings).

        Returns:
            Stats dict with ``checked``, ``uncovered``, ``instances_created``.
        """
        stats: dict[str, int] = {"checked": 0, "uncovered": 0, "instances_created": 0}

        # 1. Collect every label already present in any instance
        covered_labels: set[str] = set()
        for cn in self.graph.nodes:
            for inst in getattr(cn, "_instances", []) or []:
                for lab in inst.get("message_labels") or []:
                    covered_labels.add(_message_label_key(lab))

        # 2. Identify uncovered messages with substantive content
        uncovered: list[dict] = []
        for batch, _ctx in all_batches:
            for msg in batch:
                stats["checked"] += 1
                label = msg.get("label", "")
                if _message_label_key(label) in covered_labels:
                    continue
                text = msg.get("message", "")
                if len(text.strip()) < min_text_len:
                    continue
                uncovered.append(msg)

        stats["uncovered"] = len(uncovered)
        if not uncovered:
            _logger.debug("sweep_uncovered_messages: all messages accounted for")
            return stats

        _logger.info(
            "sweep_uncovered_messages: %d/%d messages uncovered, creating hash instances",
            len(uncovered), stats["checked"],
        )

        # 3. For each uncovered message, TF-IDF match to the best class
        #    and create a minimal hash instance.
        for msg in uncovered:
            text = msg.get("message", "")
            label = msg.get("label", "")

            tfidf_result = self._sense_classes_by_tfidf(
                query=text, top_k_class=1, threshold=0.0, allow_below_threshold=True
            )
            selected = tfidf_result.get("selected_classes", [])
            class_node = None
            if selected:
                cid = selected[0].get("class_id")
                if cid:
                    class_node = self.get_classnode(cid)

            if class_node is None:
                # Fall back to the first class node (usually the broadest)
                nodes = list(self.graph.nodes)
                if nodes:
                    class_node = nodes[0]
            if class_node is None:
                continue

            instances = getattr(class_node, "_instances", None) or []
            current_count = len(instances)
            instance = {
                "instance_id": f"instance_{current_count + 1}",
                "instance_name": f"Supplementary content (label {label})",
                "attributes": {},
                "operations": {},
                "uninstance_field": text,
                "message_labels": [label],
            }
            if not hasattr(class_node, "_instances") or class_node._instances is None:
                class_node._instances = []
            class_node._instances.append(instance)
            stats["instances_created"] += 1
            _logger.debug(
                "sweep_uncovered: label %s → class %s (%s)",
                label,
                getattr(class_node, "class_id", "?"),
                getattr(class_node, "class_name", "?"),
            )

        _logger.info(
            "sweep_uncovered_messages: created %d hash instances for dropped messages",
            stats["instances_created"],
        )
        return stats

    def enrich_dual_graph_edges_post_build(self) -> dict[str, Any]:
        """
        构图全批次结束后：BGE 建 E_A、可选 LLM 非对称先决 E_P（DAG 安全），并刷新快照 / EntityGraph。

        见 docs/optimization.md §5 B-2、``[EDGE]`` 配置、``src/graph/dual/edge_construction.py``。
        """
        from src.graph.dual.edge_construction import enrich_class_graph_dual_edges

        return enrich_class_graph_dual_edges(self, llm=getattr(self, "_llm", None))

    def message_passing(self):
        pass

    def consistency_valid_dynamic(self, relevant_class_messages: dict, new_class_messages: dict) -> dict:
        """
        分类已有类和新增类的信息，不进行冲突检查，只做好信息分组。

        Args:
            relevant_class_messages: 与已有类相关的信息字典，class_id -> {related_message, dependent_context}
            new_class_messages: 需要新增的类相关信息字典，class_name -> {related_message, dependent_context}

        Returns:
            包含 existing_classes 与 new_classes 的 classified_data
        """
        classified_data = {
            "existing_classes": {},
            "new_classes": {}
        }

        for class_id, class_info in relevant_class_messages.items():
            target_class = None
            for class_node in self.graph.nodes:
                if class_node.class_id == class_id:
                    target_class = class_node
                    break

            if target_class is None:
                _logger.warning(f"未找到类ID为 {class_id} 的类节点")
                continue

            # 收集该类相关的信息
            class_data = {
                "class_id": class_id,
                "class_name": getattr(target_class, 'class_name', '未知类名'),
                "new_messages": class_info.get("related_message", []),
                "dependent_context": class_info.get("dependent_context", []),
                "existing_messages": []  # 这里将存储转换后的完整消息标签信息
            }

            # 收集该类已有实例的信息标签，并转换为包含完整消息的格式
            if hasattr(target_class, '_instances') and target_class._instances:
                label_to_message = {}
                for item in self.message_labels:
                    label_to_message[item['label']] = item['message']

                # 收集所有实例的标签并去重
                all_label_ids = set()
                for instance in target_class._instances:
                    if 'message_labels' in instance:
                        all_label_ids.update(instance['message_labels'])

                # 转换为包含完整信息的格式
                for label_id in all_label_ids:
                    if label_id in label_to_message:
                        class_data["existing_messages"].append({
                            "message": label_to_message[label_id],
                            "label": label_id
                        })
                    else:
                        _logger.warning(f"未找到标签ID为 {label_id} 对应的消息内容")

            classified_data["existing_classes"][class_id] = class_data

        for class_name, class_info in new_class_messages.items():
            new_class_data = {
                "class_name": class_name,
                "new_messages": class_info.get("related_message", []),
                "dependent_context": class_info.get("dependent_context", []),
            }
            classified_data["new_classes"][class_name] = new_class_data

        # 3. 记录分类结果用于调试
        _logger.debug(f"信息分类完成：")
        _logger.debug(f"- 已有类: {len(classified_data['existing_classes'])}")
        _logger.debug(f"- 新增类: {len(classified_data['new_classes'])}")

        #对分类好的数据逐个检查
        self.detect_conflicts(classified_data)

        return classified_data

    def detect_conflicts(self, classified_data):

        # 1. 分析existing_classes中的每个类别
        if "existing_classes" in classified_data:
            for class_id, class_data in classified_data["existing_classes"].items():
                # 合并new_messages和existing_messages
                all_messages = []

                if "new_messages" in class_data:
                    all_messages.extend(class_data["new_messages"])
                if "existing_messages" in class_data:
                    all_messages.extend(class_data["existing_messages"])

                if all_messages:
                    # 调用llm去判断信息片段中是否有冲突
                    conflict_prompt = Template(PROMPT_CONFLICT).substitute(
                        {
                            "messages": all_messages,
                        }
                    )
                    _logger.debug(f"conflict prompt: %s", conflict_prompt)
                    result = None
                    for _retry in range(2):
                        with llm_call_scope("build.conflict_existing"):
                            response = self._llm.invoke(conflict_prompt)
                        conflict_raw = getattr(response, "content", None) or str(response)
                        _logger.debug(f"conflict prompt response: %s", conflict_raw)
                        result = parse_llm_json_object(conflict_raw)
                        if result is not None:
                            break
                        _logger.warning("冲突检测(existing) JSON 解析失败 (尝试 %d/2)", _retry + 1)
                    if result is None:
                        _logger.warning("冲突检测 LLM 回复无法解析为 JSON，跳过该 existing_class。")
                        continue

                    if result.get("is_conflict", False):
                        # 获取所有冲突涉及的消息标签
                        conflict_labels = set()
                        for conflict in result.get("conflicts", []):
                            conflict_labels.update(conflict.get("conflict_message_labels", []))

                        conflict_messages = [
                            m for m in all_messages if m.get("label") in conflict_labels
                        ]
                        items = {
                            "class_id": class_data.get("class_id", class_id),
                            "class_name": class_data.get("class_name", f"未命名类别_{class_id}"),
                            "class_type": "existing_class",
                            "is_conflict": True,
                            "conflicts": result.get("conflicts", []),
                            "messages": conflict_messages
                        }
                        self.warning_items.append(items)

        # 2. 分析new_classes中的每个类别
        if "new_classes" in classified_data:
            for class_name, class_data in classified_data["new_classes"].items():
                # 只使用new_messages
                messages = class_data.get("new_messages", [])

                if messages:
                    conflict_prompt = Template(PROMPT_CONFLICT).substitute(
                        {
                            "messages": messages,
                        }
                    )
                    _logger.debug(f"conflict prompt: %s", conflict_prompt)
                    result = None
                    for _retry in range(2):
                        with llm_call_scope("build.conflict_new"):
                            response = self._llm.invoke(conflict_prompt)
                        conflict_raw = getattr(response, "content", None) or str(response)
                        _logger.debug(f"conflict prompt response: %s", conflict_raw)
                        result = parse_llm_json_object(conflict_raw)
                        if result is not None:
                            break
                        _logger.warning("冲突检测(new) JSON 解析失败 (尝试 %d/2)", _retry + 1)
                    if result is None:
                        _logger.warning("冲突检测 LLM 回复无法解析为 JSON，跳过该 new_class。")
                        continue

                    if result.get("is_conflict", False):

                        # 获取所有冲突涉及的消息标签
                        conflict_labels = set()
                        for conflict in result.get("conflicts", []):
                            conflict_labels.update(conflict.get("conflict_message_labels", []))

                        conflict_messages = [
                            m for m in messages if m.get("label") in conflict_labels
                        ]

                        items = {
                            "class_id": class_data.get("class_id", class_name),
                            "class_name": class_data.get("class_name", class_name),
                            "class_type": "new_class",
                            "is_conflict": True,
                            "conflicts": result.get("conflicts", []),
                            "messages": conflict_messages
                        }
                        self.warning_items.append(items)

        debug_dir = os.path.join(os.getcwd(), "results", "conflict")
        os.makedirs(debug_dir, exist_ok=True)
        debug_filename = os.path.join(debug_dir, "warning_items.json")
        with open(debug_filename, 'w', encoding='utf-8') as f:
            json.dump(self.warning_items, f, indent=2, ensure_ascii=False)

    def process_kw(self, filepath=None):
        tags = []
        if filepath and os.path.exists(filepath):
            tags = read_to_file_json(filepath)
        else:
            for class_node in self.graph.nodes:
                class_id = getattr(class_node, 'class_id', '')
                class_instances = getattr(class_node, '_instances', [])

                for instance in class_instances:
                    tag = self.process_instance(instance, class_id)
                    tags.append(tag)
            # print(tags)
            # 保存到JSON文件
            filename = filepath
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(tags, f, indent=2, ensure_ascii=False)
        self.tags = tags

    def process_instance(self, instance, class_id):
        """
        处理单个实例，提取关键词
        """
        instance_id = instance.get('instance_id', 'unknown')
        # 收集所有文本内容用于关键词提取
        combined_text = serialize_instance_kw(instance)

        model = embedding_model
        kw_model = KeyBERT(model=model)
        # 提取关键词
        keywords = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 2),  # 关键词长度范围，表示提取从单个单词到两个单词的关键词
            stop_words='english',
            top_n=10  # 返回前10个关键词
        )
        _logger.debug("keybert 结果: keywords=%s", keywords)
        keyword_strings = keywords_process(keywords)
        _logger.debug("keyword_strings=%s", keyword_strings)
        # 返回格式化的结果
        return {
            "keywords": keyword_strings,  # 仅包含关键词字符串的列表
            "class_id": class_id,
            "instance_id": instance_id
        }

    def generate_tags_tfidf(self, filepath: str, top_n: int = 10) -> None:
        """
        仅用 TF-IDF 为图中每个实例生成关键词 tags（不调用 KeyBERT/LLM），并写入 filepath，同时设置 self.tags。
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from src.utils.constants import DEFAULT_TFIDF_VECTORIZER_PARAMS, tfidf_params_for_corpus_size

        tags = []
        doc_list = []
        meta = []  # (class_id, instance_id) per doc

        for class_node in self.graph.nodes:
            class_id = getattr(class_node, "class_id", "")
            for instance in getattr(class_node, "_instances", []):
                if not isinstance(instance, dict):
                    continue
                text = serialize_instance_kw(instance)
                if not text or not text.strip():
                    text = " "
                doc_list.append(text)
                meta.append((class_id, instance.get("instance_id", "unknown")))

        if not doc_list:
            _logger.warning("图中无实例，跳过 tags 生成")
            self.tags = []
            if filepath:
                os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
            return

        params = tfidf_params_for_corpus_size(
            dict(DEFAULT_TFIDF_VECTORIZER_PARAMS), len(doc_list)
        )
        vectorizer = TfidfVectorizer(**params)
        X = vectorizer.fit_transform(doc_list)
        feature_names = vectorizer.get_feature_names_out()

        for i, (class_id, instance_id) in enumerate(meta):
            row = X.getrow(i)
            if row.nnz == 0:
                keywords = []
            else:
                scores = row.toarray().flatten()
                top_idx = scores.argsort()[-top_n:][::-1]
                keywords = [
                    feature_names[j] for j in top_idx
                    if scores[j] > 0 and feature_names[j].strip()
                ][:top_n]
            tags.append({"class_id": class_id, "instance_id": instance_id, "keywords": keywords})

        self.tags = tags
        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(tags, f, indent=2, ensure_ascii=False)
        _logger.debug("已生成 %s 条 TF-IDF tags 并写入 %s", len(tags), filepath)

