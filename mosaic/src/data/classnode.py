from __future__ import annotations
from typing import Dict, List, Any, Tuple,TYPE_CHECKING

from src.assist import similarity_score
import numpy as np
from src.logger import setup_logger
from src.data.instance import  update_data_from_messages,create_instances_from_messages
from src.assist import build_instance_fragments,calculate_tfidf_similarity
_logger = setup_logger("class cudr")

class ClassNode:
    def __init__(self, 
                 class_name: str,
                 attributes: Dict[str, Any]={}):
        self.class_id = None
        self.class_name = class_name
        self.attributes = []
        self.operations = []
        self.unclassified = []
        self._instances = []


    #需要哈希值时触发，当用于集合/字典的键的时候，如hash(obj)
    def __hash__(self):
        return hash(f"{self.class_name}")

    #当使用==比较对象的时候 obj1==obj2
    def __eq__(self, other):
        return self.class_id == other.class_id

    #打印或转字符串时触发，当使用str(obj)或者print(obj)
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        # 提取所有实例的属性描述
        attributes_desc = []
        for inst in self._instances:
            # 检查实例是否有attributes字段
            if "attributes" in inst and isinstance(inst["attributes"], dict):
                for key, content in inst["attributes"].items():
                    if "description" in content:
                        attributes_desc.append(content["description"])

        # 提取所有实例的操作描述
        operations_desc = []
        for inst in self._instances:
            # 检查实例是否有operations字段
            if "operations" in inst and isinstance(inst["operations"], dict):
                for key, content in inst["operations"].items():
                    if "description" in content:
                        operations_desc.append(content["description"])

        # 提取所有实例的名称
        instance_names = []
        for inst in self._instances:
            if "instance_name" in inst:
                instance_names.append(inst["instance_name"])

        #提取所有实例的unclassified字段
        unclassified_fields = []
        for inst in self._instances:
            # 检查实例是否有uninstance_field字段
            if "uninstance_field" in inst and inst["uninstance_field"]:
                unclassified_fields.append(inst["uninstance_field"])

        # 去重
        unique_attributes = list(set(attributes_desc))
        unique_operations = list(set(operations_desc))
        unique_unclassified = list(set(unclassified_fields))

        return f"""{self.class_name} (ID: {getattr(self, 'class_id', 'N/A')})
    Attributes: {", ".join(unique_attributes) if unique_attributes else "No attributes"}
    Operations: {", ".join(unique_operations) if unique_operations else "No operations"}
    Unclassified: {", ".join(unique_unclassified) if unique_unclassified else "No unclassified"}
    Instances: { ", ".join(instance_names) if instance_names else "No instances"}"""

    @staticmethod
    def new_classnode(class_name: str) -> "ClassNode":
        """创建仅含类名的新类节点。"""
        return ClassNode(class_name=class_name)

   #找到和信息最匹配的实例
    def _fetch_instance(self, message: str, threshold: float, top_k_instance: int = 10000):
        """使用TF-IDF返回与消息最匹配的top_k_instance个实例（不足时返回全部匹配的）"""

        # 1. 构建所有实例的文本片段
        instance_documents = []  # 存储每个实例的文本表示
        instance_indices = []  # 存储每个片段对应的实例索引

        for idx, instance in enumerate(self._instances):
            # 构建该实例的多个文本片段
            fragments = build_instance_fragments(instance)

            for fragment_type, fragment_text in fragments:
                if fragment_text.strip():  # 只添加非空片段
                    instance_documents.append(fragment_text)
                    instance_indices.append(idx)

        if not instance_documents:
            _logger.warning("没有可检索的实例片段")
            return []

        # 2. 计算TF-IDF相似度
        try:
            similarities, _, _ = calculate_tfidf_similarity(message, instance_documents)
        except Exception as e:
            _logger.error(f"TF-IDF计算失败: {e}")
            return []

        # 3. 为每个实例找出最高相似度的片段
        instance_max_scores = {}  # 实例索引 -> 最高相似度
        instance_best_fragments = {}  # 实例索引 -> 最佳片段文本

        for fragment_idx, score in enumerate(similarities):
            instance_idx = instance_indices[fragment_idx]
            fragment_text = instance_documents[fragment_idx]

            # 如果这是该实例的更高相似度片段，更新记录
            if instance_idx not in instance_max_scores or score > instance_max_scores[instance_idx]:
                instance_max_scores[instance_idx] = score
                instance_best_fragments[instance_idx] = fragment_text

        if not instance_max_scores:
            _logger.warning("没有找到任何匹配的实例")
            return []

        # 4. 将实例按最高相似度排序
        instance_scores = list(instance_max_scores.items())
        instance_scores.sort(key=lambda x: x[1], reverse=True)

        _logger.info(f"TF-IDF实例检索: 总共{len(self._instances)}个实例，{len(instance_documents)}个片段，"
                     f"找到{len(instance_scores)}个有匹配片段的实例")

        # 5. 过滤低于阈值的实例
        above_threshold = [(idx, score) for idx, score in instance_scores if score >= threshold]
        below_threshold = [(idx, score) for idx, score in instance_scores if score < threshold]

        _logger.info(f"相似度统计: 高于阈值({threshold})实例数: {len(above_threshold)}, "
                     f"低于阈值实例数: {len(below_threshold)}")

        # 6. 安全处理top_k_instance（确保是正整数）
        if top_k_instance <= 0:
            return []

        selected_instance_scores =[]
        # 7. 根据新逻辑选择实例
        if len(above_threshold) >= top_k_instance:
            # 情况1: 高于阈值的实例足够
            selected_instance_scores = above_threshold[:top_k_instance]
            _logger.info(f"高于阈值实例充足，直接返回前{len(selected_instance_scores)}个")
        # else:
        #     # 情况2: 高于阈值的实例不足，用低于阈值但分数最高的补足
        #     num_needed = top_k_instance - len(above_threshold)
        #     # 从低于阈值部分取前num_needed个最高分的
        #     supplementary_instances = below_threshold[:min(num_needed, len(below_threshold))]
        #     selected_instance_scores = above_threshold + supplementary_instances
        #     _logger.info(f"高于阈值实例不足，用{len(supplementary_instances)}个低于阈值但分数最高的实例补足")

        # _logger.info(f"最终选中的实例数量: {len(selected_instance_scores)}")

        # 8. 准备返回的实例列表
        result_instances = []
        for idx, score in selected_instance_scores:
            instance = self._instances[idx]

            # 添加相似度分数和匹配片段信息
            instance_with_score = instance.copy()
            instance_with_score['similarity_score'] = float(score)
            instance_with_score['matched_fragment'] = instance_best_fragments.get(idx, "")

            result_instances.append(instance_with_score)

            _logger.debug(f"选择实例 {idx}: 相似度={score:.4f}, "
                          f"匹配片段={instance_best_fragments.get(idx, '')[:50]}...")

        return result_instances
    def get_message_allocation_from_instance(
            self,
            messages,
            context_messages: List[str],
            threshold: float,
    ) -> Tuple[Dict[str, List[str]], Dict[ClassNode, Dict[str, List[str]]]]:
        """
        从消息中提取相关实例和未知实例

        Args:
            messages: 需要处理的消息列表
            context_messages: 上下文消息列表
            threshold: 相似度阈值，用于判断实例相关性

        Returns:
            Tuple[instance_id_to_messages, unmatched_messages_by_node]:
            - instance_id_to_messages: 要更新的实例 ID -> 消息列表
            - unmatched_messages_by_node: 类节点 -> {messages, context_messages}，用于新增实例
        """
        instance_id_to_messages: Dict[str, List[str]] = {}
        unmatched_message_items: List = []

        for message_item in messages:
            threshold = 1.2
            message_content = message_item['message']
            matching_instances = self._fetch_instance(message_content, threshold)

            if matching_instances:
                for instance in matching_instances:
                    instance_id = instance.get('instance_id')
                    if not instance_id:
                        _logger.warning("实例缺少instance_id，跳过处理")
                        continue
                    if instance_id in instance_id_to_messages:
                        instance_id_to_messages[instance_id].append(message_item)
                    else:
                        instance_id_to_messages[instance_id] = [message_item]
            else:
                unmatched_message_items.append(message_item)

        separated_messages_dict: Dict[str, List] = {"messages": [], "context_messages": []}
        if unmatched_message_items:
            separated_messages_dict["messages"] = messages.copy()
            separated_messages_dict["context_messages"] = context_messages.copy()
            _logger.info("检测到 %s 条未匹配消息，已分开存储 %s 条消息和 %s 条上下文消息",
                         len(unmatched_message_items), len(messages), len(context_messages))
        else:
            _logger.info("所有消息都已匹配到现有实例")

        unmatched_messages_by_node: Dict[ClassNode, Dict[str, List]] = {self: separated_messages_dict}

        _logger.info("识别到需要更新的实例数量: %s", len(instance_id_to_messages))
        _logger.info("识别到需要新增的消息片段数量: %s (关联到类节点: %s)",
                     len(unmatched_message_items), self.class_id)

        return instance_id_to_messages, unmatched_messages_by_node

    def update_relevant_instances(self, instance_id_to_messages: Dict[str, List[str]], use_hash: bool = False):
        """
        更新相关实例，通过 prompt 完成实例数据的更新（或 use_hash 时仅文本拼接）。

        Args:
            instance_id_to_messages: 实例 ID -> 消息列表（消息为 dict，含 'message' 等）
            use_hash: 若 True 使用 update_data_from_messages_hash，不调用 LLM
        """
        from src.data.instance import update_data_from_messages_hash
        for instance_id, messages in instance_id_to_messages.items():
            # 在self._instances中查找对应instance_id的实例
            found_index = -1
            existing_instance = None

            for i, inst in enumerate(self._instances):
                # 安全地获取实例ID，兼容字典和对象两种格式
                if isinstance(inst, dict):
                    current_id = inst.get('instance_id')
                else:
                    current_id = getattr(inst, 'instance_id', None)

                if current_id == instance_id:
                    found_index = i
                    existing_instance = inst if isinstance(inst, dict) else None
                    break

            if found_index == -1 or existing_instance is None:
                _logger.warning(f"未找到实例ID为 {instance_id} 的实例，跳过更新")
                continue
            if use_hash:
                updated_instance = update_data_from_messages_hash(existing_instance, messages)
            else:
                message_strings = [m.get("message", m) if isinstance(m, dict) else str(m) for m in messages]
                updated_instance = update_data_from_messages(existing_instance, message_strings)
                existing_messages = existing_instance.get('messages', []) if isinstance(existing_instance, dict) else []
                updated_instance['messages'] = existing_messages + messages
            self._instances[found_index] = updated_instance
        self._class_formatter_by_instance()
        _logger.info(f"已完成 {len(instance_id_to_messages)} 个实例的更新")

    def add_instances(self, unmatched_messages_by_node: Dict[ClassNode, Dict[str, List[str]]], use_hash: bool = False):
        """
        在当前类节点下新增实例。

        Args:
            unmatched_messages_by_node: 类节点 -> {messages, context_messages}，用于创建新实例
            use_hash: 若 True 使用 create_instances_from_messages_hash，不调用 LLM
        """
        from src.data.instance import create_instances_from_messages_hash
        current_instance_count = len(self._instances)
        instances_added = 0

        for _class_node, messages_dict in unmatched_messages_by_node.items():
            messages = messages_dict.get("messages", [])
            context_messages = messages_dict.get("context_messages", [])

            if not messages:
                _logger.info("没有需要创建实例的消息")
                continue

            if use_hash:
                new_instances = create_instances_from_messages_hash(messages, context_messages, _class_node)
            else:
                new_instances = create_instances_from_messages(
                    messages,
                    context_messages,
                    _class_node
                )

            for instance in new_instances:
                instance["instance_id"] = f"instance_{current_instance_count + 1}"
                # 将字典实例添加到实例列表
                self._instances.append(instance)
                current_instance_count += 1
                instances_added += 1
                _logger.info(f"已创建新实例: {instance['instance_id']} - {instance.get('instance_name', '未命名实例')}")

            # 如果有新增实例，重新格式化类
            if instances_added > 0:
                self._class_formatter_by_instance()
                _logger.info(f"成功添加 {instances_added} 个新实例到类 {self.class_id}")
            else:
                _logger.info("没有新增实例")

    def _class_formatter_by_instance(self):
        """根据实例数据重新格式化类的属性、操作和未分类字段"""
        try:
            # 修正1：正确处理attributes - 遍历值而非键
            attributes_descriptions = []
            for inst in self._instances:
                if isinstance(inst, dict) and "attributes" in inst:
                    # 遍历attributes字典的值（即嵌套字典）
                    for attr_value in inst["attributes"].values():
                        if isinstance(attr_value, dict) and "description" in attr_value:
                            attributes_descriptions.append(attr_value["description"])

            self.attributes = set(attributes_descriptions)

            # 修正2：正确处理operations
            operations_descriptions = []
            for inst in self._instances:
                if isinstance(inst, dict) and "operations" in inst:
                    for op_value in inst["operations"].values():
                        if isinstance(op_value, dict) and "description" in op_value:
                            operations_descriptions.append(op_value["description"])

            self.operations = set(operations_descriptions)

            # 修正3：正确处理uninstance_field（注意字段名修正）
            uninstance_values = []
            for inst in self._instances:
                if isinstance(inst, dict) and "uninstance_field" in inst:
                    field_value = inst["uninstance_field"]
                    if field_value:  # 确保值不为空
                        uninstance_values.append(field_value)

            self.unclassified = set(uninstance_values)  # 或者保持字段名一致

            _logger.info(
                f"格式化完成: {len(self.attributes)} 个属性, {len(self.operations)} 个操作, {len(self.unclassified)} 个未分类字段")

        except Exception as e:
            _logger.error(f"格式化实例时发生错误: {e}")
            # 设置默认值避免后续错误
            self.attributes = set()
            self.operations = set()
            self.unclassified = set()



    def process_classnode_initialization(self,
                                         messages: List,
                                         context_messages: List[str],
                                         threshold: float,
                                         use_hash: bool = False) -> None:
        """根据消息初始化类节点属性与实例；新增类时仅会得到未匹配消息并创建新实例。"""
        _logger.info("新节点信息: %s", messages)
        _logger.info("新节点上下文: %s", context_messages)
        instance_id_to_messages, unmatched_messages_by_node = self.get_message_allocation_from_instance(
            messages, context_messages, threshold=0.0)
        assert len(instance_id_to_messages) == 0, "new classnode should have no relevant instances"
        self.add_instances(unmatched_messages_by_node, use_hash=use_hash)



