import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
from string import Template
from src.assist import similarity_score, calculate_tfidf_similarity, build_instance_fragments, fetch_default_llm_model, \
    serialize_instance
from src.data.instance import update_data_from_messages, create_instances_from_messages
from src.logger import setup_logger
from networkx import Graph
from src.unclass.prompts_unclass import PROMPT_CREATE_INSTANCE_UNCLASS, PROMPT_UPDATE_INSTANCE

_logger = setup_logger("instance_graph")


class InstanceGraph:
    def __init__(self,
                 instance_sense_threshold: float = 0.7,
                 llm=None):  # llm从外部传入
        self._llm = llm
        self.graph: Graph = Graph()
        self._instance_sense_threshold = instance_sense_threshold

        self.name = ""
        self.description = ""
        self.edges = []  # 存储边
        self.message_labels = []  # 存储每个信息被分配的标签
        self.warning_items = []  # 存储警告信息
        self.tags = []  # 存储每个实例的tags
        self.selected_instance_keys = set()  # 存储已选中实例的唯一标识
        self._all_instances = []  # 存储所有实例
        self._llm = fetch_default_llm_model()
        self.filepath = ""

    def sense_instances(
            self,
            data: list,
            context: list,
            tfidf_threshold: float = 0.9,
            top_k: int = None  # 这个参数现在可选，不使用时设为None
    ) -> tuple[dict, list]:
        """
        去掉类结构，直接处理实例的混合感知方法

        Args:
            data: 信息片段列表，每个元素包含'message'和'label'
            context: 背景片段列表
            tfidf_threshold: TF-IDF相似度阈值
            top_k_instance: 每个信息片段最多匹配的实例数
        """
        # 第一阶段：使用TF-IDF匹配现有实例
        instances_to_update = {}  # 需要更新的实例 {instance_id: {instance_data, messages}}
        unmatched_data = []  # 未匹配到任何实例的信息片段
        processed_labels = set()  # 已处理的信息片段标签

        for item in data:
            message = item.get("message", "")
            label = item.get("label", "")

            if label in processed_labels:
                continue

            # 匹配现有实例
            matched_instances = self._match_instances_by_tfidf(
                query=message,
                all_instances=self._all_instances,  # 存储所有实例的列表
                threshold=tfidf_threshold
            )

            if matched_instances:
                # 该信息片段匹配到了一个或多个现有实例
                for instance_info in matched_instances:
                    instance_id = instance_info.get("instance_id")
                    if not instance_id:
                        continue

                    # 初始化该实例在instances_to_update中的条目
                    if instance_id not in instances_to_update:
                        instances_to_update[instance_id] = {
                            "instance_data": self._get_instance_by_id(instance_id),
                            "messages": []
                        }

                    # 添加信息片段
                    instances_to_update[instance_id]["messages"].append({
                        "message": message,
                        "label": label
                    })

                processed_labels.add(label)
            else:
                # 该信息片段未匹配到任何现有实例
                unmatched_data.append(item)

        _logger.info(
            f"TF-IDF匹配结果: 匹配到{len(instances_to_update)}个实例，{len(processed_labels)}个信息片段，{len(unmatched_data)}个片段未匹配")

        # 第二阶段：对未匹配的片段使用LLM创建新实例
        new_instances = []

        if unmatched_data:
            _logger.info(f"对{len(unmatched_data)}个未匹配片段使用LLM创建新实例")

            # 构建LLM提示词，只处理未匹配的片段
            add_instance_prompt = Template(PROMPT_CREATE_INSTANCE_UNCLASS).substitute(
                related_messages="\n".join([f"- {msg}" for msg in unmatched_data]),
                context_messages="\n".join([f"- {msg}" for msg in context])
            )

            _logger.info(f"LLM创建实例提示词: {add_instance_prompt}")

            response = self._llm.invoke(add_instance_prompt)
            _logger.info(f"LLM创建实例响应: {response.content}")
            result = json.loads(response.content)

            # 处理LLM返回的新实例
            new_instances = result
            if not isinstance(new_instances, list):
                new_instances = []

        _logger.info(f"最终结果: 需要更新的实例: {len(instances_to_update)}个, 需要新增的实例: {len(new_instances)}个")
        return instances_to_update, new_instances

    def _match_instances_by_tfidf(
            self,
            query: str,
            all_instances: list,
            threshold: float = 0.9
    ) -> list:
        """
        通过TF-IDF匹配现有实例
        """
        if not all_instances:
            return []

            # 1. 构建所有实例的文本片段
        instance_documents = []  # 存储每个片段的文本
        instance_objects = []  # 存储实例对象
        fragment_instance_map = {}  # 映射片段索引到实例索引

        for instance_idx, instance in enumerate(all_instances):
            # 构建实例的文本片段
            fragments = build_instance_fragments(instance)

            # 为每个片段添加到文档列表
            for fragment_type, fragment_text in fragments:
                if fragment_text and fragment_text.strip():  # 只添加非空片段
                    fragment_idx = len(instance_documents)
                    instance_documents.append(fragment_text)

                    # 记录片段到实例的映射
                    fragment_instance_map[fragment_idx] = instance_idx

        if not instance_documents:
            _logger.warning("没有可检索的实例片段")
            return []

        # 2. 计算TF-IDF相似度
        try:
            similarities, _, _ = calculate_tfidf_similarity(query, instance_documents)
        except Exception as e:
            _logger.error(f"TF-IDF计算失败: {e}")
            return []

        # 3. 为每个实例收集其所有片段的相似度
        instance_similarities = {}  # {instance_idx: [similarity1, similarity2, ...]}

        for fragment_idx, similarity in enumerate(similarities):
            instance_idx = fragment_instance_map.get(fragment_idx)
            if instance_idx is not None:
                if instance_idx not in instance_similarities:
                    instance_similarities[instance_idx] = []
                instance_similarities[instance_idx].append(similarity)

        # 4. 获取所有超过阈值的实例
        matching_instances = []

        for instance_idx, similarities_list in instance_similarities.items():
            if similarities_list:
                # 取该实例所有片段中的最高相似度
                max_similarity = max(similarities_list)

                if max_similarity >= threshold:
                    instance = all_instances[instance_idx]
                    matching_instances.append({
                        "instance_id": instance.get("instance_id"),
                        "instance_data": instance,
                        "similarity": float(max_similarity)
                    })

        # 5. 按相似度排序
        matching_instances.sort(key=lambda x: x["similarity"], reverse=True)

        _logger.info(f"TF-IDF实例匹配: 查询'{query[:50]}...' 匹配到{len(matching_instances)}个实例")
        return matching_instances

    def _get_instance_by_id(self, instance_id: str) -> dict:
        """
        根据实例ID获取实例
        """
        for instance in self._all_instances:
            if instance.get("instance_id") == instance_id:
                return instance
        return {}

    def process_instances(
            self,
            instances_to_update: dict,
            new_instances: list
    ) -> None:
        """
        处理实例的更新和创建
        """
        # 1. 更新现有实例
        updated_count = 0
        for instance_id, data in instances_to_update.items():
            instance_data = data["instance_data"]
            messages = data["messages"]

            if instance_data and messages:
                # 提取消息内容列表
                message_contents = []
                for msg in messages:
                    if isinstance(msg, dict) and "message" in msg:
                        message_contents.append(msg["message"])
                    elif isinstance(msg, str):
                        message_contents.append(msg)

                if message_contents:
                    # 使用update_data_from_messages函数更新实例
                    # 通过prompt更新实例，消息分配已在prompt中处理
                    updated_instance = self.update_data_from_messages(
                        instance=instance_data,
                        messages=message_contents
                    )

                    # 保存原来的instance_id
                    original_instance_id = instance_data.get("instance_id")
                    updated_instance["instance_id"] = original_instance_id

                    # 更新实例在all_instances中的位置
                    for i, inst in enumerate(self._all_instances):
                        if inst.get("instance_id") == original_instance_id:
                            self._all_instances[i] = updated_instance
                            updated_count += 1
                            _logger.info(f"更新实例 {original_instance_id}")
                            break
                else:
                    _logger.warning(f"实例 {instance_id} 没有有效的消息内容，跳过更新")

        # 2. 创建新实例
        created_count = 0
        for new_instance_data in new_instances:
            # 创建新实例
            new_instance = new_instance_data
            if new_instance:
                # 为新实例分配ID
                new_instance["instance_id"] = f"instance_{len(self._all_instances) + 1}"

                # 将实例添加到实例列表
                self._all_instances.append(new_instance)
                created_count += 1
                _logger.info(f"创建新实例 {new_instance.get('instance_id')}")

        _logger.info(f"处理完成: 更新{updated_count}个实例, 创建{created_count}个新实例")

        # 3. 保存快照
        self._save_instances_snapshot()

    def update_data_from_messages(self, instance, messages: List[str]):
        """通过prompt更新实例数据，消息分配已在prompt中处理"""
        align_update_prompt = Template(PROMPT_UPDATE_INSTANCE).substitute(
            update_message="\n".join([f"- {msg}" for msg in messages]),
            instance=instance
        )

        _logger.info(f"ALIGN_UPDATE_PROMPT: {align_update_prompt}")
        response = self._llm.invoke(align_update_prompt)
        _logger.info(f"ALIGN_UPDATE_RESPONSE: {response.content}")
        updated_instances = json.loads(response.content)

        return updated_instances

    def create_instances_from_messages(self, messages: List[str], context_messages: List[str]):
        """通过prompt从消息创建实例，消息分配已在prompt中处理"""
        align_add_prompt = Template(PROMPT_CREATE_INSTANCE_UNCLASS).substitute(
            related_messages="\n".join([f"- {msg}" for msg in messages]),
            context_messages="\n".join([f"- {msg}" for msg in context_messages])
        )

        _logger.info(f"ALIGN_ADD_PROMPT: {align_add_prompt}")
        response = self._llm.invoke(align_add_prompt)
        _logger.info(f"ALIGN_ADD_RESPONSE: {response.content}")

        # 解析LLM返回的实例信息
        instances_data = json.loads(response.content)
        return instances_data

    def _save_instances_snapshot(self) -> None:
        """
        保存实例快照
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        operation = self.filepath
        filename = f"D:/model/conv/GraphConv/results/instances/instances_snapshot_{operation}_{timestamp}.json"

        # snapshot_data = {
        #     "timestamp": datetime.now().isoformat(),
        #     "total_instances": len(self._all_instances),
        #     "instances": self._all_instances
        # }

        # 保存到JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self._all_instances, f, indent=2, ensure_ascii=False)

        _logger.info(f"实例快照已保存: {filename}")

    def _save_instances_edge_snapshot(self) -> None:
        """
        保存实例快照
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        operation = self.filepath
        filename = f"D:/model/conv/GraphConv/results/instances/edges_snapshot_{operation}_{timestamp}.json"


        # 保存到JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.edges, f, indent=2, ensure_ascii=False)

        _logger.info(f"实例快照已保存: {filename}")



    def update_instance_relationships(
            self,
            data: list
    ) -> None:
        """
        更新实例间的关系
        """
        for data_item in data:
            message_content = data_item['message']
            message_label = data_item['label']

            # 查找包含当前标签的实例
            matching_instances = []

            for instance in self._all_instances:
                if not isinstance(instance, dict):
                    continue

                # 检查实例的消息标签
                instance_labels = []
                for msg in instance.get("messages", []):
                    if isinstance(msg, dict) and "label" in msg:
                        instance_labels.append(msg["label"])

                if message_label in instance_labels:
                    matching_instances.append(instance)

            # 如果找到至少2个匹配的实例，则建立关联
            if len(matching_instances) >= 2:
                # 创建edge记录
                edge_record = {
                    "content": message_content,
                    "label": message_label,
                    "connections": []
                }

                # 为每个匹配的实例构建连接信息
                for instance in matching_instances:
                    connection_info = {
                        "instance_id": instance.get("instance_id"),
                        "instance_name": instance.get("instance_name", "")
                    }
                    edge_record["connections"].append(connection_info)

                # 添加到edges列表
                if not hasattr(self, 'edges'):
                    self.edges = []
                self.edges.append(edge_record)

                # 为每个实例添加关联
                for instance in matching_instances:
                    if "relationships" not in instance:
                        instance["relationships"] = []

                    # 添加与其他实例的关联
                    for other_instance in matching_instances:
                        if other_instance["instance_id"] != instance["instance_id"]:
                            relationship = {
                                "related_instance_id": other_instance["instance_id"],
                                "content": message_content,
                                "label": message_label
                            }

                            # 避免重复
                            if relationship not in instance["relationships"]:
                                instance["relationships"].append(relationship)

                _logger.info(f"为消息标签 {message_label} 建立了 {len(matching_instances)} 个实例间的关联")
        self._save_instances_edge_snapshot()
        _logger.info(f"关系更新完成：建立了 {len(self.edges) if hasattr(self, 'edges') else 0} 个关联边")
    def _search_by_sub_hash(self,
                            query,
                            top_k_instances=30,
                            ):

        # 3. 在全图中搜索实例，排除已找到的实例
        relv_ins = self._fetch_instances_by_tfidf(query, top_k_instances, threshold=0.1)
        return relv_ins

    def _fetch_instances_by_tfidf(self, query, top_k_instances, threshold) -> str:
        """
        基于TF-IDF向量化的实例检索方法

        Args:
            query: 查询字符串
            top_k_instances: 返回的实例数量上限
            threshold: 相似度阈值

        Returns:
            str: 相关实例列表
        """
        # 1. 从所有实例中收集实例及其片段
        all_instances = []  # 存储实例对象
        instance_documents = []  # 存储每个片段的文本表示
        fragment_instance_map = {}  # 存储片段索引到实例索引的映射
        instance_keys_map = {}  # 存储实例索引到唯一标识的映射

        # 从self._all_instances中收集所有实例
        for instance_idx, instance in enumerate(self._all_instances):
            all_instances.append(instance)
            _logger.debug(f"用于去构建片段的实例: {instance.get('instance_id', 'unknown')}")

            # 为实例的每个片段构建文本表示
            fragments = build_instance_fragments(instance)

            _logger.debug(f"当前实例的片段数量: {len(fragments)}")

            for fragment_type, fragment_text in fragments:
                if fragment_text and fragment_text.strip():  # 只添加非空片段
                    fragment_idx = len(instance_documents)
                    instance_documents.append(fragment_text)

                    # 记录片段到实例的映射
                    fragment_instance_map[fragment_idx] = {
                        'instance_idx': instance_idx,
                        'fragment_type': fragment_type,
                        'fragment_text': fragment_text
                    }

        if not all_instances or not instance_documents:
            _logger.warning("未找到任何实例或实例片段")
            return ""

        _logger.info(f"TF-IDF实例检索: 总共收集到 {len(all_instances)} 个实例，{len(instance_documents)} 个片段")

        # 2. 使用TF-IDF进行向量化和相似度计算
        try:
            similarities, vectorizer, tfidf_matrix = calculate_tfidf_similarity(query, instance_documents)
        except Exception as e:
            _logger.error(f"TF-IDF计算失败: {e}")
            return ""

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
            return ""

        # 4. 将实例按最高相似度排序
        instance_scores = list(instance_max_scores.items())
        instance_scores.sort(key=lambda x: x[1], reverse=True)

        _logger.info(f"实例相似度统计: 共{len(instance_scores)}个实例有匹配片段")

        # 5. 按阈值过滤并选择实例
        above_threshold = [(idx, score) for idx, score in instance_scores if score >= threshold]
        below_threshold = [(idx, score) for idx, score in instance_scores if score < threshold]

        log_msg = f"相似度统计: 高于阈值({threshold})实例数: {len(above_threshold)}, 低于阈值实例数: {len(below_threshold)}"
        _logger.info(log_msg)

        # 6. 根据新逻辑选择实例
        if top_k_instances <= 0:
            return ""
        above_count = len(above_threshold)
        if len(above_threshold) >= top_k_instances:
            # 情况1: 高于阈值的实例足够
            selected_instance_scores = above_threshold[:top_k_instances]
            _logger.info(f"高于阈值实例充足，直接返回前{len(selected_instance_scores)}个")
        else:
            # 情况2: 高于阈值的实例不足
            num_needed = top_k_instances - above_count
            _logger.info(f"高于阈值实例只有{above_count}个，需要从低于阈值实例中补充{num_needed}个")

            # 计算可以从低于阈值实例中获取的最大数量
            available_below = len(below_threshold)

            if available_below >= num_needed:
                # 有足够的低于阈值实例来补足
                supplementary_instances = below_threshold[:num_needed]
                selected_instance_scores = above_threshold + supplementary_instances
                _logger.info(f"低于阈值实例充足，用{len(supplementary_instances)}个补足到{top_k_instances}个")
            else:
                # 低于阈值实例也不足，只能返回尽可能多的实例
                supplementary_instances = below_threshold[:available_below]
                selected_instance_scores = above_threshold + supplementary_instances
                _logger.warning(
                    f"低于阈值实例也不足，只能返回{len(selected_instance_scores)}个实例，而不是期望的{top_k_instances}个")

        _logger.info(f"最终选中的实例数量: {len(selected_instance_scores)}")

        # 7. 创建清理后的实例副本
        cleaned_instances = []
        for instance_idx, score in selected_instance_scores:
            original_instance = all_instances[instance_idx]

            # 获取最佳片段信息
            best_fragment = instance_best_fragments.get(instance_idx, {})

            # 获取实例的唯一标识并记录到全局集合
            instance_key = instance_keys_map.get(instance_idx)
            if instance_key:
                self.selected_instance_keys.add(instance_key)

            # 创建不包含敏感字段的实例副本
            # 注意：这里去掉了'message_labels'过滤，因为现在的实例可能没有这个字段
            # 可以根据需要添加其他需要过滤的字段
            cleaned_instance = original_instance.copy()

            # 删除可能不需要的字段
            fields_to_remove = ['message_labels', 'internal_data']  # 根据需要调整
            for field in fields_to_remove:
                if field in cleaned_instance:
                    del cleaned_instance[field]

            # 添加TF-IDF相似度分数和匹配片段信息
            cleaned_instance['similarity_score'] = float(score)
            cleaned_instance['matched_fragment'] = {
                'type': best_fragment.get('fragment_type', 'unknown'),
                'content': best_fragment.get('fragment_text', ''),
                'similarity_score': float(score)
            }

            # 由于没有类结构，我们可以添加一个空的类信息或完全去掉
            # 这里选择添加一个默认的类信息结构
            cleaned_instance['class_info'] = {
                'class_id': 'default',
                'class_name': 'Default Class',
                'fully_covering': True,  # 设置为True，表示这个实例是直接匹配的
                'stage': 0
            }

            cleaned_instances.append(cleaned_instance)

            # 记录日志
            instance_id = original_instance.get('instance_id', f'instance_{instance_idx}')
            _logger.info(f"选择实例 {instance_id}: 相似度={score:.4f}, "
                         f"匹配片段类型={best_fragment.get('fragment_type', 'unknown')}")

        return serialize_instance(cleaned_instances)