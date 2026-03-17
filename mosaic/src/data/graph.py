# Define the data structures and functions for the class graph
from __future__ import annotations

import ast
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple

from src.data.classnode import ClassNode
from src.prompts_en import *
from src.prompts_ch import PROMPT_CONFLICT, PROMPT_TAGS, PROMPT_TAGS_QUERY

from src.assist import *
from networkx import Graph
import json
from keybert import KeyBERT
from src.logger import setup_logger

_logger = setup_logger("graph cudr")


class ClassGraph:
    def __init__(self,
                 instance_sense_threshold: float = 0.7,
                 llm=fetch_default_llm_model()):
        self._llm = llm
        self.graph: Graph = Graph()
        self._instance_sense_threshold = 0.7
        self._built_in = ["unclassified", "attributes", "operations"]
        self.name = ""
        self.description = ""
        self.edges = []  #存储边
        self.message_labels = []  #存储每个信息被分配的标签，外部传入，用于consistency_vaild_dynamic函数
        self.warning_items = []  #存储警告信息

        self.tags = []  #用来存储每个实例的tags
        self.filepath = "" #存储构图过程中的图文件
        # 图持久化目录：优先实例属性，其次环境变量 GRAPH_SAVE_DIR，否则当前目录下 results/graph
        self._graph_save_dir = os.environ.get("GRAPH_SAVE_DIR") or os.path.join(os.getcwd(), "results", "graph")

        self.selected_instance_keys = set()  # 用于存储已选中实例的唯一标识

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

    ###感知信息和现有类，看哪些类需要更新，哪些类需要新增。对于需要更新的类，保留更新的类id和对应这个类的信息片段和背景片段；对于需要新增的类，保留这个新增类的类名和新增类的信息片段和背景片段
    def sense_classes(self,
                      data: list,
                      context: list,
                      tfidf_threshold: float = 0.5,  # TF-IDF相似度阈值
                      top_k_class: int = 3,  # 每个片段最多匹配的类数
                      ) -> tuple[dict[Any, dict[str, Any]], dict[Any, dict[str, Any]]]:
        """
        混合感知方法：先用TF-IDF匹配现有类，再用LLM创建新类

        Args:
            data: 信息片段列表，每个元素包含'message'和'label'
            context: 背景片段列表
            tfidf_threshold: TF-IDF相似度阈值
            top_k_class: 每个信息片段最多匹配的类数量
        """
        # 第一阶段：使用TF-IDF匹配现有类
        relv_msg = {}  # 需要更新的已有类
        unmatched_data = []  # 未匹配到任何类的信息片段
        processed_labels = set()  # 已处理的信息片段标签，用于跟踪

        for item in data:
            message = item.get("message", "")
            label = item.get("label", "")

            if label in processed_labels:
                continue

            # 使用TF-IDF匹配相似类
            tfidf_result = self._sense_classes_by_tfidf(
                query=message,
                top_k_class=top_k_class,
                threshold=tfidf_threshold,
                allow_below_threshold=False
            )

            selected_classes = tfidf_result.get("selected_classes", [])

            if selected_classes:
                # 该信息片段匹配到了一个或多个现有类
                for class_info in selected_classes:
                    class_id = class_info.get("class_id")
                    if not class_id:
                        continue

                    # 初始化该类在relv_msg中的条目
                    if class_id not in relv_msg:
                        relv_msg[class_id] = {
                            "related_message": [],
                            "dependent_context": []
                        }

                    # 添加信息片段
                    relv_msg[class_id]["related_message"].append({
                        "message": message,
                        "label": label
                    })

                    # # 查找相关的背景片段（可选，可根据需要实现）
                    # related_context = self._find_relevant_context(message, context)
                    # if related_context:
                    #     relv_msg[class_id]["dependent_context"].extend(related_context)

                processed_labels.add(label)
            else:
                # 该信息片段未匹配到任何现有类
                unmatched_data.append(item)

        _logger.info(
            f"TF-IDF匹配结果: 匹配到{len(relv_msg)}个类，{len(processed_labels)}个信息片段，{len(unmatched_data)}个片段未匹配")

        # 第二阶段：对未匹配的片段使用LLM创建新类
        unknown_msg = {}

        if unmatched_data:
            _logger.info(f"对{len(unmatched_data)}个未匹配片段使用LLM创建新类")

            # 构建LLM提示词，只处理未匹配的片段
            sense_prompt = Template(PROMPT_NEW_CLASS_SENSE).substitute(
                {
                    "DATA": unmatched_data,  # 只传入未匹配的片段
                    "CONTEXT": context
                }
            )

            _logger.info(f"LLM感知提示词: {sense_prompt}")

            response = self._llm.invoke(sense_prompt)
            _logger.info(f"LLM感知响应: {response.content}")
            result = json.loads(response.content)

            # 处理LLM返回的新类
            new_classes = result.get("new_classes", [])
            if not isinstance(new_classes, list):
                new_classes = []

            for item in new_classes:
                class_name = item.get("class_name")
                if class_name:
                    unknown_msg[class_name] = {
                        "related_message": item.get("related_message", []),
                        "dependent_context": item.get("dependent_context", [])
                    }
        _logger.info(f"最终结果: 需要更新的相关类: {len(relv_msg)}个, 需要新增的新类: {len(unknown_msg)}个")
        return relv_msg, unknown_msg

    ##处理和新信息相关的类，相关的信息片段需要更新类下面的部分实例或者在相关类下面新增实例。
    def process_relvclass_instances(self,
                                    relv_classes_msg: Dict[str, Dict[str, List[str]]],
                                    threshold: float = 0.9
                                    ) -> list[ClassNode]:
        """
        处理相关类中的实例，根据阈值筛选相关实例和未知实例

        Args:
            relv_classes: 从sense_classes返回的相关类字典，包含类ID和对应的消息
            threshold: 相似度阈值，用于判断实例相关性
        """
        processed_classes = []

        for class_id, messages_dict in relv_classes_msg.items():
            # 通过类ID找到对应的类节点
            class_node = self.get_classnode(class_id)
            if class_node is None:
                _logger.warning(f"找不到对应的类节点: {class_id}")
                continue

            # 提取相关消息和上下文消息
            relv_messages = messages_dict.get("related_message", [])
            context_messages = messages_dict.get("dependent_context", [])

            relv_instances, unknown_instances = class_node.get_message_allocation_from_instance(
                relv_messages, context_messages, threshold)

            # 更新类的实例
            class_node.update_relv_instances(relv_instances)
            # 在该类下面新增实例
            class_node.add_instances(unknown_instances)

            processed_classes.append(class_node)

            _logger.info(f"Processed class {class_id}: "
                         f"found {len(relv_instances)} relevant instances, "
                         f"{len(unknown_instances)} new instances")
        self.save_graph_snapshot()
        return processed_classes

    ##对于第一步感知出来的需要新创建的类，去对这些类进行实例的创建和类节点创建和类id的分配。
    def add_classnodes(self,
                       classnodesmsg: Dict[str, Dict[str, List[str]]]):
        """
        对于第一步感知出来的需要新创建的类，去对这些类进行实例的创建和类节点创建和类id的分配。

        Args:
            classnodes: 从sense_classes返回的新类字典，包含类名和对应的消息片段
        """
        curr_class_count = len(self.graph.nodes)
        new_classes = []
        # 遍历新类字典，键现在是类名字符串，值包含相关信息
        for class_name, content in classnodesmsg.items():
            # 创建新的ClassNode对象
            classnode = ClassNode.new_classnode(class_name)
            # 分配类ID
            class_id = f"class_{curr_class_count + 1}"
            classnode.class_id = class_id

            # 处理类节点的初始化，创建实例
            classnode.process_classnode_initialization(
                content.get("related_message", []),
                content.get("dependent_context", []),
                threshold=0
            )
            # 将类节点添加到图中
            self.graph.add_node(classnode)
            _logger.info(f"Added new class {class_id}: {class_name}")
            curr_class_count += 1
            new_classes.append(classnode)

        self.save_graph_snapshot()
        return new_classes

    def save_graph_snapshot(self) -> None:
        """
        实时保存图节点的快照信息，用于调试和验证存储是否正确

        Args:
            operation: 操作描述，用于标识保存时机
        """
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.now().strftime("%Y%m%d")
        operation = self.filepath
        os.makedirs(self._graph_save_dir, exist_ok=True)
        filename = os.path.join(self._graph_save_dir, f"graph_snapshot_{operation}_{timestamp}.json")

        snapshot_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "graph_info": {
                "total_classes": len(self.graph.nodes),
                "total_edges": len(self.graph.edges)
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

        _logger.info(f"图快照已保存: {filename}")

        # 2. 新增：保存整个图结构
        self._save_complete_graph(operation, timestamp)

        #3.保存边
        filename = os.path.join(self._graph_save_dir, f"graph_edge_{operation}_{timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.edges, f, indent=2, ensure_ascii=False)
        _logger.info(f"图边已保存: {filename}")

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
        _logger.info(f"图完整结构已经保存 {filename}")

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
        all_classes = processed_classes_safe + new_classes_safe

        # 遍历data中的每条信息
        for data_item in data:
            message_content = data_item['message']
            message_label = data_item['label']

            # 存储与当前消息标签匹配的实例
            matching_instances = []
            _logger.info(f"message_label:{message_label};data_item: {data_item}")

            # 在所有类节点中查找包含当前标签的实例
            for class_node in all_classes:
                if not hasattr(class_node, '_instances') or not class_node._instances:
                    continue

                for instance_dict in class_node._instances:  # 重命名变量，明确它是字典
                    #_logger.info(f"instance_dict: {instance_dict}")
                    #_logger.info(f"instance_message_label: {instance_dict['message_labels']}")

                    # 检查实例的message_labels是否包含当前标签
                    if message_label in instance_dict['message_labels']:
                        # _logger.info(f"message_label matched: {message_label}")
                        matching_instances.append({
                            'class_node': class_node,
                            'instance': instance_dict
                        })

            #_logger.info(f"matching_instances: {matching_instances}")

            # 如果找到至少2个匹配的实例，则建立关联
            if len(matching_instances) >= 2:
                # 创建edge记录
                edge_record = {
                    "content": message_content,
                    "label": message_label,
                    "connections": []
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

                _logger.info(f"为消息标签 {message_label} 建立了 {len(matching_instances)} 个实例间的关联")

        # 记录处理结果
        _logger.info(f"关系更新完成：建立了 {len(self.edges)} 个关联边")
        _logger.info(f"处理了 {len(data)} 条数据，涉及 {len(all_classes)} 个类节点")

        self.save_graph_snapshot()

    def message_passing(self):
        pass

    def consistency_vaild_dynamic(self, relv_msg, unknown_msg):
        """
        分类已有类和新增类的信息，不进行冲突检查，只做好信息分组

        Args:
            relv_msg: 与已有类相关的信息字典
            unknown_msg: 需要新增的类相关信息字典

        Returns:
            dict: 包含分类信息的字典，仅包含已有类和新增类的分组信息
        """
        classified_data = {
            "existing_classes": {},
            "new_classes": {}
        }

        # 1. 分类已有类（relv_msg）的信息
        for class_id, class_info in relv_msg.items():
            # 在graph nodes中查找对应的类节点
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
                # 创建标签到消息的映射字典，便于快速查找
                label_to_message = {}
                for msg_item in self.message_labels:
                    label_to_message[msg_item['label']] = msg_item['message']

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

        # 2. 分类新增类（unknown_msg）的信息
        for class_name, class_info in unknown_msg.items():
            new_class_data = {
                "class_name": class_name,
                "new_messages": class_info.get("related_message", []),
                "dependent_context": class_info.get("dependent_context", []),
            }
            classified_data["new_classes"][class_name] = new_class_data

        # 3. 记录分类结果用于调试
        _logger.info(f"信息分类完成：")
        _logger.info(f"- 已有类: {len(classified_data['existing_classes'])}")
        _logger.info(f"- 新增类: {len(classified_data['new_classes'])}")

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
                    _logger.info(f"conflict prompt: {conflict_prompt}")
                    response = self._llm.invoke(conflict_prompt)
                    _logger.info(f"conflict prompt response: {response.content}")
                    result = json.loads(response.content)

                    if result.get("is_conflict", False):
                        # 获取所有冲突涉及的消息标签
                        conflict_labels = set()
                        for conflict in result.get("conflicts", []):
                            conflict_labels.update(conflict.get("conflict_message_labels", []))

                        # 只筛选出冲突相关的消息
                        conflict_messages = []

                        for msg in all_messages:
                            if msg.get("label") in conflict_labels:
                                conflict_messages.append(msg)
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
                    _logger.info(f"conflict prompt: {conflict_prompt}")
                    response = self._llm.invoke(conflict_prompt)
                    _logger.info(f"conflict prompt response: {response.content}")
                    result = json.loads(response.content)

                    if result.get("is_conflict", False):

                        # 获取所有冲突涉及的消息标签
                        conflict_labels = set()
                        for conflict in result.get("conflicts", []):
                            conflict_labels.update(conflict.get("conflict_message_labels", []))

                        # 只筛选出冲突相关的消息
                        conflict_messages = []
                        for msg in messages:
                            if msg.get("label") in conflict_labels:
                                conflict_messages.append(msg)

                        items = {
                            "class_id": class_data.get("class_id", class_name),
                            "class_name": class_data.get("class_name", class_name),
                            "class_type": "new_class",
                            "is_conflict": True,
                            "conflicts": result.get("conflicts", []),
                            "messages": conflict_messages
                        }
                        self.warning_items.append(items)

        debug_filename = "D:/model/conv/GraphConv/oop_graph/src/error_case/conflict/warning_items.json"
        with open(debug_filename, 'w', encoding='utf-8') as f:
            # 简化输出，避免循环引用
            json.dump(self.warning_items, f, indent=2, ensure_ascii=False)

    #用llm的方式去搜索
    def _search_by_sub_llm(self,
                           query,
                           llm,
                           top_k_class=3,
                           top_k_instances=7,
                           ):
        # 重置已选实例集合
        self.selected_instance_keys.clear()
        # 1. 在指定类中搜索实例
        rel_tag = self.find_kw_relvinstance_tags(query)
        sensed_classes = self._sense_classes_by_llm(query, llm, top_k_class)
        relevant_instances = self._fetch_instances_by_tfidf(query, top_k_instances, threshold=0.5,
                                                            classes=sensed_classes)
        # 3. 合并结果
        combined_instances = rel_tag + relevant_instances
        # 4. 清空已选实例集合，为下一次搜索做准备
        self.selected_instance_keys.clear()

        return combined_instances

    #用hash的方式去搜索
    def _search_by_sub_hash(self,
                            query,
                            top_k_class=10,
                            top_k_instances=15,
                            ):
        # 重置已选实例集合
        self.selected_instance_keys.clear()

        # # 1. 关键词覆盖查询，在第一步和第二步剩下的实例中进行
        # # 提取关键词
        # model = embedding_model
        # kw_model = KeyBERT(model=model)
        # keywords = kw_model.extract_keywords(
        #     query,
        #     keyphrase_ngram_range=(1, 2),  # 关键词长度范围，表示提取从单个单词到两个单词的关键词
        #     stop_words='english',
        #     top_n=3  # 返回前10个关键词
        # )
        # print(keywords)
        # query_keywords = keywords_process(keywords)
        # print(query_keywords)
        # kw_coverage_instances = self.find_kw_coverage_instances_with_tfidf(
        #     query_keywords,
        #     similarity_threshold=0.15,
        #     single_top_k=7,
        #     combo_top_k=7
        # )

        # 2. 在指定类中搜索实例, 排除已经找到的实例
        sensed_classes = self._sense_classes_by_tfidf(query, top_k_class, threshold=0.6, allow_below_threshold=True)
        len1, relevant_instances = self._fetch_instances_by_tfidf(query, top_k_instances, threshold=0.5,
                                                            classes=sensed_classes)
        _logger.info(f"第一阶段选择实例数量{len1}")

        # 3. 在全图中搜索实例，排除已找到的实例
        len2, relv_ins = self._fetch_instances_by_tfidf(query, top_k_instances, threshold=0.1)

        # 4. 合并结果
        # 确保都是列表
        combined_instances =relv_ins + relevant_instances
        #combined_instances = kw_coverage_instances + relv_ins + relevant_instances

        # 5. 清空已选实例集合，为下一次搜索做准备
        self.selected_instance_keys.clear()

        return combined_instances

    def _sense_classes_by_llm(self, query, llm, top_k_class):
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

        #_logger.info(f"QUERY_CLASSES_PROMPT: {query_classes_prompt}")
        response = llm.invoke(query_classes_prompt)
        _logger.info(f"QUERY_CLASSES_RESPONSE: {response.content}")

        try:
            query_results = json.loads(response.content)
            return query_results
        except json.JSONDecodeError as e:
            _logger.error(f"解析类查询结果失败: {e}")
            return {"error": "解析类查询结果失败", "selected_classes": []}

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

        _logger.info(f"开始构建类片段文本表示...")

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

        _logger.info(f"TF-IDF类感知: 总共收集到 {len(class_nodes)} 个类，{len(class_fragments)} 个片段")

        # 2. 使用TF-IDF进行向量化和相似度计算
        vectorizer_params = {
            'lowercase': True,
            'stop_words': 'english',
            'ngram_range': (1, 2),  # 包含1-gram和2-gram
            'min_df': 1,
            'max_df': 0.6,
            'use_idf': True,
            'smooth_idf': True
        }

        similarities, vectorizer, tfidf_matrix = calculate_tfidf_similarity(
            query,
            class_fragments,
            vectorizer_params=vectorizer_params
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

        # 4. 将类按最高相似度排序
        class_scores = list(class_max_scores.items())
        class_scores.sort(key=lambda x: x[1], reverse=True)

        _logger.info(f"类相似度统计: 共{len(class_scores)}个类有匹配片段")

        # 5. 按阈值过滤并选择类
        above_threshold = [(idx, score) for idx, score in class_scores if score >= threshold]
        below_threshold = [(idx, score) for idx, score in class_scores if score < threshold]

        _logger.info(
            f"TF-IDF类感知: 高于阈值({threshold})的类数量: {len(above_threshold)}, 低于阈值的类数量: {len(below_threshold)}")

        # 6. 选择策略
        if top_k_class <= 0:
            return {"selected_classes": []}

        if len(above_threshold) >= top_k_class:
            # 情况1: 高于阈值的类足够
            selected_class_scores = above_threshold[:top_k_class]
            _logger.info(f"选择前{top_k_class}个高于阈值的类")
        else:
            if allow_below_threshold:
                # 情况2: 高于阈值的类不足，用低于阈值但分数最高的补足
                above_threshold.sort(key=lambda x: x[1], reverse=True)
                below_threshold.sort(key=lambda x: x[1], reverse=True)

                remaining_slots = top_k_class - len(above_threshold)
                actual_slots = min(remaining_slots, len(below_threshold))
                selected_class_scores = above_threshold + below_threshold[:actual_slots]

                _logger.info(f"补充{actual_slots}个低于阈值但分数最高的类")
            else:
                # 情况3: 不允许使用低于阈值的类，只返回高于阈值的类
                selected_class_scores = above_threshold
                _logger.info(f"不允许使用低于阈值类，选择{len(selected_class_scores)}个高于阈值的类")

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
        _logger.info(f"TF-IDF最终选择的{len(selected_classes_info)}个类: {class_names}")
        _logger.info(f"对应TF-IDF相似度分数: {scores}")

        return {"selected_classes": selected_classes_info}



    def _fetch_instances_by_tfidf(self, query, top_k_instances, threshold, classes=None) :
        """
        基于TF-IDF向量化的实例检索方法

        Args:
            query: 查询字符串
            top_k_instances: 返回的实例数量上限
            threshold: 相似度阈值
            classes: 从_sense_classes_by_tfidf返回的相关类列表，如果为None则从所有类中收集

        Returns:
            str: 相关实例列表
        """
        # 1. 从相关类中收集所有实例及其片段
        all_instances = []  # 存储实例对象
        instance_documents = []  # 存储每个片段的文本表示
        fragment_instance_map = {}  # 存储片段索引到实例索引的映射
        instance_class_map = {}  # 存储实例索引到所属类的映射
        instance_keys_map = {}  # 存储实例索引到唯一标识的映射

        if classes is None:
            _logger.info(f"开始从所有类中收集实例片段...")
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
                       # _logger.info(f"实例的片段文本表示{fragments}")

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

            _logger.info(
                f"从 {class_count} 个类中收集到 {instance_count} 个实例，共 {fragment_count} 个片段，"
                f"跳过了 {len(self.selected_instance_keys)} 个已选中的实例")
        else:
            _logger.info(f"开始从指定类中收集实例片段...")

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
                _logger.info(f"当前类下面的实例{class_instances}")

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

                    _logger.info(f"用于去构建片段的实例{instance}")
                    # 为实例的每个片段构建文本表示
                    fragments = build_instance_fragments(instance)

                    _logger.info(f"当前实例的片段{fragments}")
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

                _logger.info(
                    f"从类 {class_id} 获取到 {len(class_instances)} 个实例，共 {len(instance_documents)} 个片段")

        if not all_instances or not instance_documents:
            _logger.warning("未找到任何实例或实例片段")
            return ""

        _logger.info(f"TF-IDF实例检索: 总共收集到 {len(all_instances)} 个实例，{len(instance_documents)} 个片段")

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
        selected_instance_scores = []
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
                _logger.info(f"选择实例 {instance_idx}: 最高相似度={score:.4f}, "
                             f"匹配片段类型={best_fragment.get('fragment_type', 'unknown')}, "
                             f"类={class_info.get('class_id', 'unknown')}")
            else:
                _logger.info(f"选择实例 {instance_idx}: 相似度={score:.4f}, "
                             f"类={class_info.get('class_id', 'unknown')}, "
                             f"匹配片段类型={best_fragment.get('fragment_type', 'unknown')}")

        return len(cleaned_instances), serialize_instance(cleaned_instances)

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
        print("*********************keybert结果*************************")
        print(keywords)
        # 将(keyword, score)元组列表转换为仅包含关键词字符串的列表
        keyword_strings = keywords_process(keywords)
        print(keyword_strings)
        # 返回格式化的结果
        return {
            "keywords": keyword_strings,  # 仅包含关键词字符串的列表
            "class_id": class_id,
            "instance_id": instance_id
        }

    def find_kw_relvinstance_tags(self, query) -> str:
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
        _logger.info(f"keywords_prompt: {find_keywords_prompt}")
        response = self._llm.invoke(find_keywords_prompt)
        print(f"keywords_prompt response: {response.content}")
        instances_data = json.loads(response.content)

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

        _logger.info(f"已将 {added_instance_count} 个实例添加到已选实例集合中")

        # 根据返回的class_id和instance_id查找完整实例数据
        relv_instances = []
        for instance_info in instances_data:
            class_id = instance_info.get('class_id')
            instance_id = instance_info.get('instance_id')

            # 在知识图谱中查找对应的实例
            found_instance = self._find_instance_by_ids(class_id, instance_id)
            if found_instance:
                relv_instances.append(found_instance)
            else:
                _logger.warning(f"Instance not found: class_id={class_id}, instance_id={instance_id}")

        _logger.info(f"Found {len(relv_instances)} relevant instances")
        # print(serialize_instance(relv_instances))
        return serialize_instance(relv_instances)

    def find_kw_coverage_instances_with_tfidf(self, query_keywords, similarity_threshold=0.1,
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
        _logger.info(f"成功加载 {len(tags_data)} 个tags数据")

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

        _logger.info(f"查询关键词: {query_keywords}")
        _logger.info(f"相似度阈值: {similarity_threshold}")
        _logger.info(f"单个实例返回数量: {single_top_k}")
        _logger.info(f"组合返回数量: {combo_top_k}")

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

        _logger.info(f"总计 {len(all_keywords_set)} 个唯一关键词，排除 {excluded_count} 个已选实例")

        if not all_keywords_set:
            _logger.warning("没有找到任何关键词")
            return serialize_instance([])

        # 4. 计算TF-IDF相似度矩阵
        # 准备文档：每个关键词作为一个文档
        keyword_docs = list(all_keywords_set)
        query_docs = query_keywords

        # 使用自定义参数，注意原代码中max_df=0.7
        vectorizer_params = {
            'lowercase': True,
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'min_df': 1,
            'max_df': 0.7,  # 原代码中max_df=0.7
            'use_idf': True,
            'smooth_idf': True
        }

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
        _logger.info(f"TF-IDF向量维度: {tfidf_matrix.shape}")
        _logger.info(f"相似度矩阵形状: {similarity_matrix.shape}")

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

            _logger.info(f"查询关键词 '{query_keyword}' 找到 {len(similar_keywords)} 个相似关键词")
            if similar_keywords:
                top_similar = similar_keywords[:3]
                similarity_str = ', '.join([f"{item['keyword']}:{item['similarity']:.3f}" for item in top_similar])
                _logger.info(f"  Top 3: {similarity_str}")

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
            _logger.info(f"查询关键词 '{query_keyword}': 找到 {len(instances_for_query)} 个相关实例")

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

        _logger.info(f"构建完成: {len(instance_to_queries)} 个实例至少匹配到一个查询关键词")

        # 10. 跳过第一阶段，直接进入第二阶段
        _logger.info("跳过第一阶段，直接进入第二阶段：贪心算法寻找最小覆盖组合")

        # 11. 第二阶段：贪心算法寻找覆盖所有查询关键词的实例组合
        all_combinations = []  # 存储所有找到的组合
        stage1_instance_keys_set = set()  # 第一阶段为空，因为我们跳过了第一阶段

        # 为了找到多个组合，我们需要多次运行贪心算法
        for combo_idx in range(combo_top_k):
            _logger.info(f"开始寻找第 {combo_idx + 1} 个组合...")

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
                _logger.info(f"第{iteration}轮贪心选择: 选择实例 {best_instance}, 覆盖 {best_coverage} 个新查询关键词")

                # 记录覆盖详情
                coverage_details = []
                for query in best_queries:
                    for detail in instance_to_queries_details[best_instance]:
                        if detail['query_keyword'] == query:
                            coverage_details.append(f"{query}->{detail['matched_keyword']}:{detail['similarity']:.3f}")
                            break

                if coverage_details:
                    _logger.info(f"  覆盖详情: {coverage_details}")

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
                _logger.info(f"第 {combo_idx + 1} 个组合已完全覆盖所有查询关键词")
            elif not current_instance_coverage_map:
                _logger.warning(f"第 {combo_idx + 1} 个组合: 没有更多可选的实例来覆盖剩余查询关键词")

            _logger.info(
                f"第 {combo_idx + 1} 个组合结果: 选择 {len(current_selected_instances)} 个实例, 覆盖了 {len(current_covered_queries)}/{len(query_keywords)} 个查询关键词")

            if current_remaining_queries:
                _logger.warning(f"第 {combo_idx + 1} 个组合剩余未覆盖的查询关键词: {list(current_remaining_queries)}")

            # 如果没有找到足够的实例，继续下一个组合
            if not current_selected_instances:
                _logger.info(f"第 {combo_idx + 1} 个组合为空，停止寻找更多组合")
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

        _logger.info(f"第二阶段: 找到 {len(all_combinations)} 个有效组合")

        # 12. 获取所有组合的完整实例数据
        all_instances = []
        instance_processed = set()  # 记录已处理的实例，避免重复

        for combo_idx, combo in enumerate(all_combinations):
            _logger.info(f"处理第 {combo_idx + 1} 个组合，包含 {len(combo['instances'])} 个实例")

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

                    _logger.info(
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
                _logger.info(f"尝试为剩余 {len(remaining_queries)} 个查询关键词寻找额外实例...")

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
                        _logger.info(f"添加额外实例 {instance_key}, 最高相似度: {similarity:.4f}")

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

        _logger.info(f"已将 {len(added_instance_keys)} 个实例添加到已选实例集合中")

        # 15. 按阶段、组合索引、覆盖数量和相似度排序
        all_instances.sort(key=lambda x: (
            x.get('stage', 2),  # 第一阶段在前
            x.get('combo_index', 0),  # 按组合排序
            -x.get('coverage_count', 0),  # 覆盖数量降序
            -x.get('avg_similarity', 0)  # 相似度降序
        ))

        _logger.info(f"最终返回 {len(all_instances)} 个相关实例")

        return serialize_instance(all_instances)

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
