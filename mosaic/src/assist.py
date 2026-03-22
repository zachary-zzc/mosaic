from typing import Any, Dict, List, Sequence, Union
import os
import re
import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from string import Template

from src.llm.llm import load_chat_model
from src.logger import setup_logger
from src.config_loader import get_embedding_model_path
from src.utils.constants import DEFAULT_TFIDF_VECTORIZER_PARAMS
from src.utils import io_utils as _io
from src.utils.io_utils import parse_llm_json_object

_logger = setup_logger("graph assist")



#用这样的方式去加载调用的api
def fetch_default_llm_model():
    return load_chat_model("ali_api|qwen3.5-plus")
    #return load_chat_model("ali_api|deepseek-r1")

def build_class_fragments(class_node):
    """
    构建类的片段文本表示用于TF-IDF计算
    返回: 包含(片段类型, 片段文本)的列表
    """
    fragments = []

    # 1. 类名作为一个片段
    class_name = getattr(class_node, 'class_name', '')
    if class_name:
        fragments.append(("class_name", class_name))

    # 2. 每个属性作为一个独立片段
    attributes = getattr(class_node, 'attributes', [])
    if isinstance(attributes, list):
        for attr in attributes:
            if isinstance(attr, str) and attr.strip():
                fragments.append(("attribute", attr))
    elif isinstance(attributes, set):
        for attr in attributes:
            if isinstance(attr, str) and attr.strip():
                fragments.append(("attribute", str(attr)))
    elif isinstance(attributes, dict):
        for attr_name, attr_value in attributes.items():
            fragment_text = f"{attr_name}"
            if isinstance(attr_value, str):
                fragment_text += f" {attr_value}"
            fragments.append(("attribute", fragment_text))

    # 3. 每个操作作为一个独立片段
    operations = getattr(class_node, 'operations', [])
    if isinstance(operations, list):
        for op in operations:
            if isinstance(op, str) and op.strip():
                fragments.append(("operation", op))
    elif isinstance(operations, set):
        for op in operations:
            if isinstance(op, str) and op.strip():
                fragments.append(("operation", str(op)))
    elif isinstance(operations, dict):
        for op_name, op_value in operations.items():
            fragment_text = f"{op_name}"
            if isinstance(op_value, str):
                fragment_text += f" {op_value}"
            fragments.append(("operation", fragment_text))

    # 4. 每个未分类字段作为一个独立片段
    unclassified = getattr(class_node, 'unclassified', [])
    if isinstance(unclassified, list):
        for item in unclassified:
            if isinstance(item, str) and item.strip():
                fragments.append(("unclassified", item))
    elif isinstance(unclassified, set):
        for item in unclassified:
            if isinstance(item, str) and item.strip():
                fragments.append(("unclassified", str(item)))
    elif isinstance(unclassified, dict):
        for item_key, item_value in unclassified.items():
            fragment_text = f"{item_key}"
            if isinstance(item_value, str):
                fragment_text += f" {item_value}"
            fragments.append(("unclassified", fragment_text))

    return fragments

def build_instance_fragments(instance):
    """
    构建实例的多个片段文本表示
    返回: 包含(片段类型, 片段文本)的列表
    """
    fragments = []

    # 1. 实例名称作为一个片段
    instance_name = instance.get("instance_name", "")
    if instance_name:
        fragments.append(("instance_name", instance_name))

    # 2. 每个属性作为一个独立片段
    if "attributes" in instance and isinstance(instance["attributes"], dict):
        for attr_name, attr_content in instance["attributes"].items():
            attr_value = str(attr_content.get("value", ""))
            attr_description = attr_content.get("description", "")

            # 构建属性片段文本
            if attr_value or attr_description:
                fragment_text = f"{attr_name}"
                if attr_value:
                    fragment_text += f" {attr_value}"
                if attr_description:
                    fragment_text += f" {attr_description}"

                fragments.append(("attribute", fragment_text))

    # 3. 每个操作函数作为一个独立片段
    if "operations" in instance and isinstance(instance["operations"], dict):
        for op_name, op_content in instance["operations"].items():
            op_description = op_content.get("description", "")

            # 构建操作片段文本
            fragment_text = f"{op_name}"
            if op_description:
                fragment_text += f" {op_description}"

            fragments.append(("operation", fragment_text))

    # 4. 未分类字段作为一个独立片段
    uninstance_field = instance.get("uninstance_field", "")
    if uninstance_field:
        fragments.append(("unclassified", uninstance_field))

    return fragments
#序列化
def serialize_query(data):
    """
    序列化 ClassNode 列表为简洁的格式化字符串
    """
    result_lines=[]
    for node in data:

        class_id = node.class_id
        class_name = node.class_name
        attributes = node.attributes
        operations = node.operations
        unclassified = node.unclassified
        instances = node._instances

        instance_names = []

        # 安全地从每个实例中提取名称
        for inst in instances:
            if isinstance(inst, dict) and 'instance_name' in inst:
                instance_names.append(inst['instance_name'])
            elif hasattr(inst, 'instance_name'):
                instance_names.append(getattr(inst, 'instance_name', 'Unknown_Instance'))

        node_info = f"""{class_name} ({class_id}):
                     Attributes: {", ".join(attributes) if attributes else "No attributes"}
                     Operations: {", ".join(operations) if operations else "No operations"}
                     Unclassified: {", ".join(unclassified) if unclassified else "No unclassified"}
                     Instance names: {", ".join(instance_names) if instance_names else "No instances"}
                     """

        result_lines.append(node_info)

    return "\n".join(result_lines)


def serialize(data) -> List[Dict[str, Any]]:
    """
    序列化 ClassNode 列表为 字符串

    """
    serialized_list = []
    for node in data:
        # 只序列化关键属性（避免包含实例列表）
        serialized_node = {
            "class_id": node.class_id,
            "class_name": node.class_name
        }
        serialized_list.append(serialized_node)
    return serialized_list


def serialize_instance_kw(instance):
    """
    序列化单个实例数据为详细格式化的字符串，展示所有层级信息（包括属性、操作的描述和值），用于关键词生成

    Args:
        instance: 单个实例数据，为字典形式

    Returns:
        str: 格式化的详细字符串
    """
    result_lines = []

    if not instance or not isinstance(instance, dict):
        return "Invalid instance data."

    # 提取实例基本信息
    instance_id = instance.get('instance_id', 'unknown')
    instance_name = instance.get('instance_name', 'Unknown_Instance')
    uninstance_field = instance.get('uninstance_field', '')
    functions = instance.get('functions', [])

    # 开始构建实例的详细信息
    result_lines.append(f"{instance_name}")

    # 处理属性信息
    attributes = instance.get('attributes', {})
    if attributes and isinstance(attributes, dict):

        for attr_name, attr_content in attributes.items():
            if isinstance(attr_content, dict):
                description = attr_content.get('description', 'no description')
                value = attr_content.get('value', 'no value')
                occurred = attr_content.get('occurred', 'none')
                recorded_at = attr_content.get('recorded_at', 'none')
                result_lines.append(f" {description}; {value}; {occurred};")

    # 处理操作信息
    operations = instance.get('operations', {})
    if operations and isinstance(operations, dict):

        for op_name, op_content in operations.items():
            if isinstance(op_content, dict):
                description = op_content.get('description', 'no description')
                result_lines.append(f"{op_name};{description}")

    # 处理未分类字段
    if uninstance_field:
        result_lines.append(f"{uninstance_field}")

    # # 处理函数信息（如果需要）
    # if functions and isinstance(functions, list):
    #     result_lines.append(f"Functions: {len(functions)} function(s) associated")

    return "\n".join(result_lines)
def serialize_instance(data):

    """
    序列化实例数据为详细格式化的字符串，展示所有层级信息（包括属性、操作的描述和值）,用于查询

    Args:
        data: 实例数据列表，每个实例为字典形式

    Returns:
        str: 格式化的详细字符串
    """
    result_lines = []

    if not data:
        return ""
    for i, instance in enumerate(data, 1):
        # 提取实例基本信息
        instance_name = instance.get('instance_name', f'Unknown_Instance_{i}')
        uninstance_field = instance.get('uninstance_field', '')
        functions = instance.get('functions', [])

        # 开始构建单个实例的详细信息
        result_lines.append(f"\n─── instance {i}: {instance_name} ───")
        # 处理属性信息
        attributes = instance.get('attributes', {})
        if attributes:
            for attr_name, attr_content in attributes.items():
                description = attr_content.get('description', 'no description')
                value = attr_content.get('value', 'no value')
                occurred = attr_content.get('occurred', 'none')
                recorded_at = attr_content.get('recorded_at', 'none')

                result_lines.append(f"Description: {description};Value: {value};Occurred Time: {occurred};Recorded Time: {recorded_at}")
        # 处理操作信息
        operations = instance.get('operations', {})
        if operations:
            for op_name, op_content in operations.items():
                description = op_content.get('description', 'no description')
                result_lines.append(f"Operation Name: {op_name};Operation Description: {description}")
        # 处理未分类字段
        if uninstance_field:
            result_lines.append(f"Uninstance_field: {uninstance_field}")

        #处理和其他实例的连边字段
        if functions:
            result_lines.append(f"functions: {functions}")
    return "\n".join(result_lines)


#查询问题
def query_question(
    llm,
    question: str,
    information: Union[str, Sequence[str]],
    prompt_template: str,
) -> str:
    #用来确认这些条件为真，否则触发异常
    assert("{QUESTION}" in prompt_template and "{INFORMATION}" in prompt_template), \
        "Prompt template must contain {QUESTION} and {INFORMATION} placeholders."
    if isinstance(information, (list, tuple)):
        info_str = "\n\n".join(str(x) for x in information if str(x).strip())
    else:
        info_str = str(information or "")
    prompt = Template(prompt_template).substitute(
        {"QUESTION": question, "INFORMATION": info_str}
    )
    _logger.debug("query prompt: %s", prompt)
    response = llm.invoke(prompt)
    content = getattr(response, "content", None) or str(response)
    parsed = parse_llm_json_object(content)
    if parsed is not None:
        ans = parsed.get("response")
        if ans is not None:
            return str(ans).strip()
    _logger.warning(
        "query_question: 无法从 LLM 回复解析 JSON 对象或缺少 response 字段，返回空串。原文前 500 字: %r",
        content[:500],
    )
    return ""



def conv_message_splitter(data):
    """
    将消息列表按10条分组，每组返回 (当前10条消息, 前三条上文)

    参数:
        messages: 已经用restructured_message处理好的字符串列表

    返回:
        list of tuples: (current_batch, context_history)
        - current_batch: 当前10条消息列表
        - context_history: 上文前三条消息（用于下一批次）
    """
    # 找出所有会话键（如session_1, session_2等）
    session_keys = [key for key in data.keys() if re.match(r'session_\d+', key)]
    # 过滤掉不需要的 '_date_time' 键
    session_keys = [key for key in session_keys if not key.endswith('_date_time')]
    # 按数字排序会话键，确保按顺序处理
    session_keys.sort(key=lambda x: int(x.split('_')[1]))

    blocks = []  # 存储最终分组结果
    global_message_count = 1


    #session_keys= session_keys[:1]
    for session_key in session_keys:
        #print(f"===================== 开始处理会话 {session_key} =====================")
        # 获取当前会话的时间戳（尝试匹配 session_X_date_time 格式）
        time_key = f"{session_key}_date_time"
        dialogue_time = data.get(time_key, "未知时间")
        # 获取当前会话的消息列表
        messages = data.get(session_key, [])
        #print(messages)  # 会话原始消息列表

        context_history = []  # 存储上文前三条
        batch = []  # 临时存储当前批次的10条消息
        for index, message in enumerate(messages):
            # print(f"************************* 正在处理 {session_key} 的信息 {index + 1} ********************************")
            # 构建当前消息的上下文（保持原逻辑不变）
            if "img_url" in message and message["img_url"]:
                restructured_message = f"""【{message['speaker']} sent an image: {message.get('query', 'No description')}. Image content: {message.get('blip_caption', 'No description')}. Dialogue: {message['text']} Dialogue time:{dialogue_time}】"""
            else:
                restructured_message = f"""【{message['speaker']} said: "{message['text']} Dialogue time:{dialogue_time}】"""
            #print(restructured_message)
            # 创建消息字典，标签与消息内容分离
            message_dict = {
                "message": restructured_message,
                "label": global_message_count  # 全局唯一标签
            }

            # 添加到当前批次
            batch.append(message_dict)
            global_message_count += 1  # 全局计数器递增

            ##之前的版本是10条，-3
            # 每10条触发分组
            if len(batch) == 10:
                # 保存当前批次和当前上下文（注意：这是上一批的最后3条）
                blocks.append((batch[:], context_history[:]))  # 复制避免引用问题
                # 更新上下文：当前批次的最后3条
                context_history = batch[-3:]
                batch = []  # 重置批次
        # 处理剩余不足10条的消息
        if batch:
            blocks.append((batch[:], context_history[:]))

    return blocks


def message_splitter(data):
    """
    将文本分割成块，然后按句子分割，为每个句子分配唯一标签

    Args:
        content: 输入的文本内容

    Returns:
        包含消息和标签的字典列表，格式为 [{"message": 句子内容, "label": 标签编号}]
    """
    # 处理结果列表
    all_sentences = []
    global_message_count = 1
    blocks = []

    #data=data[:1]

    for item in data:
        # 取出当前元素的processed_text字段
        processed_text = item["processed_text"]
        sentences = re.split(r'[。]', processed_text)
        for sentence in sentences:
            # 清理句子：去除空白字符
            cleaned_sentence = sentence.strip()

            # 跳过空句子
            if cleaned_sentence:
                # 创建消息字典，格式与 conv_message_splitter 保持一致
                message_dict = {
                    "message": cleaned_sentence,
                    "label": global_message_count
                }
                all_sentences.append(message_dict)
                global_message_count += 1

    # 按每5个句子进行分组，并为每个分组提供上下文
    batch_size = 5
    previous_context = None  # 上一个分组的最后一句话

    for i in range(0, len(all_sentences), batch_size):
        # 获取当前批次（5句话）
        current_batch = all_sentences[i:i + batch_size]

        # 只有当前批次有内容时才处理
        if current_batch:
            # 添加上下文信息：上一个分组的最后一句话[8](@ref)
            context_info = previous_context if previous_context else []

            # 将当前批次和上下文信息添加到结果中
            blocks.append((current_batch, context_info))

            # 更新上下文：当前批次的最后一句话，用于下一个分组
            previous_context = [current_batch[-1]] if len(current_batch) > 0 else []

    return blocks

def format_messages_for_prompt(messages: List[Dict]) -> str:
    """格式化消息列表，用于插入到提示词中。"""
    formatted = []
    for item in messages:
        if "label" in item and "message" in item:
            formatted.append(f"标签{item['label']}: {item['message']}")
    return "\n".join(formatted)


def calculate_tfidf_similarity(query, documents, vectorizer_params=None):
    """
    使用TF-IDF计算查询与文档列表的相似度

    Args:
        query: 查询字符串
        documents: 文档列表（list of strings）
        vectorizer_params: TF-IDF向量化器的参数字典，默认为None使用预设参数

    Returns:
        similarities: 相似度数组，每个元素是查询与对应文档的相似度
        vectorizer: 训练好的TF-IDF向量化器
        tfidf_matrix: 文档的TF-IDF矩阵
    """
    if vectorizer_params is None:
        vectorizer_params = dict(DEFAULT_TFIDF_VECTORIZER_PARAMS)
    vectorizer = TfidfVectorizer(**vectorizer_params)

    # 拟合和转换文档文本
    tfidf_matrix = vectorizer.fit_transform(documents)

    # 转换查询文本
    query_vector = vectorizer.transform([query])

    # 计算余弦相似度
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    return similarities, vectorizer, tfidf_matrix

def calculate_cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    if vec1 is None or vec2 is None:
        return 0.0

    # 使用点积和模长计算余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)

_embedding_model = None


def _get_embedding_model():
    """懒加载本地 SentenceTransformer，路径来自 config [PATHS] 或环境变量（绝对路径）。"""
    global _embedding_model
    if _embedding_model is None:
        path = get_embedding_model_path()
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"嵌入模型目录不存在: {path}\n"
                "请在 mosaic/config/config.cfg 的 [PATHS] embedding_model 中配置绝对路径，"
                "或设置环境变量 MOSAIC_EMBEDDING_MODEL，或将模型放到该目录。"
            )
        _embedding_model = SentenceTransformer(path)
        _logger.debug("嵌入模型已加载: %s", path)
    return _embedding_model


def similarity_score(text1: str, text2: str) -> float:
    """
    计算两个文本之间的余弦相似度

    Args:
        text1: 第一个文本字符串
        text2: 第二个文本字符串

    Returns:
        float: 两个文本的余弦相似度得分，范围[0, 1]
    """
    model = _get_embedding_model()
    text1_vec = model.encode(text1)
    text2_vec = model.encode(text2)

    # 从相似度矩阵中提取标量值
    cosine_score = calculate_cosine_similarity(text1_vec,text2_vec)
    return cosine_score

def save_to_file_json(file_name: str, struct: Any) -> None:
    """Backward-compatible: write JSON to file."""
    _io.write_json(file_name, struct, indent=4)


def read_to_file_json(file_name: str) -> Any:
    """Backward-compatible: read JSON from file."""
    return _io.read_json(file_name)


def save_to_file(filepath: str, nodes_data: Any) -> None:
    """Backward-compatible: write pickle."""
    _io.write_pickle(filepath, nodes_data)


def load_graphs(filepath: str) -> Any:
    """Backward-compatible: load graph from pickle."""
    return _io.read_pickle(filepath)

def keywords_process(keywords: List) -> List[str]:
    """将 (keyword, score) 元组或字符串列表转为关键词字符串列表。"""
    result = []
    for item in keywords:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            keyword = item[0]
            if keyword:
                keyword = str(keyword).strip().replace("_", " ")
                result.append(keyword)
        elif isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result



