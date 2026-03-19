import json
import os

from src.data.graph import ClassGraph
from src.assist import (
    conv_message_splitter,
    message_splitter,
    read_to_file_json,
)
from src.logger import setup_logger

_logger = setup_logger("save")


def _process_data_truncation(memory: ClassGraph,
                             data: list,
                             context: list
                             ) -> ClassGraph:
    relevant_class_messages, new_class_messages = memory.sense_classes(data, context)
    _logger.info("relevant_class_messages: %s; new_class_messages: %s", relevant_class_messages, new_class_messages)

    # 可选: memory.consistency_valid_dynamic(relevant_class_messages, new_class_messages)

    processed_classes = memory.process_relevant_class_instances(relevant_class_messages)
    _logger.info("processed_classes: %s", processed_classes)

    added_class_nodes = memory.add_classnodes(new_class_messages)
    _logger.info("added_class_nodes: %s", added_class_nodes)

    # #4.处理类内的边关系和类间的边关系
    memory.update_class_relationships(data, processed_classes, added_class_nodes)

#     # # step 5: 图的冲突检测和消息传递
#     # memory.message_passing()

    return memory


def _process_data_truncation_hash(memory: ClassGraph, data: list, context: list) -> ClassGraph:
    """仅用 TF-IDF/hash 构图，不调用 LLM（未匹配片段归入 Unclassified，实例用文本拼接）。"""
    relevant_class_messages, new_class_messages = memory.sense_classes(
        data, context, use_llm_for_new=False
    )
    _logger.info("relevant_class_messages: %s; new_class_messages: %s", relevant_class_messages, new_class_messages)

    processed_classes = memory.process_relevant_class_instances(
        relevant_class_messages, use_hash=True
    )
    _logger.info("processed_classes: %s", processed_classes)

    added_class_nodes = memory.add_classnodes(new_class_messages, use_hash=True)
    _logger.info("added_class_nodes: %s", added_class_nodes)

    memory.update_class_relationships(data, processed_classes, added_class_nodes)
    return memory


#error compounding
def save_error(data):
    # 初始化memory
    memory = ClassGraph()

    # 调用message_splitter函数
    result = message_splitter(data)

    message_labels = []
    for i, (batch, context) in enumerate(result):
        message_labels.extend(batch)

    # 需要提前建立好message_labes数组，因为要记录下error涉及到哪些信息label,去查找具体的信息
    memory.message_labels = message_labels

    for i, (batch, context) in enumerate(result):
        if i > 80:
            print(f"\n分组 {i + 1}:")
            print("当前消息:", batch)
            print("上文前三条:", context)
            memory = _process_data_truncation(memory, batch, context)
    return memory



def save(data, conv_name):
    # 调用函数
    result = conv_message_splitter(data)

    memory = ClassGraph()
    memory.filepath = conv_name

    for i, (batch, context) in enumerate(result):
    #if i > 17:
        print(f"\n分组 {i + 1}:")
        print("当前消息:", batch)
        print("上文前三条:", context)
        memory = _process_data_truncation(memory, batch, context)
    return memory


def save_hash(data, conv_name, graph_save_dir=None, final_graph_path=None, final_tags_path=None):
    """
    仅用 TF-IDF/hash 构图，不调用 LLM。适合无 API 或基线实验。
    返回构建好的 ClassGraph。
    若提供 final_graph_path / final_tags_path，则最后将图 pickle 到该路径，并用 TF-IDF 生成 tags 写入 final_tags_path。
    """
    result = conv_message_splitter(data)
    memory = ClassGraph()
    memory.filepath = conv_name
    if graph_save_dir is not None:
        memory._graph_save_dir = graph_save_dir
    for i, (batch, context) in enumerate(result):
        _logger.info("分组 %s: 当前消息数 %s", i + 1, len(batch))
        memory = _process_data_truncation_hash(memory, batch, context)
    if final_graph_path:
        import pickle
        os.makedirs(os.path.dirname(os.path.abspath(final_graph_path)) or ".", exist_ok=True)
        with open(final_graph_path, "wb") as f:
            pickle.dump(memory.graph, f)
        _logger.info("图已保存到 %s", final_graph_path)
    if final_tags_path and hasattr(memory, "generate_tags_tfidf"):
        memory.generate_tags_tfidf(final_tags_path)
    return memory


def process_single_conv(file_path):
    """处理单个conv文件"""
    print(f"处理文件: {file_path}")

    # 提取conv名称（例如从"locomo_conv9.json"中提取"conv9"）
    file_name = os.path.basename(file_path)
    conv_name = file_name.replace("locomo_", "").replace(".json", "")

    # 加载数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 保存处理结果
    return save(data, conv_name)


def process_all_convs():
    """处理目录下所有符合条件的conv文件"""
    # 定义要处理的文件模式
    file_pattern = "./locomo results/conv/locomo_conv*.json"

    conv_files = [
        "./locomo results/conv/locomo_conv1.json",
        "./locomo results/conv/locomo_conv2.json",
        "./locomo results/conv/locomo_conv5.json",
        "./locomo results/conv/locomo_conv6.json",
        "./locomo results/conv/locomo_conv8.json"
    ]



    if not conv_files:
        print(f"未找到匹配的文件: {file_pattern}")
        return

    print(f"找到 {len(conv_files)} 个conv文件: {conv_files}")

    # 处理每个文件
    results = {}
    for file_path in conv_files:

        memory = process_single_conv(file_path)
        file_name = os.path.basename(file_path)
        results[file_name] = memory

    return results
if __name__ == '__main__':
    #对话准确率数据的测试
    process_all_convs()
#
#     # #高血压数据用于去做error_case
#     # file_name = "D:/model/conv/GraphConv/oop_graph/src/error_case/chunks.json"
#     # with open(file_name, 'r', encoding='utf-8') as f:
#     #     data = json.load(f)
#     # save_error(data)
