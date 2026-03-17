from glob import glob

from src.data.graph import ClassGraph
from assist import *
from logger import setup_logger
import json
import argparse
_logger = setup_logger("save")


def _process_data_truncation(memory: ClassGraph,
                             data: list,
                             context: list
                             ) -> ClassGraph:
    #1.对于新的信息，感知信息的分类
    relv_msg, unknown_msg= memory.sense_classes(data, context)
    _logger.info(f"relv_msg: {relv_msg}; unknown_msg: {unknown_msg}")

    # # 1.1.当前信息的冲突检测，可选，需要检测冲突的时候就用这个
    # memory.consistency_vaild_dynamic(relv_msg,unknown_msg)


    # 2.处理相关的类
    processed_classes = memory.process_relvclass_instances(relv_msg)
    _logger.info(f"processed_classes: {processed_classes}")


    # 3.处理未知的类
    new_classes = memory.add_classnodes(unknown_msg)
    _logger.info(f"new_classes: {new_classes}")

    # #4.处理类内的边关系和类间的边关系
    memory.update_class_relationships(data, processed_classes, new_classes)

#     # # step 5: 图的冲突检测和消息传递
#     # memory.message_passing()

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



def save(data,conv_name):
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

import os
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
