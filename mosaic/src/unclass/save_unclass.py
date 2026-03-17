from glob import glob

from src.unclass.graph_unclass import InstanceGraph
from src.assist import *
from src.logger import setup_logger
import json
import argparse
_logger = setup_logger("save")


def _process_data_truncation(memory: InstanceGraph,
                             data: list,
                             context: list
                             ) -> InstanceGraph:
    #1.
    # 此方法会返回两部分：需要更新的现有实例 和 需要创建的新实例
    instances_to_update, new_instances = memory.sense_instances(
        data=data,
        context=context,
        tfidf_threshold=0.9,  # TF-IDF相似度阈值，高于此值则认为匹配到现有实例
        top_k=None  # 可选，每个信息片段最多匹配的实例数，None 表示匹配所有超过阈值的
    )

    _logger.info(f"relv_msg: {instances_to_update}; unknown_msg: {new_instances}")

    #2 处理感知结果（更新旧实例，添加新实例到图）
    memory.process_instances(instances_to_update, new_instances)

    # 3. (可选) 基于信息标签更新实例间的关系
    # 此步骤会遍历 data，将拥有相同 label 的信息所关联的实例连接起来
    memory.update_instance_relationships(data)

#     # # step 5: 图的冲突检测和消息传递
#     # memory.message_passing()

    return memory


def save(data,conv_name):
    # 调用函数
    result = conv_message_splitter(data)

    memory = InstanceGraph()
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
        "D:/model/conv/GraphConv/oop_graph/src/locomo results/conv/locomo_conv0.json",
        "D:/model/conv/GraphConv/oop_graph/src/locomo results/conv/locomo_conv1.json",
        "D:/model/conv/GraphConv/oop_graph/src/locomo results/conv/locomo_conv2.json",
        "D:/model/conv/GraphConv/oop_graph/src/locomo results/conv/locomo_conv3.json",
        "D:/model/conv/GraphConv/oop_graph/src/locomo results/conv/locomo_conv4.json",
        "D:/model/conv/GraphConv/oop_graph/src/locomo results/conv/locomo_conv5.json",
        "D:/model/conv/GraphConv/oop_graph/src/locomo results/conv/locomo_conv6.json",
        "D:/model/conv/GraphConv/oop_graph/src/locomo results/conv/locomo_conv7.json",
        "D:/model/conv/GraphConv/oop_graph/src/locomo results/conv/locomo_conv8.json",
        "D:/model/conv/GraphConv/oop_graph/src/locomo results/conv/locomo_conv9.json",
    ]

    # # 获取所有匹配的文件
    # conv_files = glob(file_pattern)

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
