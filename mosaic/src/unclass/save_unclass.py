import json
import os
from glob import glob

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from src.unclass.graph_unclass import InstanceGraph
from src.assist import conv_message_splitter, read_to_file_json
from src.logger import setup_logger
_logger = setup_logger("save")


def _progress_bar(iterable, total: int, desc: str):
    if tqdm is None or total <= 0:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="batch", smoothing=0.08, mininterval=0.3)


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

    _logger.debug("instances_to_update: %s; new_instances: %s", instances_to_update, new_instances)

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

    total = len(result)
    pbar = _progress_bar(result, total, "构图(unclass)")
    for i, (batch, context) in enumerate(pbar):
        _logger.debug("构图进度: [%d/%d] 处理本组 %d 条消息", i + 1, total, len(batch))
        _logger.debug("当前消息: %s; 上文: %s", batch, context[:3] if context else [])
        memory = _process_data_truncation(memory, batch, context)
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(n=len(memory.graph.nodes))
    _logger.info("构图完成")
    return memory


def process_single_conv(file_path):
    """处理单个conv文件"""
    _logger.info("处理文件: %s", file_path)

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
    base_dir = os.environ.get("MOSAIC_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", ".."))
    conv_dir = os.path.join(base_dir, "locomo results", "conv")
    file_pattern = os.path.join(conv_dir, "locomo_conv*.json")
    conv_files = sorted(glob(file_pattern))

    # # 获取所有匹配的文件
    # conv_files = glob(file_pattern)

    if not conv_files:
        _logger.info("未找到匹配的文件: %s", file_pattern)
        return

    _logger.info("找到 %d 个 conv 文件，开始构图", len(conv_files))

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
