import json
import os

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from src.data.graph import ClassGraph
from src.assist import (
    conv_message_splitter,
    message_splitter,
    read_to_file_json,
)
from src.logger import setup_logger

_logger = setup_logger("save")


def _progress_bar(iterable, total: int, desc: str):
    """构图主 stdout：批次进度条（与主程序按批存储一致）。"""
    if tqdm is None or total <= 0:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="batch",
        smoothing=0.08,
        mininterval=0.3,
    )


def _twrite(msg: str) -> None:
    """在 tqdm 运行时安全输出一行摘要（不破坏进度条）。"""
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


def _conversation_message_totals(result: list) -> tuple[int, list[int]]:
    """返回 (全部分组内对话消息总条数, 每批条数列表)。"""
    sizes = [len(batch) for batch, _ in result]
    return sum(sizes), sizes


def _write_construction_progress(
    current_1based: int,
    total_batches: int,
    *,
    messages_done: int | None = None,
    total_messages: int | None = None,
) -> None:
    """
    若设置环境变量 MOSAIC_PROGRESS_FILE，写入批次与（可选）消息级进度。
    messages_done: 已完成处理的对话消息条数（累计）
    """
    path = os.environ.get("MOSAIC_PROGRESS_FILE")
    if not path or total_batches <= 0:
        return
    try:
        pct_b = 100.0 * current_1based / total_batches
        lines = [f"batches {current_1based}/{total_batches} ({pct_b:.1f}%)"]
        if total_messages is not None and total_messages > 0 and messages_done is not None:
            pct_m = 100.0 * messages_done / total_messages
            rem = total_messages - messages_done
            lines.append(f"messages {messages_done}/{total_messages} ({pct_m:.1f}%) remaining {rem}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


def _process_data_truncation(memory: ClassGraph,
                             data: list,
                             context: list
                             ) -> ClassGraph:
    relevant_class_messages, new_class_messages = memory.sense_classes(data, context)
    _logger.debug("relevant_class_messages: %s; new_class_messages: %s", relevant_class_messages, new_class_messages)

    processed_classes = memory.process_relevant_class_instances(relevant_class_messages)
    _logger.debug("processed_classes: %s", processed_classes)

    added_class_nodes = memory.add_classnodes(new_class_messages)
    _logger.debug("added_class_nodes: %s", added_class_nodes)

    memory.update_class_relationships(data, processed_classes, added_class_nodes)

#     # # step 5: 图的冲突检测和消息传递
#     # memory.message_passing()

    return memory


def _process_data_truncation_hash(memory: ClassGraph, data: list, context: list) -> ClassGraph:
    """仅用 TF-IDF/hash 构图，不调用 LLM（未匹配片段归入 Unclassified，实例用文本拼接）。"""
    relevant_class_messages, new_class_messages = memory.sense_classes(
        data, context, use_llm_for_new=False
    )
    _logger.debug("relevant_class_messages: %s; new_class_messages: %s", relevant_class_messages, new_class_messages)

    processed_classes = memory.process_relevant_class_instances(
        relevant_class_messages, use_hash=True
    )
    _logger.debug("processed_classes: %s", processed_classes)

    added_class_nodes = memory.add_classnodes(new_class_messages, use_hash=True)
    _logger.debug("added_class_nodes: %s", added_class_nodes)

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

    total = len(result)
    total_msgs, _ = _conversation_message_totals(result)
    done_before = 0
    pbar = _progress_bar(result, total, "构图(error)")
    for i, (batch, context) in enumerate(pbar):
        if i > 80:
            n = len(batch)
            k, pct_b = i + 1, 100.0 * (i + 1) / total
            done_after = done_before + n
            pct_m = 100.0 * done_after / total_msgs if total_msgs else 0.0
            rem = total_msgs - done_after
            _logger.debug(
                "构图进度: 批 [%d/%d] %.1f%% | 对话消息 本批 %d 条，此前已处理 %d/%d，本批完成后 %d/%d (%.1f%%)，尚余 %d 条",
                k, total, pct_b, n, done_before, total_msgs, done_after, total_msgs, pct_m, rem,
            )
            _logger.debug("当前消息: %s; 上文前三条: %s", batch, context[:3] if context else [])
            memory = _process_data_truncation(memory, batch, context)
            _write_construction_progress(k, total, messages_done=done_after, total_messages=total_msgs)
            done_before = done_after
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    msg=f"{done_after}/{total_msgs}",
                    n_cls=len(memory.graph.nodes),
                )
    if total:
        _write_construction_progress(total, total, messages_done=total_msgs, total_messages=total_msgs)
    return memory



def save(data, conv_name):
    # 调用函数
    result = conv_message_splitter(data)

    memory = ClassGraph()
    memory.filepath = conv_name

    total = len(result)
    total_msgs, _ = _conversation_message_totals(result)
    _logger.info("构图开始: 共 %d 个批次，合计 %d 条对话消息", total, total_msgs)
    _twrite(f"构图开始: {total} 批次，合计 {total_msgs} 条对话消息（每批最多 10 条，与主程序 conv_message_splitter 一致）")

    done_before = 0
    pbar = _progress_bar(result, total, "构图")
    for i, (batch, context) in enumerate(pbar):
        n = len(batch)
        k, pct_b = i + 1, 100.0 * (i + 1) / total
        done_after = done_before + n
        pct_m = 100.0 * done_after / total_msgs if total_msgs else 0.0
        rem = total_msgs - done_after
        _logger.debug(
            "构图进度: 批 [%d/%d] %.1f%% | 对话消息 本批 %d 条，此前已处理 %d/%d，本批完成后 %d/%d (%.1f%%)，尚余 %d 条 → 当前类数: %d",
            k, total, pct_b, n, done_before, total_msgs, done_after, total_msgs, pct_m, rem,
            len(memory.graph.nodes),
        )
        _logger.debug("当前消息: %s; 上文: %s", batch, context[:3] if context else [])
        memory = _process_data_truncation(memory, batch, context)
        _write_construction_progress(k, total, messages_done=done_after, total_messages=total_msgs)
        done_before = done_after
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(msgs=f"{done_after}/{total_msgs}", n_cls=len(memory.graph.nodes))
    if total:
        _write_construction_progress(total, total, messages_done=total_msgs, total_messages=total_msgs)
    _logger.info("构图完成: 共 %d 个类，累计处理对话消息 %d 条", len(memory.graph.nodes), total_msgs)
    _twrite(f"构图完成: {len(memory.graph.nodes)} 个类，累计 {total_msgs} 条对话消息")
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
    total = len(result)
    total_msgs, _ = _conversation_message_totals(result)
    _logger.info("构图开始(hash): 共 %d 个批次，合计 %d 条对话消息", total, total_msgs)
    _twrite(f"构图开始(hash): {total} 批次，合计 {total_msgs} 条对话消息")

    done_before = 0
    pbar = _progress_bar(result, total, "构图(hash)")
    for i, (batch, context) in enumerate(pbar):
        n = len(batch)
        k, pct_b = i + 1, 100.0 * (i + 1) / total
        done_after = done_before + n
        pct_m = 100.0 * done_after / total_msgs if total_msgs else 0.0
        rem = total_msgs - done_after
        _logger.debug(
            "构图进度: 批 [%d/%d] %.1f%% | 对话消息 本批 %d 条，此前已处理 %d/%d，本批完成后 %d/%d (%.1f%%)，尚余 %d 条 → 当前类数: %d",
            k, total, pct_b, n, done_before, total_msgs, done_after, total_msgs, pct_m, rem,
            len(memory.graph.nodes),
        )
        memory = _process_data_truncation_hash(memory, batch, context)
        _write_construction_progress(k, total, messages_done=done_after, total_messages=total_msgs)
        done_before = done_after
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(msgs=f"{done_after}/{total_msgs}", n_cls=len(memory.graph.nodes))
    if total:
        _write_construction_progress(total, total, messages_done=total_msgs, total_messages=total_msgs)
    _logger.info("构图完成: 共 %d 个类，累计处理对话消息 %d 条", len(memory.graph.nodes), total_msgs)
    _twrite(f"构图完成: {len(memory.graph.nodes)} 个类，累计 {total_msgs} 条对话消息")
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
    file_pattern = "./locomo results/conv/locomo_conv*.json"

    conv_files = [
        "./locomo results/conv/locomo_conv1.json",
        "./locomo results/conv/locomo_conv2.json",
        "./locomo results/conv/locomo_conv5.json",
        "./locomo results/conv/locomo_conv6.json",
        "./locomo results/conv/locomo_conv8.json"
    ]



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
#
#     # #高血压数据用于去做error_case
#     # file_name = "D:/model/conv/GraphConv/oop_graph/src/error_case/chunks.json"
#     # with open(file_name, 'r', encoding='utf-8') as f:
#     #     data = json.load(f)
#     # save_error(data)
