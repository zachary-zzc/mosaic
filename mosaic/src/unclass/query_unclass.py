import json
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.unclass.graph_unclass import InstanceGraph
from src.prompts_en import PROMPT_QUERY_TEMPLATE
from src.assist import fetch_default_llm_model, query_question, read_to_file_json
from src.logger import setup_logger
from src.qa_common import run_qa_loop

_logger = setup_logger("graph query unclass")


def _query_by_heuristic(query: str, memory: InstanceGraph) -> str:
    ret = memory._search_by_sub_hash(query)
    llm = fetch_default_llm_model()
    return query_question(llm, query, ret, PROMPT_QUERY_TEMPLATE)


def query(query: str, memory: InstanceGraph, method: str = "hash") -> str:
    return _query_by_heuristic(query, memory)


def _print_qa_summary(qa_results, category_stats, error_records, qa_file_name):
    total_count = len(qa_results)
    total_correct = sum(1 for r in qa_results if r.get("judgment") == "CORRECT")
    print("\n" + "=" * 50)
    print(f"✅ {qa_file_name} 处理完成!")
    print("=" * 50)
    if error_records:
        print(f"❌ 错误统计: 共 {len(error_records)} 个问题处理失败")
    else:
        print("✅ 所有问题处理成功，无错误发生")
    print("\n分类准确率:")
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"Category {cat}: {stats['correct']}/{stats['total']} = {acc:.2%}")
    if total_count > 0:
        print(f"\n总体准确率: {total_correct}/{total_count} = {total_correct / total_count:.2%}")
    print(f"正确: {total_correct} | 错误: {total_count - total_correct}")


def process_single_qa(qa_file_path, graph_file_path, tag_file_path, output_file_path, summary_file_path, max_questions=None):
    """处理单个qa文件的查询（无类图实例）。"""
    print(f"\n处理QA文件: {os.path.basename(qa_file_path)}")
    print(f"使用图文件: {os.path.basename(graph_file_path)}")
    print(f"输出文件: {output_file_path}")

    questions = read_to_file_json(qa_file_path)
    memory = InstanceGraph()
    if not os.path.exists(graph_file_path):
        print(f"图文件不存在: {graph_file_path}")
        return None
    memory._all_instances = read_to_file_json(graph_file_path)

    query_fn = lambda q, mem: query(q, mem, "hash")
    qa_results, category_stats, error_records = run_qa_loop(
        questions,
        memory,
        query_fn,
        skip_category=5,
        max_questions=max_questions,
        progress_callback=lambda total, correct, acc: print(f"当前进度: 已处理 {total} 个问题，准确率: {acc:.2%}"),
    )

    total_count = len(qa_results)
    total_correct = sum(1 for r in qa_results if r.get("judgment") == "CORRECT")
    summary = {
        "total_questions": total_count,
        "total_correct": total_correct,
        "total_wrong": total_count - total_correct,
        "overall_accuracy": total_correct / total_count if total_count > 0 else 0,
        "category_stats": category_stats,
        "errors": error_records,
    }
    result_data = {
        "qa_file": os.path.basename(qa_file_path),
        "graph_file": os.path.basename(graph_file_path),
        "tag_file": os.path.basename(tag_file_path),
        "summary": summary,
        "results": qa_results,
    }
    with open(summary_file_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {summary_file_path}")
    _print_qa_summary(qa_results, category_stats, error_records, os.path.basename(qa_file_path))
    return result_data


def batch_process_qas_selected():
    """批量处理选定的qa文件"""
    # 只处理指定的文件
    file_list = [
        ("qa_0", "instances_snapshot_conv0_20260313.json", "conv1_tag_new.json", "qa_0_i_answer.json"),
        ("qa_1", "instances_snapshot_conv1_20260313.json", "conv1_tag_new.json", "qa_1_i_answer.json"),
        ("qa_2", "instances_snapshot_conv2_20260313.json", "conv1_tag_new.json", "qa_2_i_answer.json"),
        ("qa_3", "instances_snapshot_conv3_20260313.json", "conv1_tag_new.json", "qa_3_i_answer.json"),
        ("qa_4", "instances_snapshot_conv4_20260313.json", "conv1_tag_new.json", "qa_4_i_answer.json"),
        ("qa_5", "instances_snapshot_conv5_20260314.json", "conv1_tag_new.json", "qa_5_i_answer.json"),
        ("qa_6", "instances_snapshot_conv6_20260314.json", "conv1_tag_new.json", "qa_6_i_answer.json"),
        ("qa_7", "instances_snapshot_conv7_20260314.json", "conv1_tag_new.json", "qa_7_i_answer.json"),
        ("qa_8", "instances_snapshot_conv8_20260314.json", "conv1_tag_new.json", "qa_8_i_answer.json"),
        ("qa_9", "instances_snapshot_conv9_20260314.json", "conv1_tag_new.json", "qa_9_i_answer.json"),
    ]
    # 定义基础路径
    qa_dir = "D:/model/conv/GraphConv/oop_graph/src/locomo results/qa/"
    graph_dir = "D:/model/conv/GraphConv/results/instances/"
    tag_dir = "D:/model/conv/GraphConv/results/tags/"
    output_dir = "D:/model/conv/GraphConv/results/answers/instances_20"
    summary_output_dir = "D:/model/conv/GraphConv/results/answers/summary/"

    all_results = {}

    for qa_name, graph_name, tag_name, output_name in file_list:
        qa_file = os.path.join(qa_dir, f"{qa_name}.json")
        graph_file = os.path.join(graph_dir, graph_name)
        tag_file = os.path.join(tag_dir, tag_name)
        output_file = os.path.join(output_dir, output_name)
        summary_output_file = os.path.join(summary_output_dir, output_name)

        print(qa_file)
        print(graph_file)

        # 检查文件是否存在
        if not all([os.path.exists(qa_file), os.path.exists(graph_file)]):
            print(f"跳过 {qa_name}，文件不完整")
            continue

        print(f"\n处理: {qa_name}")

        result = process_single_qa(
            qa_file_path=qa_file,
            graph_file_path=graph_file,
            tag_file_path=tag_file,
            output_file_path=output_file,
            summary_file_path=summary_output_file
        )

        if result:
            all_results[qa_name] = result

    return all_results


if __name__ == '__main__':
    all_res = batch_process_qas_selected()
