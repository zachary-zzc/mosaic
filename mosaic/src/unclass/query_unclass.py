import argparse
import json
import os
from collections import defaultdict
from string import Template

from datasets import tqdm

from src.unclass.graph_unclass import InstanceGraph
#from prompts_ch import *
from src.prompts_en import *
from src.assist import fetch_default_llm_model, query_question, serialize, read_to_file_json, load_graphs
from src.logger import setup_logger

_logger = setup_logger("graph query")


def _query_by_heuristic(query: str,
                        memory: InstanceGraph
                        ):
    ret = memory._search_by_sub_hash(query)
    llm = fetch_default_llm_model()
    return query_question(llm, query, ret, PROMPT_QUERY_TEMPLATE)


def query(query,
          memory: InstanceGraph,
          method: str = "llm"
          ):
    if method == "llm":
        return _query_by_heuristic(query, memory)
    elif method == "hash":
        return _query_by_heuristic(query, memory)


def Judge_as_llm(question, gold_answer, generated_answer):
    judge_prompt = Template(JUDGE_ANSWER).substitute(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer
    )
    llm = fetch_default_llm_model()
    # self._logger.info(f"UPDATE_SINGLE_NODE_PROMPT {update_single_node}")
    response = llm.invoke(judge_prompt)
    _logger.info(f"JUDGE_ANSWER_RESPONSE {response.content}")
    judge_ans = json.loads(response.content)
    # print(update_single_node)
    return judge_ans


def process_single_qa(qa_file_path, graph_file_path, tag_file_path, output_file_path, summary_file_path):
    """处理单个qa文件的查询"""
    print(f"\n处理QA文件: {os.path.basename(qa_file_path)}")
    print(f"使用图文件: {os.path.basename(graph_file_path)}")
    print(f"使用标签文件: {os.path.basename(tag_file_path)}")
    print(f"输出文件: {output_file_path}")

    # 读取QA数据
    questions = read_to_file_json(qa_file_path)

    qa_results = []
    category_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
    total_correct = 0
    total_wrong = 0
    total_count = 0
    error_records = []

    # 初始化图
    memory = InstanceGraph()
    memory._all_instances = read_to_file_json(graph_file_path)

    #questions = questions[:2]
    # 处理每个问题
    for i, qa_item in enumerate(tqdm(questions, desc="Processing QA"), start=1):
        try:
            question = qa_item["question"]
            category = qa_item.get("category", 0)

            # 跳过 category=5 的问题
            if category == 5:
                continue

            # 获取期望答案
            expected_answer = qa_item.get("adversarial_answer") or qa_item.get("answer", "")
            if not isinstance(expected_answer, str):
                expected_answer = str(expected_answer).strip()

            # 检查图数据是否加载成功
            if memory.graph is None:
                raise Exception("图数据未正确加载，无法处理问题")

            # 查询
            answer = query(question, memory, "hash")

            _logger.info(f"expected_answer：{expected_answer}; answer:{answer}")

            # 评估答案
            judgment = Judge_as_llm(question, expected_answer, answer)
            label = judgment.get("label", "").strip().upper()

            # 记录结果
            qa_item.update({
                "generated_answer": answer,
                "judgment": label
            })
            qa_results.append(qa_item)

            # 更新统计
            total_count += 1
            if label == "CORRECT":
                total_correct += 1
            elif label == "WRONG":
                total_wrong += 1

            # 更新分类统计
            category_stats[category]["total"] += 1
            if label == "CORRECT":
                category_stats[category]["correct"] += 1
            elif label == "WRONG":
                category_stats[category]["wrong"] += 1

            # 定期打印进度
            if total_count % 10 == 0:
                acc = total_correct / total_count if total_count > 0 else 0
                print(f"当前进度: 已处理 {total_count} 个问题，准确率: {acc:.2%}")

            #实时保存结果
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(qa_results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 记录错误信息
            error_info = {
                "question_index": i,
                "question": qa_item.get("question", "未知问题"),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            error_records.append(error_info)
            print(f"处理问题 {i} 时出错: {e}")
            continue

    # 生成统计信息
    summary = {
        "total_questions": total_count,
        "total_correct": total_correct,
        "total_wrong": total_wrong,
        "overall_accuracy": total_correct / total_count if total_count > 0 else 0,
        "category_stats": dict(category_stats),
        "errors": error_records
    }

    result_data = {
        "qa_file": os.path.basename(qa_file_path),
        "graph_file": os.path.basename(graph_file_path),
        "tag_file": os.path.basename(tag_file_path),
        "summary": summary,
        "results": qa_results
    }

    with open(summary_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {summary_file_path}")

    # 打印报告
    print("\n" + "=" * 50)
    print(f"✅ {os.path.basename(qa_file_path)} 处理完成!")
    print("=" * 50)

    if error_records:
        print(f"❌ 错误统计: 共 {len(error_records)} 个问题处理失败")
    else:
        print("✅ 所有问题处理成功，无错误发生")

    # 打印分类准确率
    print("\n分类准确率:")
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"Category {cat}: {stats['correct']}/{stats['total']} = {acc:.2%}")

    # 打印总体准确率
    overall_acc = total_correct / total_count if total_count > 0 else 0
    print(f"\n总体准确率: {total_correct}/{total_count} = {overall_acc:.2%}")
    print(f"正确: {total_correct} | 错误: {total_wrong}")

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
