import json
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.data.graph import ClassGraph
from src.prompts_en import PROMPT_QUERY_TEMPLATE
from src.assist import fetch_default_llm_model, query_question, read_to_file_json, load_graphs
from src.logger import setup_logger
from src.qa_common import judge_answer_llm, run_qa_loop

_logger = setup_logger("graph query")


def _query_by_llm(query: str, memory: ClassGraph, llm=None):
    if llm is None:
        llm = fetch_default_llm_model()
    ret, _trace = memory._search_by_sub_llm(query, llm)
    return query_question(llm, query, ret, PROMPT_QUERY_TEMPLATE)


def _query_by_heuristic(query: str, memory: ClassGraph) -> str:
    ret, _trace = memory._search_by_sub_hash(query)
    llm = fetch_default_llm_model()
    return query_question(llm, query, ret, PROMPT_QUERY_TEMPLATE)


def query(query: str, memory: ClassGraph, method: str = "llm") -> str:
    if method == "llm":
        return _query_by_llm(query, memory)
    if method == "hash":
        return _query_by_heuristic(query, memory)
    raise ValueError(f"Unknown method: {method}")


def query_with_telemetry(question: str, memory: ClassGraph, method: str = "llm") -> dict:
    """
    检索 + 作答，并返回 E-1 字段：retrieved_context、graph_stats（docs/optimization.md §7 D-3）。
    """
    llm = fetch_default_llm_model()
    if method == "llm":
        ctx, rctx = memory._search_by_sub_llm(question, llm)
    elif method == "hash":
        ctx, rctx = memory._search_by_sub_hash(question)
    else:
        raise ValueError(f"Unknown method: {method}")
    ans = query_question(llm, question, ctx, PROMPT_QUERY_TEMPLATE)
    return {
        "answer": ans,
        "retrieved_context": rctx,
        "graph_stats": memory.graph_stats_for_qa(),
    }


def _print_qa_summary(qa_results, category_stats, error_records, qa_file_name):
    total_count = len(qa_results)
    total_correct = sum(1 for r in qa_results if r.get("judgment") == "CORRECT")
    total_wrong = sum(1 for r in qa_results if r.get("judgment") == "WRONG")
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
    print(f"正确: {total_correct} | 错误: {total_wrong}")


def process_single_qa(
    qa_file_path,
    graph_file_path,
    tag_file_path,
    output_file_path,
    summary_file_path,
    max_questions=None,
    *,
    method: str = "hash",
):
    """处理单个 qa 文件：检索 + LLM 作答 + LLM 评判。method: \"llm\"（类感知+实例检索用 LLM 路径）或 \"hash\"（TF-IDF 检索）。"""
    print(f"\n处理QA文件: {os.path.basename(qa_file_path)}")
    print(f"使用图文件: {os.path.basename(graph_file_path)}")
    print(f"使用标签文件: {os.path.basename(tag_file_path)}")
    print(f"检索方式: {method}")
    print(f"输出文件: {output_file_path}")

    questions = read_to_file_json(qa_file_path)
    memory = ClassGraph()
    if not os.path.exists(graph_file_path):
        print(f"图文件不存在: {graph_file_path}")
        return None
    memory.graph = load_graphs(graph_file_path)
    memory.process_kw(tag_file_path)

    query_fn = lambda q, mem: query_with_telemetry(q, mem, method)
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
    total_wrong = sum(1 for r in qa_results if r.get("judgment") == "WRONG")

    summary = {
        "total_questions": total_count,
        "total_correct": total_correct,
        "total_wrong": total_wrong,
        "overall_accuracy": total_correct / total_count if total_count > 0 else 0,
        "category_stats": category_stats,
        "errors": error_records,
    }
    result_data = {
        "qa_eval_schema_version": 1,
        "qa_file": os.path.basename(qa_file_path),
        "graph_file": os.path.basename(graph_file_path),
        "tag_file": os.path.basename(tag_file_path),
        "query_method": method,
        "summary": summary,
        "results": qa_results,
    }
    summary_only = {
        "qa_file": os.path.basename(qa_file_path),
        "graph_file": os.path.basename(graph_file_path),
        "tag_file": os.path.basename(tag_file_path),
        "query_method": method,
        "summary": summary,
    }
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n完整结果已保存到: {output_file_path}")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        json.dump(summary_only if output_file_path else result_data, f, ensure_ascii=False, indent=2)
    print(f"汇总已保存到: {summary_file_path}")
    _print_qa_summary(qa_results, category_stats, error_records, os.path.basename(qa_file_path))
    return result_data


def batch_process_qas_selected():
    """批量处理选定的qa文件"""
    # 只处理指定的文件
    file_list = [
        # ("qa_9", "graph_network_conv9_20260312.pkl", "conv9_tag_new.json", "qa_9_answer.json"),
        ("qa_0", "graph_network_conv00_20260311.pkl", "conv0_tag_new.json", "qa_0_p_answer.json"),
        ("qa_3", "graph_network_conv33_20260312.pkl", "conv3_tag_new.json", "qa_3_p_answer.json"),
        ("qa_4", "graph_network_conv44_20260312.pkl", "conv4_tag_new.json", "qa_4_p_answer.json"),
        ("qa_7", "graph_network_conv77_20260311.pkl", "conv7_tag_new.json", "qa_7_p_answer.json"),
        ("qa_9", "graph_network_conv99_20260312.pkl", "conv9_tag_new.json", "qa_9_p_answer.json"),
        # ("qa_3", "graph_network_conv3_20260312.pkl", "conv3_tag_new.json","qa_3_answer.json"),
        # ("qa_7", "graph_network_conv7_20260312.pkl", "conv7_tag_new.json","qa_7_answer.json"),
    ]


    # 定义基础路径
    qa_dir = "D:/model/conv/GraphConv/oop_graph/src/locomo results/qa/"
    graph_dir = "D:/model/conv/GraphConv/results/graph/"
    tag_dir = "D:/model/conv/GraphConv/results/tags/"
    output_dir = "D:/model/conv/GraphConv/results/answers/"
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

