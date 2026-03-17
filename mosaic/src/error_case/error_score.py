import os
import json
from src.assist import read_to_file_json, save_to_file_json, fetch_default_llm_model
from src.prompts_ch import PROMPT_CONFLICT_JUDGE
from string import Template

_llm = fetch_default_llm_model()


def generate_single_error_data_str(error_pair, index):
    """将单个error_data对转换为可读的字符串格式"""
    error_str = f"原始陈述: {error_pair['original_statement']['statement']}\n"
    error_str += f"冲突陈述: {error_pair['conflicting_statement']['statement']}\n"
    return error_str


def save_intermediate_results(
    error_data,
    matched_error_indices,
    matched_warning_indices,
    total_warning_conflicts,
    total_error_pairs,
    output_dir
):
    """实时保存当前匹配结果和统计矩阵"""
    # 提取当前已匹配的 error_data
    matched_error_data = [error_data[i] for i in sorted(matched_error_indices)]

    # 计算当前统计矩阵
    stats_matrix = {
        "交集": len(matched_error_indices),
        "warning_items独有": total_warning_conflicts - len(matched_warning_indices),
        "error_data独有": total_error_pairs - len(matched_error_indices)
    }

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存文件（覆盖写入）
    matched_path = os.path.join(output_dir, "matched_error_data.json")
    matrix_path = os.path.join(output_dir, "conflict_stats_matrix.json")

    save_to_file_json(matched_path, matched_error_data)
    save_to_file_json(matrix_path, stats_matrix)

    return stats_matrix


def main():
    # 读取数据
    filename = "D:/model/conv/GraphConv/oop_graph/src/error_case/error_data1.json"
    error_data = read_to_file_json(filename)

    filename = "D:/model/conv/GraphConv/oop_graph/src/error_case/conflict/warning_items.json"
    warning_items = read_to_file_json(filename)

    total_error_pairs = len(error_data)
    total_warning_conflicts = sum(len(cat["conflicts"]) for cat in warning_items)

    # 初始化匹配记录
    matched_pairs = []
    matched_error_indices = set()
    matched_warning_indices = set()

    # 输出路径
    output_dir = "D:/model/conv/GraphConv/oop_graph/src/error_case/conflict/results/"

    # 实时保存初始状态（空结果）
    save_intermediate_results(
        error_data,
        matched_error_indices,
        matched_warning_indices,
        total_warning_conflicts,
        total_error_pairs,
        output_dir
    )

    print(f"开始处理 {len(warning_items)} 个类别，共 {total_warning_conflicts} 个冲突...")

    # 遍历 warning_items 进行匹配
    processed_conflicts = 0
    for category_idx, category in enumerate(warning_items):
        for conflict_idx, conflict in enumerate(category["conflicts"]):
            processed_conflicts += 1
            warning_id = f"{category_idx}_{conflict_idx}"

            current_match_found = False  # 标记本轮是否有新匹配

            for error_idx, error_pair in enumerate(error_data):
                single_error_str = generate_single_error_data_str(error_pair, error_idx)

                judge_prompt = Template(PROMPT_CONFLICT_JUDGE).substitute(
                    conflict_reason=conflict['conflict_reason'],
                    error_data_str=single_error_str
                )

                try:
                    response = _llm.invoke(judge_prompt)
                    result = json.loads(response.content)
                    match = result.get("label", False) if isinstance(result, dict) else False
                except Exception as e:
                    print(f"⚠️ JSON解析或API调用错误: {e}")
                    match = False

                print(
                    f"[{processed_conflicts}/{total_warning_conflicts}] "
                    f"冲突 '{conflict['conflict_reason'][:30]}...' vs Error{error_idx + 1} → "
                    f"{'✅ 匹配' if match else '❌ 不匹配'}"
                )

                if match:
                    if error_idx not in matched_error_indices or warning_id not in matched_warning_indices:
                        current_match_found = True
                    matched_pairs.append((error_idx, category_idx, conflict_idx))
                    matched_error_indices.add(error_idx)
                    matched_warning_indices.add(warning_id)

            # ✅ 实时保存：每处理完一个 conflict 就更新结果文件
            # （即使无新匹配也更新 stats_matrix，因为 total_warning_conflicts 已推进）
            stats_matrix = save_intermediate_results(
                error_data,
                matched_error_indices,
                matched_warning_indices,
                processed_conflicts,  # 注意：这里用已处理的冲突数作为分母更准确
                total_error_pairs,
                output_dir
            )

            # 可选：打印进度摘要
            if processed_conflicts % 10 == 0 or current_match_found:
                print(f"📌 进度: 已处理 {processed_conflicts}/{total_warning_conflicts} 冲突 | "
                      f"匹配 error_data: {stats_matrix['交集']}/{total_error_pairs}")

    # 最终统计（与最后一次保存一致，但再打印一次）
    print(f"\n{'=' * 60}")
    print("✅ 所有冲突处理完成！最终统计:")
    print(f"{'=' * 60}")

    final_stats = {
        "交集": len(matched_error_indices),
        "warning_items独有": total_warning_conflicts - len(matched_warning_indices),
        "error_data独有": total_error_pairs - len(matched_error_indices)
    }

    warning_denominator = total_warning_conflicts or 1
    error_denominator = total_error_pairs or 1

    print(f"{'交集':<20} | {final_stats['交集']:<8} | {final_stats['交集'] / error_denominator:.2%}")
    print(f"{'warning_items独有':<20} | {final_stats['warning_items独有']:<8} | {final_stats['warning_items独有'] / warning_denominator:.2%}")
    print(f"{'error_data独有':<20} | {final_stats['error_data独有']:<8} | {final_stats['error_data独有'] / error_denominator:.2%}")

    # 多对一分析（仅最终输出）
    error_match_count = {}
    for error_idx, _, _ in matched_pairs:
        error_match_count[error_idx] = error_match_count.get(error_idx, 0) + 1

    if error_match_count:
        print(f"\n📊 匹配分布:")
        print(f"  共 {len(error_match_count)} 个 error_data 被匹配")
        avg = len(matched_pairs) / len(error_match_count)
        print(f"  平均每个被匹配的 error_data 对应 {avg:.1f} 个 warning")

    print(f"\n📁 结果已实时保存至: {output_dir}")
    print(f"   - matched_error_data.json")
    print(f"   - conflict_stats_matrix.json")

    return final_stats, [error_data[i] for i in sorted(matched_error_indices)]


if __name__ == "__main__":
    result_matrix, matched_errors = main()