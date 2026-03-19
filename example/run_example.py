"""
在示例数据上运行 mosaic 构图与查询流程（可选 LLM）。

用法（在 LongtermMemory 根目录）:
  # 1. 先构建最小图与 tags（无需 API）
  PYTHONPATH=mosaic python example/build_minimal_graph.py

  # 2. 运行查询示例（需要配置 config/config.cfg 中的 API key 才会调用 LLM）
  PYTHONPATH=mosaic python example/run_example.py

  # 3. 仅测试加载与检索，不调用 LLM（--no-llm 时用占位答案，便于无 API 环境测试）
  PYTHONPATH=mosaic python example/run_example.py --no-llm
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.dirname(os.path.abspath(__file__))
MOSAIC = os.path.join(ROOT, "mosaic")
if MOSAIC not in sys.path:
    sys.path.insert(0, MOSAIC)

# 可选：不加载 embedding/KeyBERT，减少本地依赖
os.chdir(MOSAIC)


def main():
    parser = argparse.ArgumentParser(description="Run mosaic example (build graph + query)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls; use placeholder answers for pipeline test")
    parser.add_argument("--max-questions", type=int, default=2, help="Max QA pairs to run (default 2)")
    args = parser.parse_args()

    out_dir = os.path.join(EXAMPLE, "output")
    graph_path = os.path.join(out_dir, "graph_small.pkl")
    tags_path = os.path.join(out_dir, "tags_small.json")
    qa_path = os.path.join(EXAMPLE, "qa_small.json")

    if not os.path.isfile(graph_path) or not os.path.isfile(tags_path):
        print("请先运行: PYTHONPATH=mosaic python example/build_minimal_graph.py")
        sys.exit(1)
    if not os.path.isfile(qa_path):
        print(f"未找到 {qa_path}")
        sys.exit(1)

    from src.data.graph import ClassGraph
    from src.assist import load_graphs, read_to_file_json

    memory = ClassGraph()
    memory.graph = load_graphs(graph_path)
    memory.process_kw(tags_path)

    questions = read_to_file_json(qa_path)
    questions = questions[: args.max_questions]

    print("示例问题与检索结果（前几条）:\n")
    for i, qa in enumerate(questions):
        q = qa.get("question", "")
        serialized = memory._search_by_sub_hash(q, top_k_class=5, top_k_instances=5)
        snippet = (serialized[:400] + "...") if len(serialized) > 400 else serialized
        print(f"Q{i+1}: {q}")
        print(f"检索片段:\n{snippet}\n")

    if args.no_llm:
        print("(--no-llm: 已跳过 LLM 调用，仅测试加载与检索。)")
        return

    from src.query import query, process_single_qa

    result_dir = os.path.join(out_dir, "qa_results")
    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, "summary_small.json")
    output_path = os.path.join(result_dir, "results_small.json")

    result = process_single_qa(
        qa_file_path=qa_path,
        graph_file_path=graph_path,
        tag_file_path=tags_path,
        output_file_path=output_path,
        summary_file_path=summary_path,
        max_questions=args.max_questions,
    )
    if result:
        print("\n完整结果已写入:", summary_path)


if __name__ == "__main__":
    main()
