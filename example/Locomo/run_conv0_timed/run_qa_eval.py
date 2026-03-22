#!/usr/bin/env python3
"""
构图完成后：对 example/Locomo/qa_0.json 逐题检索 + LLM 作答，再用与 mosaic 相同的 API（llm.py / fetch_default_llm_model）
做 JUDGE_ANSWER 评判；按 category 汇总与整体准确率。

输出：
  - results/qa_0_eval_full.json   完整逐题结果（含 judgment、generated_answer）
  - results/qa_0_eval_summary.json 汇总（category_stats、overall_accuracy）

日志：
  - log/qa_eval.log  （mosaic 库详细日志，与 experiments/02_baselines/step2 行为一致）

用法（需先完成 run.py 构图）:
  conda activate mosaic
  cd example/Locomo/run_conv0_timed
  python run_qa_eval.py
  python run_qa_eval.py --method llm
  python run_qa_eval.py --method hash --max-questions 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from paths_config import (
    apply_mosaic_env,
    build_resolved_paths,
    default_graph_and_tags,
    load_paths_raw,
    repo_root_from_file,
    run_dir_from_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="conv0 构图产物 + qa_0 评测（LLM 评判 + 分类统计）")
    parser.add_argument("--paths", type=Path, default=None, help="paths.json，默认本目录")
    parser.add_argument(
        "--method",
        choices=("llm", "hash"),
        default="llm",
        help="检索方式：llm=与 mosaic query.py 一致；hash=TF-IDF 检索（基线）",
    )
    parser.add_argument("--max-questions", type=int, default=None, help="最多评测题数")
    args = parser.parse_args()

    this = Path(__file__)
    run_dir = run_dir_from_file(this)
    repo = repo_root_from_file(this)
    paths_file = args.paths or (run_dir / "paths.json")
    if not paths_file.is_file():
        print(f"[错误] 找不到 {paths_file}", file=sys.stderr)
        return 1

    raw = load_paths_raw(paths_file)
    paths = build_resolved_paths(repo, run_dir, raw)

    artifacts = paths["artifacts_dir"]
    results_dir = paths["results_dir"]
    log_dir = paths["log_dir"]
    assert isinstance(artifacts, Path) and isinstance(results_dir, Path) and isinstance(log_dir, Path)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    graph_pkl, tags_json = default_graph_and_tags(artifacts)
    qa_json = paths["qa_json"]
    assert isinstance(qa_json, Path)

    if not graph_pkl.is_file():
        print(f"[错误] 未找到构图产物: {graph_pkl}\n请先在本目录运行: python run.py", file=sys.stderr)
        return 1
    if not tags_json.is_file():
        print(f"[错误] 未找到 tags: {tags_json}", file=sys.stderr)
        return 1
    if not qa_json.is_file():
        print(f"[错误] 未找到 QA 文件: {qa_json}", file=sys.stderr)
        return 1

    mcfg = paths["mosaic_config_path"]
    assert isinstance(mcfg, Path)
    if not mcfg.is_file():
        print(f"[警告] 配置文件不存在: {mcfg}", file=sys.stderr)

    apply_mosaic_env(
        paths,
        log_basename="qa_eval.log",
        progress_relpath=None,
        set_graph_snapshots_dir=False,
    )

    mosaic_root = repo / "mosaic"
    mosaic_src = mosaic_root / "src"
    # 与 mosaic/cli.py、experiments/run_utils.setup_mosaic_path 一致：src.* 导入需 mosaic 在 PYTHONPATH 上
    for p in (mosaic_root, mosaic_src):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.chdir(mosaic_src)

    from src.query import process_single_qa

    out_full = results_dir / "qa_0_eval_full.json"
    out_summary = results_dir / "qa_0_eval_summary.json"

    print(f"[任务] QA 文件: {qa_json}")
    print(f"[任务] 图: {graph_pkl}")
    print(f"[任务] tags: {tags_json}")
    print(f"[任务] 检索方式: {args.method}")
    print(f"[任务] 结果目录: {results_dir}")
    print(f"[任务] 详细日志: {log_dir / 'qa_eval.log'}")

    process_single_qa(
        str(qa_json),
        str(graph_pkl),
        str(tags_json),
        str(out_full),
        str(out_summary),
        max_questions=args.max_questions,
        method=args.method,
    )
    print(f"[完成] 汇总: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
