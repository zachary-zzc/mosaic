#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conv5 Full Experiment — build graphs (hybrid + hash_only) and run QA evaluation.

Usage (from project root):
    # Full pipeline: build both graphs + evaluate
    python _temp_exp/run_conv5.py

    # Build only (skip QA)
    python _temp_exp/run_conv5.py --skip-qa

    # QA only (reuse existing graphs)
    python _temp_exp/run_conv5.py --skip-build

    # Single strategy
    python _temp_exp/run_conv5.py --strategy hybrid
    python _temp_exp/run_conv5.py --strategy hash_only

    # Limit QA questions (faster debugging)
    python _temp_exp/run_conv5.py --max-questions 5

    # Multi-hop only evaluation
    python _temp_exp/run_conv5.py --skip-build --category 1
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOSAIC_DIR = os.path.join(PROJECT_ROOT, "mosaic")
MOSAIC_SRC = os.path.join(MOSAIC_DIR, "src")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "locomo")
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXP_DIR, "results")

CONV_ID = "conv5"
CONV_JSON = os.path.join(DATASET_DIR, "locomo_conv5.json")
QA_JSON = os.path.join(DATASET_DIR, "qa_5.json")

CATEGORY_NAMES = {1: "Multi-hop", 2: "Temporal", 3: "Single-hop", 4: "Open-domain"}

# ── Helpers ─────────────────────────────────────────────────────────


def setup_mosaic_path():
    for p in (MOSAIC_DIR, MOSAIC_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data, indent=2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def hr(char="=", width=70):
    print(char * width)


# ── Build ───────────────────────────────────────────────────────────


def build_graph(strategy):
    """Build graph via mosaic CLI subprocess. Returns (graph_pkl, tags_json, elapsed_s)."""
    art_dir = os.path.join(EXP_DIR, "artifacts", strategy)
    log_dir = os.path.join(EXP_DIR, "logs", strategy)
    os.makedirs(os.path.join(art_dir, "graph_snapshots"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    graph_pkl = os.path.join(art_dir, f"graph_network_{CONV_ID}.pkl")
    tags_json = os.path.join(art_dir, f"{CONV_ID}_tags.json")

    env = os.environ.copy()
    env["PYTHONPATH"] = MOSAIC_DIR
    env["GRAPH_SAVE_DIR"] = os.path.join(art_dir, "graph_snapshots")
    env["MOSAIC_LLM_IO_LOG"] = os.path.join(log_dir, "llm_io.jsonl")
    env["MOSAIC_INGEST_JSONL"] = os.path.join(log_dir, "ingest.jsonl")
    env["MOSAIC_PROGRESS_FILE"] = os.path.join(log_dir, "progress.txt")

    cmd = [
        sys.executable, "-m", "mosaic", "build",
        "--conv-json", CONV_JSON,
        "--conv-name", CONV_ID,
        "--graph-save-dir", os.path.join(art_dir, "graph_snapshots"),
        "--graph-out", graph_pkl,
        "--tags-out", tags_json,
        "--progress-file", os.path.join(log_dir, "progress.txt"),
        "--log-prompt", os.path.join(log_dir, "llm_io.jsonl"),
    ]
    if strategy == "hash_only":
        cmd.append("--hash")

    print(f"  [build] {CONV_ID} strategy={strategy}")
    print(f"  [build] graph → {graph_pkl}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=MOSAIC_DIR)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [build] FAILED (exit {result.returncode})")
        return None, None, elapsed

    print(f"  [build] done in {elapsed:.0f}s")
    return graph_pkl, tags_json, elapsed


# ── QA Evaluation ───────────────────────────────────────────────────


def run_qa_eval(graph_pkl, tags_json, strategy, *, max_questions=None, category_filter=None):
    """Run QA evaluation. Returns (summary_dict, elapsed_s)."""
    setup_mosaic_path()
    from query import process_single_qa

    out_dir = os.path.join(RESULTS_DIR, strategy)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"qa_5_eval_full_{strategy}.json")
    sum_path = os.path.join(out_dir, f"qa_5_eval_summary_{strategy}.json")

    print(f"  [qa] {CONV_ID} strategy={strategy}")
    t0 = time.time()
    try:
        result = process_single_qa(
            QA_JSON, graph_pkl, tags_json,
            out_path, sum_path,
            max_questions=max_questions,
        )
    except Exception as e:
        import traceback
        print(f"  [qa] FAILED: {e}")
        traceback.print_exc()
        return None, time.time() - t0

    elapsed = time.time() - t0
    print(f"  [qa] done in {elapsed:.0f}s")

    summary = load_json(sum_path) if os.path.isfile(sum_path) else {}
    return summary, elapsed


def filter_qa_results_by_category(results_path, category):
    """Print results filtered by QA category."""
    if not os.path.isfile(results_path):
        return
    data = load_json(results_path)
    results = data.get("results", [])
    filtered = [r for r in results if r.get("category") == category]
    correct = sum(1 for r in filtered if r.get("judgment") == "CORRECT")
    total = len(filtered)
    cat_name = CATEGORY_NAMES.get(category, f"cat{category}")
    print(f"    {cat_name} (cat={category}): {correct}/{total} "
          f"({100*correct/total:.1f}%)" if total > 0 else f"    {cat_name}: 0/0")

    if total > 0:
        wrong = [r for r in filtered if r.get("judgment") != "CORRECT"]
        if wrong:
            print(f"    Wrong answers ({len(wrong)}):")
            for r in wrong[:10]:
                print(f"      Q: {r.get('question','')[:80]}")
                print(f"      Gold: {r.get('gold_answer','')[:80]}")
                print(f"      Got:  {r.get('generated_answer','')[:80]}")
                print()


# ── Report ──────────────────────────────────────────────────────────


def print_summary(strategy, summary):
    """Print formatted summary for one strategy."""
    s = summary.get("summary", {})
    total = s.get("total_questions", 0)
    correct = s.get("total_correct", 0)
    acc = s.get("overall_accuracy", 0)

    print(f"\n  Strategy: {strategy}")
    print(f"  Overall: {correct}/{total} ({100*acc:.1f}%)")

    cat_stats = s.get("category_stats", {})
    for cat_key in sorted(cat_stats.keys(), key=lambda x: int(x)):
        cat = cat_stats[cat_key]
        c = cat.get("correct", 0)
        t = cat.get("total", 0)
        a = cat.get("accuracy", 0)
        name = CATEGORY_NAMES.get(int(cat_key), f"cat{cat_key}")
        print(f"    {name}: {c}/{t} ({100*a:.1f}%)")


def write_results_markdown(timing, summaries):
    """Write a markdown summary to results dir."""
    lines = [
        f"# Conv5 Experiment Results",
        f"",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
    ]

    for strategy, summary in summaries.items():
        if summary is None:
            continue
        s = summary.get("summary", {})
        total = s.get("total_questions", 0)
        correct = s.get("total_correct", 0)
        acc = s.get("overall_accuracy", 0)

        lines.append(f"## Strategy: {strategy}")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Overall | {correct}/{total} ({100*acc:.1f}%) |")

        cat_stats = s.get("category_stats", {})
        for cat_key in sorted(cat_stats.keys(), key=lambda x: int(x)):
            cat = cat_stats[cat_key]
            c = cat.get("correct", 0)
            t = cat.get("total", 0)
            a = cat.get("accuracy", 0)
            name = CATEGORY_NAMES.get(int(cat_key), f"cat{cat_key}")
            lines.append(f"| {name} | {c}/{t} ({100*a:.1f}%) |")

        t_info = timing.get(strategy, {})
        if t_info:
            lines.append(f"")
            lines.append(f"Build: {t_info.get('build_s', 'N/A')}s | QA: {t_info.get('qa_s', 'N/A')}s")
        lines.append(f"")

    md_path = os.path.join(RESULTS_DIR, "RESULTS.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nResults written to {md_path}")


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Conv5 full experiment: build + QA")
    parser.add_argument("--strategy", choices=["hybrid", "hash_only"],
                        help="Run only one strategy (default: both)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip graph construction (reuse existing)")
    parser.add_argument("--skip-qa", action="store_true",
                        help="Skip QA evaluation")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit QA questions (for debugging)")
    parser.add_argument("--category", type=int, default=None,
                        help="After eval, print detailed results for this category only")
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else ["hybrid", "hash_only"]

    # Validate inputs
    if not os.path.isfile(CONV_JSON):
        print(f"[error] Conversation file not found: {CONV_JSON}")
        sys.exit(1)
    if not os.path.isfile(QA_JSON):
        print(f"[error] QA file not found: {QA_JSON}")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    hr()
    print(f"  Conv5 Experiment — {timestamp}")
    print(f"  Strategies: {strategies}")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Output:  {EXP_DIR}")
    hr()

    timing = {}
    summaries = {}
    t_start = time.time()

    for strategy in strategies:
        hr("#")
        print(f"#  {CONV_ID} — {strategy}")
        hr("#")

        art_dir = os.path.join(EXP_DIR, "artifacts", strategy)
        graph_pkl = os.path.join(art_dir, f"graph_network_{CONV_ID}.pkl")
        tags_json = os.path.join(art_dir, f"{CONV_ID}_tags.json")
        timing[strategy] = {}

        # ── Build ──
        if not args.skip_build:
            graph_pkl, tags_json, build_time = build_graph(strategy)
            timing[strategy]["build_s"] = round(build_time, 1)
            if graph_pkl is None:
                print(f"  [skip] build failed for {strategy}")
                summaries[strategy] = None
                continue
        else:
            if not os.path.isfile(graph_pkl):
                print(f"  [skip] no graph at {graph_pkl}")
                summaries[strategy] = None
                continue
            print(f"  [reuse] {graph_pkl}")

        # ── QA ──
        if not args.skip_qa:
            summary, qa_time = run_qa_eval(
                graph_pkl, tags_json, strategy,
                max_questions=args.max_questions,
            )
            timing[strategy]["qa_s"] = round(qa_time, 1)
            summaries[strategy] = summary
        else:
            summaries[strategy] = None

    total_wall = time.time() - t_start
    timing["total_wall_s"] = round(total_wall, 1)

    # ── Report ──
    hr()
    print("RESULTS")
    hr()

    for strategy in strategies:
        summary = summaries.get(strategy)
        if summary:
            print_summary(strategy, summary)

            # Detailed category view if requested
            if args.category is not None:
                out_dir = os.path.join(RESULTS_DIR, strategy)
                full_path = os.path.join(out_dir, f"qa_5_eval_full_{strategy}.json")
                print(f"\n  Detailed category {args.category} results:")
                filter_qa_results_by_category(full_path, args.category)

    hr()
    print(f"Total wall time: {total_wall:.0f}s")
    for strategy, t in timing.items():
        if strategy == "total_wall_s":
            continue
        if isinstance(t, dict):
            parts = " | ".join(f"{k}={v}" for k, v in t.items())
            print(f"  {strategy}: {parts}")
    hr()

    # Save timing + markdown
    save_json(os.path.join(RESULTS_DIR, "timing.json"), timing)
    write_results_markdown(timing, summaries)


if __name__ == "__main__":
    main()
