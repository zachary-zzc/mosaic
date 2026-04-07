#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoCoMo Experiment — Full pipeline: graph construction + QA evaluation.

Supports two MOSAIC graph variants:
  - hybrid  (dynamic/evolving graph): TF-IDF + LLM joint graph construction
  - hash_only (static graph): TF-IDF/hash baseline graph construction

Reads data from dataset/locomo/ and writes results to experiments/locomo/results/
and runs to experiments/locomo/runs/.

Usage (from project root):
  python experiments/locomo/benchmark/run.py                          # both strategies
  python experiments/locomo/benchmark/run.py --strategy hybrid        # evolving graph only
  python experiments/locomo/benchmark/run.py --strategy hash_only     # static graph only
  python experiments/locomo/benchmark/run.py --skip-build             # QA only (reuse graphs)
  python experiments/locomo/benchmark/run.py --skip-qa                # build only
  python experiments/locomo/benchmark/run.py --max-questions 10       # limit QA questions
  python experiments/locomo/benchmark/run.py --conv conv0             # single conversation
"""
import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.locomo._utils import (
    setup_mosaic_path,
    load_json_safe,
    save_json,
)
from experiments import reference_values as ref

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

CATEGORY_NAMES = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop"}
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "locomo")
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
RUNS_DIR = os.path.join(EXPERIMENT_DIR, "runs")


def load_sessions():
    """Load experiment session registry from dataset/locomo/experiment_sessions.json."""
    sessions_path = os.path.join(DATASET_DIR, "experiment_sessions.json")
    if not os.path.isfile(sessions_path):
        print(f"[error] Session registry not found: {sessions_path}")
        sys.exit(1)
    data = load_json_safe(sessions_path)
    return data.get("sessions", [])


def build_graph(conv_json_path, conv_name, strategy, run_dir, log_dir):
    """Build graph using mosaic CLI. Returns (graph_pkl, tags_json, elapsed_seconds)."""
    setup_mosaic_path()

    art_dir = os.path.join(run_dir, "artifacts", strategy)
    os.makedirs(os.path.join(art_dir, "graph_snapshots"), exist_ok=True)

    graph_pkl = os.path.join(art_dir, f"graph_network_{conv_name}.pkl")
    tags_json = os.path.join(art_dir, f"{conv_name}_tags.json")

    # Set telemetry environment
    os.environ["MOSAIC_LLM_IO_LOG"] = os.path.join(log_dir, f"llm_io_{strategy}.jsonl")
    os.environ["MOSAIC_INGEST_JSONL"] = os.path.join(log_dir, f"ingest_{strategy}.jsonl")
    os.environ["MOSAIC_NCS_TRACE_JSONL"] = os.path.join(log_dir, f"ncs_trace_{strategy}.jsonl")

    import subprocess
    cmd = [
        sys.executable, "-m", "mosaic", "build",
        "--conv-json", conv_json_path,
        "--conv-name", conv_name,
        "--graph-save-dir", os.path.join(art_dir, "graph_snapshots"),
        "--graph-out", graph_pkl,
        "--tags-out", tags_json,
        "--progress-file", os.path.join(log_dir, f"{conv_name}_progress_{strategy}.txt"),
        "--log-prompt", os.path.join(log_dir, f"llm_io_{strategy}.jsonl"),
    ]
    if strategy == "hash_only":
        cmd.append("--hash")

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    print(f"  [build] {conv_name} strategy={strategy}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [build] FAILED (exit {result.returncode})")
        return None, None, elapsed

    print(f"  [build] done in {elapsed:.0f}s")
    return graph_pkl, tags_json, elapsed


def run_qa_eval(qa_json_path, graph_pkl, tags_json, conv_name, strategy,
                results_dir, max_questions=None):
    """Run QA evaluation. Returns (summary_dict, elapsed_seconds)."""
    setup_mosaic_path()
    from query import process_single_qa

    qa_idx = conv_name.replace("conv", "")
    out_path = os.path.join(results_dir, f"qa_{qa_idx}_eval_full_{strategy}.json")
    sum_path = os.path.join(results_dir, f"qa_{qa_idx}_eval_summary_{strategy}.json")

    print(f"  [qa] {conv_name} strategy={strategy}")
    t0 = time.time()
    try:
        result = process_single_qa(
            qa_json_path, graph_pkl, tags_json,
            out_path, sum_path,
            max_questions=max_questions,
        )
    except Exception as e:
        print(f"  [qa] FAILED: {e}")
        traceback.print_exc()
        return None, time.time() - t0

    elapsed = time.time() - t0
    print(f"  [qa] done in {elapsed:.0f}s")

    summary = load_json_safe(sum_path)
    return summary, elapsed


def aggregate_results(results_dir):
    """Aggregate all QA summaries in results_dir into overall metrics."""
    overall_correct = 0
    overall_total = 0
    by_category = {1: {"correct": 0, "total": 0},
                   2: {"correct": 0, "total": 0},
                   3: {"correct": 0, "total": 0}}

    by_strategy = {}

    for name in sorted(os.listdir(results_dir)):
        if not (name.startswith("qa_") and name.endswith("_summary.json")
                and "eval_summary" in name):
            continue
        path = os.path.join(results_dir, name)
        data = load_json_safe(path)
        summary = data.get("summary", {})

        # Parse strategy from filename: qa_0_eval_summary_hybrid.json
        parts = name.replace(".json", "").split("_")
        strategy = parts[-1] if parts else "unknown"
        if strategy not in by_strategy:
            by_strategy[strategy] = {
                "correct": 0, "total": 0,
                "by_category": {1: {"correct": 0, "total": 0},
                                2: {"correct": 0, "total": 0},
                                3: {"correct": 0, "total": 0}},
            }

        correct = summary.get("total_correct", 0)
        total = summary.get("total_questions", 0)
        overall_correct += correct
        overall_total += total
        by_strategy[strategy]["correct"] += correct
        by_strategy[strategy]["total"] += total

        for cat, stats in summary.get("category_stats", {}).items():
            c = int(cat) if isinstance(cat, str) else cat
            if c in by_category:
                by_category[c]["correct"] += stats.get("correct", 0)
                by_category[c]["total"] += stats.get("total", 0)
            if c in by_strategy[strategy]["by_category"]:
                by_strategy[strategy]["by_category"][c]["correct"] += stats.get("correct", 0)
                by_strategy[strategy]["by_category"][c]["total"] += stats.get("total", 0)

    overall_acc = (overall_correct / overall_total * 100) if overall_total else 0.0
    category_acc = {}
    for c, cat_name in CATEGORY_NAMES.items():
        tot = by_category[c]["total"]
        category_acc[cat_name] = (by_category[c]["correct"] / tot * 100) if tot else 0.0

    # Per-strategy summaries
    strategy_summaries = {}
    for strategy, s_data in by_strategy.items():
        s_total = s_data["total"]
        s_correct = s_data["correct"]
        s_acc = (s_correct / s_total * 100) if s_total else 0.0
        s_cat_acc = {}
        for c, cat_name in CATEGORY_NAMES.items():
            tot = s_data["by_category"][c]["total"]
            s_cat_acc[cat_name] = (s_data["by_category"][c]["correct"] / tot * 100) if tot else 0.0
        strategy_summaries[strategy] = {
            "accuracy_pct": round(s_acc, 2),
            "correct": s_correct,
            "total": s_total,
            "by_category_pct": {k: round(v, 2) for k, v in s_cat_acc.items()},
        }

    return {
        "overall_accuracy_pct": round(overall_acc, 2),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "by_category_pct": {k: round(v, 2) for k, v in category_acc.items()},
        "by_strategy": strategy_summaries,
    }


def build_locomo_table(mosaic_metrics=None):
    """Build full LoCoMo comparison table with reference baselines + MOSAIC results."""
    table = {}
    for method, row in ref.LOCOMO_TABLE.items():
        table[method] = dict(row)
    if mosaic_metrics:
        for strategy, s_data in mosaic_metrics.get("by_strategy", {}).items():
            label = "MOSAIC (Evolving)" if strategy == "hybrid" else "MOSAIC (Static)"
            table[label] = {
                "overall": s_data["accuracy_pct"],
                "single_hop": s_data["by_category_pct"].get("Single-hop", 0),
                "multi_hop": s_data["by_category_pct"].get("Multi-hop", 0),
                "temporal": s_data["by_category_pct"].get("Temporal", 0),
            }
    return table


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Experiment Pipeline")
    parser.add_argument("--strategy", choices=["hybrid", "hash_only"],
                        default=None, help="Graph strategy (default: both)")
    parser.add_argument("--conv", type=str, default=None,
                        help="Single conversation ID (e.g., conv0)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip graph construction (reuse existing graphs)")
    parser.add_argument("--skip-qa", action="store_true",
                        help="Skip QA evaluation")
    parser.add_argument("--skip-aggregate", action="store_true",
                        help="Skip aggregation and table export")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of QA questions per conversation")
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else ["hybrid", "hash_only"]
    sessions = load_sessions()

    # Filter conversations if specified
    if args.conv:
        sessions = [s for s in sessions if s["conv_id"] == args.conv]
        if not sessions:
            print(f"[error] Conversation '{args.conv}' not found in session registry")
            sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    timing = {}
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"{'='*60}")
    print(f"LoCoMo Experiment — {run_timestamp}")
    print(f"  strategies: {strategies}")
    print(f"  conversations: {[s['conv_id'] for s in sessions]}")
    print(f"  dataset: {DATASET_DIR}")
    print(f"  results: {RESULTS_DIR}")
    print(f"{'='*60}\n")

    t_pipeline_start = time.time()

    for session in sessions:
        conv_id = session["conv_id"]
        conv_json = os.path.join(DATASET_DIR, session["conv_json"])
        qa_json = os.path.join(DATASET_DIR, session["qa_json"])

        # Validate input files
        if not os.path.isfile(conv_json):
            print(f"[error] Conversation file not found: {conv_json}")
            continue
        if not os.path.isfile(qa_json):
            print(f"[error] QA file not found: {qa_json}")
            continue

        run_dir = os.path.join(RUNS_DIR, conv_id)
        log_dir = os.path.join(run_dir, "log")
        results_dir = os.path.join(run_dir, "results")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        for strategy in strategies:
            label = f"{conv_id}_{strategy}"
            timing[label] = {}

            print(f"\n{'#'*60}")
            print(f"#  {label}")
            print(f"{'#'*60}")

            # --- Build graph ---
            art_dir = os.path.join(run_dir, "artifacts", strategy)
            graph_pkl = os.path.join(art_dir, f"graph_network_{conv_id}.pkl")
            tags_json = os.path.join(art_dir, f"{conv_id}_tags.json")

            if not args.skip_build:
                graph_pkl, tags_json, build_time = build_graph(
                    conv_json, conv_id, strategy, run_dir, log_dir
                )
                timing[label]["build_s"] = round(build_time, 1)
                if graph_pkl is None:
                    print(f"  [skip] {label}: build failed, skipping QA")
                    continue
            else:
                if not os.path.isfile(graph_pkl):
                    print(f"  [skip] {label}: no graph found at {graph_pkl}")
                    continue
                print(f"  [reuse] graph: {graph_pkl}")

            # --- QA evaluation ---
            if not args.skip_qa:
                summary, qa_time = run_qa_eval(
                    qa_json, graph_pkl, tags_json,
                    conv_id, strategy, results_dir,
                    max_questions=args.max_questions,
                )
                timing[label]["qa_s"] = round(qa_time, 1)

    t_pipeline_end = time.time()
    total_wall = t_pipeline_end - t_pipeline_start

    # --- Save timing ---
    timing["total_wall_s"] = round(total_wall, 1)
    save_json(os.path.join(RESULTS_DIR, "timing.json"), timing)

    # --- Aggregate & export ---
    if not args.skip_aggregate:
        # Collect results from all run directories
        all_results_dirs = []
        for session in sessions:
            rd = os.path.join(RUNS_DIR, session["conv_id"], "results")
            if os.path.isdir(rd):
                all_results_dirs.append(rd)

        # Copy summary files to central results dir for aggregation
        for rd in all_results_dirs:
            for f in os.listdir(rd):
                if f.endswith("_summary.json") or f.endswith("_full.json"):
                    src = os.path.join(rd, f)
                    dst = os.path.join(RESULTS_DIR, f)
                    import shutil
                    shutil.copy2(src, dst)

        metrics = aggregate_results(RESULTS_DIR)
        save_json(os.path.join(RESULTS_DIR, "locomo_metrics.json"), metrics)

        table = build_locomo_table(metrics)
        save_json(os.path.join(RESULTS_DIR, "locomo_table.json"), table)

        # Export LaTeX table
        try:
            tex = _locomo_to_latex(table)
            tex_path = os.path.join(RESULTS_DIR, "tab_locomo.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tex)
            print(f"\nLaTeX table written: {tex_path}")
        except Exception as e:
            print(f"\n[warn] LaTeX export failed: {e}")

    # --- Timing summary ---
    print(f"\n{'='*60}")
    print(f"Timing Summary (seconds)")
    print(f"{'='*60}")
    for key, val in timing.items():
        if key == "total_wall_s":
            continue
        if isinstance(val, dict):
            parts = " ".join(f"{k}={v}" for k, v in val.items())
            print(f"  {key}: {parts}")
    print(f"  total_wall: {timing.get('total_wall_s', '?')}s")
    print(f"\nDone. Results: {RESULTS_DIR}")

    return 0


def _locomo_to_latex(table):
    """Generate LaTeX tabular from the LoCoMo comparison table dict."""
    lines = [
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"Method & Overall & Single-hop & Multi-hop & Temporal \\",
        r"\midrule",
    ]
    for method, row in table.items():
        overall = f'{row.get("overall", 0):.2f}'
        single = f'{row.get("single_hop", 0):.2f}'
        multi = f'{row.get("multi_hop", 0):.2f}'
        temporal = f'{row.get("temporal", 0):.2f}'
        lines.append(f"{method} & {overall} & {single} & {multi} & {temporal} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main() or 0)
