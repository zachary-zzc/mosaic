#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Experiment — MOSAIC component ablation (C0–C6) on LoCoMo.

Each condition disables one MOSAIC component via environment variables, then
re-runs the build + QA pipeline on LoCoMo conversations.

  C0: Full MOSAIC (reuse benchmark/hybrid results)
  C1: hash_only build (reuse benchmark/hash_only results)
  C2: No relationship edges (MOSAIC_EDGE_SEMANTIC_A=0, MOSAIC_EDGE_PREREQ_LLM=0)
  C3: No graph traversal  (MOSAIC_QUERY_NEIGHBOR_HOPS=0)
  C4: Prerequisite-graph-only traversal (MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS=P)
  C5: Association-graph-only traversal  (MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS=A)
  C6: No edge enhancement, no traversal (C2 + C3 combined)

C0 and C1 copy results from benchmark/; the rest run the pipeline with the
specified overrides.

Usage (from project root):
  python experiments/locomo/ablation/run.py                # all conditions
  python experiments/locomo/ablation/run.py --condition C3  # single condition
  python experiments/locomo/ablation/run.py --conv conv0    # single conversation
  python experiments/locomo/ablation/run.py --skip-build    # QA only (for query-only ablations)
  python experiments/locomo/ablation/run.py --max-questions 10
"""
import os
import sys
import time
import shutil
import argparse
import subprocess
import traceback
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.locomo._utils import (
    setup_mosaic_path,
    load_json_safe,
    save_json,
    load_sessions,
    DATASET_LOCOMO_DIR,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

CATEGORY_NAMES = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop"}
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
RUNS_DIR = os.path.join(EXPERIMENT_DIR, "runs")
BENCHMARK_RUNS = os.path.join(EXPERIMENT_DIR, "..", "benchmark", "runs")

# ---------------------------------------------------------------------------
# Ablation conditions
# ---------------------------------------------------------------------------
# Each condition is a dict with:
#   label: display name for the table
#   env: environment variable overrides applied during build + QA
#   reuse: if set, copy results from benchmark/<strategy> instead of re-running
#   needs_build: whether graph must be (re)built (False => query-only ablation)

ABLATION_CONDITIONS = {
    "C0": {
        "label": "Full MOSAIC",
        "reuse": "hybrid",
    },
    "C1": {
        "label": "Static graph (hash_only)",
        "reuse": "hash_only",
    },
    "C2": {
        "label": r"$-$ Relationship edges",
        "env": {
            "MOSAIC_EDGE_SEMANTIC_A": "0",
            "MOSAIC_EDGE_PREREQ_LLM": "0",
        },
        "needs_build": True,
    },
    "C3": {
        "label": r"$-$ Graph traversal",
        "env": {
            "MOSAIC_QUERY_NEIGHBOR_HOPS": "0",
        },
        "needs_build": False,
    },
    "C4": {
        "label": r"$-$ Association traversal (prereq only)",
        "env": {
            "MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS": "P",
        },
        "needs_build": False,
    },
    "C5": {
        "label": r"$-$ Prerequisite traversal (assoc only)",
        "env": {
            "MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS": "A",
        },
        "needs_build": False,
    },
    "C6": {
        "label": r"$-$ Edge enhancement + traversal",
        "env": {
            "MOSAIC_EDGE_SEMANTIC_A": "0",
            "MOSAIC_EDGE_PREREQ_LLM": "0",
            "MOSAIC_QUERY_NEIGHBOR_HOPS": "0",
        },
        "needs_build": True,
    },
}


# ---------------------------------------------------------------------------
# Reuse benchmark results for C0/C1
# ---------------------------------------------------------------------------

def _copy_benchmark_results(condition_id, strategy, sessions, runs_dir):
    """Copy QA summary files from benchmark/runs/<conv>/results/ for the
    given strategy.  Returns list of copied summary paths."""
    copied = []
    benchmark_runs = os.path.abspath(BENCHMARK_RUNS)
    for session in sessions:
        conv_id = session["conv_id"]
        qa_idx = conv_id.replace("conv", "")
        src = os.path.join(
            benchmark_runs, conv_id, "results",
            f"qa_{qa_idx}_eval_summary_{strategy}.json",
        )
        if not os.path.isfile(src):
            print(f"  [warn] benchmark result not found: {src}")
            continue
        dst_dir = os.path.join(runs_dir, condition_id, conv_id, "results")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"qa_{qa_idx}_eval_summary_{condition_id}.json")
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


# ---------------------------------------------------------------------------
# Build + QA for a single condition
# ---------------------------------------------------------------------------

def build_graph(conv_json_path, conv_name, run_dir, log_dir, env_overrides):
    """Build graph using mosaic CLI with environment overrides.
    Returns (graph_pkl, tags_json, elapsed_seconds)."""
    setup_mosaic_path()

    art_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(os.path.join(art_dir, "graph_snapshots"), exist_ok=True)

    graph_pkl = os.path.join(art_dir, f"graph_network_{conv_name}.pkl")
    tags_json = os.path.join(art_dir, f"{conv_name}_tags.json")

    cmd = [
        sys.executable, "-m", "mosaic", "build",
        "--conv-json", conv_json_path,
        "--conv-name", conv_name,
        "--graph-save-dir", os.path.join(art_dir, "graph_snapshots"),
        "--graph-out", graph_pkl,
        "--tags-out", tags_json,
        "--log-prompt", os.path.join(log_dir, "llm_io.jsonl"),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    env.update(env_overrides)

    print(f"  [build] {conv_name}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [build] FAILED (exit {result.returncode})")
        return None, None, elapsed

    print(f"  [build] done in {elapsed:.0f}s")
    return graph_pkl, tags_json, elapsed


def run_qa_eval(qa_json_path, graph_pkl, tags_json, conv_name, condition_id,
                results_dir, env_overrides, max_questions=None):
    """Run QA evaluation with environment overrides.
    Returns (summary_dict, elapsed_seconds)."""
    setup_mosaic_path()

    # Apply env overrides so mosaic query config picks them up
    old_env = {}
    for k, v in env_overrides.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    t0 = time.time()
    try:
        from query import process_single_qa

        qa_idx = conv_name.replace("conv", "")
        out_path = os.path.join(results_dir, f"qa_{qa_idx}_eval_full_{condition_id}.json")
        sum_path = os.path.join(results_dir, f"qa_{qa_idx}_eval_summary_{condition_id}.json")

        print(f"  [qa] {conv_name} condition={condition_id}")
        process_single_qa(
            qa_json_path, graph_pkl, tags_json,
            out_path, sum_path,
            max_questions=max_questions,
        )
        elapsed = time.time() - t0
        print(f"  [qa] done in {elapsed:.0f}s")
        summary = load_json_safe(sum_path)
        return summary, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [qa] FAILED: {e}")
        traceback.print_exc()
        return None, elapsed
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_condition(condition_id, runs_dir):
    """Aggregate QA summaries for a single ablation condition."""
    overall_correct = 0
    overall_total = 0
    by_category = {c: {"correct": 0, "total": 0} for c in CATEGORY_NAMES}

    cond_dir = os.path.join(runs_dir, condition_id)
    if not os.path.isdir(cond_dir):
        return None

    for conv_dir_name in sorted(os.listdir(cond_dir)):
        results_dir = os.path.join(cond_dir, conv_dir_name, "results")
        if not os.path.isdir(results_dir):
            continue
        for name in os.listdir(results_dir):
            if not (name.startswith("qa_") and "eval_summary" in name and name.endswith(".json")):
                continue
            data = load_json_safe(os.path.join(results_dir, name))
            summary = data.get("summary", {})
            overall_correct += summary.get("total_correct", 0)
            overall_total += summary.get("total_questions", 0)
            for cat, stats in summary.get("category_stats", {}).items():
                c = int(cat) if isinstance(cat, str) else cat
                if c in by_category:
                    by_category[c]["correct"] += stats.get("correct", 0)
                    by_category[c]["total"] += stats.get("total", 0)

    if overall_total == 0:
        return None

    overall_acc = overall_correct / overall_total * 100
    category_acc = {}
    for c, cat_name in CATEGORY_NAMES.items():
        tot = by_category[c]["total"]
        category_acc[cat_name] = (by_category[c]["correct"] / tot * 100) if tot else 0.0

    return {
        "accuracy_pct": round(overall_acc, 2),
        "correct": overall_correct,
        "total": overall_total,
        "by_category_pct": {k: round(v, 2) for k, v in category_acc.items()},
    }


def build_ablation_table(results_by_condition):
    """Build ablation table rows: [{id, label, overall, delta, ...}, ...]."""
    rows = []
    c0_acc = None
    for cid, cond in ABLATION_CONDITIONS.items():
        metrics = results_by_condition.get(cid)
        if metrics is None:
            continue
        acc = metrics["accuracy_pct"]
        if cid == "C0":
            c0_acc = acc
        delta = "---" if c0_acc is None or cid == "C0" else f"{acc - c0_acc:+.2f}"
        row = {
            "id": cid,
            "label": cond["label"],
            "overall": acc,
            "delta": delta,
        }
        row.update(metrics.get("by_category_pct", {}))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MOSAIC Ablation Experiment on LoCoMo")
    parser.add_argument("--condition", type=str, default=None,
                        help="Single condition ID (e.g., C3). Default: all.")
    parser.add_argument("--conv", type=str, default=None,
                        help="Single conversation ID (e.g., conv0)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip graph construction (reuse existing graphs)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of QA questions per conversation")
    args = parser.parse_args()

    sessions = load_sessions()
    if args.conv:
        sessions = [s for s in sessions if s["conv_id"] == args.conv]
        if not sessions:
            print(f"[error] Conversation '{args.conv}' not found")
            sys.exit(1)

    conditions = (
        {args.condition: ABLATION_CONDITIONS[args.condition]}
        if args.condition else ABLATION_CONDITIONS
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"{'='*60}")
    print(f"Ablation Experiment — {run_timestamp}")
    print(f"  conditions: {list(conditions.keys())}")
    print(f"  conversations: {[s['conv_id'] for s in sessions]}")
    print(f"{'='*60}\n")

    timing = {}
    t_start = time.time()

    for cid, cond in conditions.items():
        print(f"\n{'#'*60}")
        print(f"#  {cid}: {cond['label']}")
        print(f"{'#'*60}")

        cond_timing = {}

        # --- Reuse from benchmark ---
        if "reuse" in cond:
            strategy = cond["reuse"]
            copied = _copy_benchmark_results(cid, strategy, sessions, RUNS_DIR)
            print(f"  Reused {len(copied)} results from benchmark/{strategy}")
            timing[cid] = {"reused": True, "files": len(copied)}
            continue

        env_overrides = cond.get("env", {})
        needs_build = cond.get("needs_build", True)

        for session in sessions:
            conv_id = session["conv_id"]
            conv_json = os.path.join(DATASET_LOCOMO_DIR, session["conv_json"])
            qa_json = os.path.join(DATASET_LOCOMO_DIR, session["qa_json"])

            run_dir = os.path.join(RUNS_DIR, cid, conv_id)
            log_dir = os.path.join(run_dir, "log")
            results_dir = os.path.join(run_dir, "results")
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)

            label = f"{cid}_{conv_id}"

            # Determine graph paths
            art_dir = os.path.join(run_dir, "artifacts")
            graph_pkl = os.path.join(art_dir, f"graph_network_{conv_id}.pkl")
            tags_json = os.path.join(art_dir, f"{conv_id}_tags.json")

            # For query-only ablations, reuse benchmark hybrid graph
            if not needs_build or args.skip_build:
                benchmark_art = os.path.abspath(os.path.join(
                    BENCHMARK_RUNS, conv_id, "artifacts", "hybrid",
                ))
                graph_pkl = os.path.join(benchmark_art, f"graph_network_{conv_id}.pkl")
                tags_json = os.path.join(benchmark_art, f"{conv_id}_tags.json")
                if not os.path.isfile(graph_pkl):
                    print(f"  [skip] {label}: benchmark graph not found at {graph_pkl}")
                    continue
                print(f"  [reuse] graph from benchmark/hybrid")
            else:
                g, t, elapsed = build_graph(
                    conv_json, conv_id, run_dir, log_dir, env_overrides,
                )
                cond_timing[f"{conv_id}_build"] = round(elapsed, 1)
                if g is None:
                    print(f"  [skip] {label}: build failed")
                    continue
                graph_pkl, tags_json = g, t

            # QA evaluation
            summary, qa_time = run_qa_eval(
                qa_json, graph_pkl, tags_json,
                conv_id, cid, results_dir, env_overrides,
                max_questions=args.max_questions,
            )
            cond_timing[f"{conv_id}_qa"] = round(qa_time, 1)

        timing[cid] = cond_timing

    # --- Aggregate all conditions ---
    results_by_condition = {}
    for cid in conditions:
        metrics = aggregate_condition(cid, RUNS_DIR)
        if metrics:
            results_by_condition[cid] = metrics

    if results_by_condition:
        table = build_ablation_table(results_by_condition)
        save_json(os.path.join(RESULTS_DIR, "ablation_metrics.json"), results_by_condition)
        save_json(os.path.join(RESULTS_DIR, "ablation_table.json"), table)

        # Generate LaTeX table
        tex = _ablation_to_latex(table)
        tex_path = os.path.join(RESULTS_DIR, "tab_ablation_mosaic.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"\nLaTeX table written: {tex_path}")

    # --- Timing summary ---
    total_wall = time.time() - t_start
    timing["total_wall_s"] = round(total_wall, 1)
    save_json(os.path.join(RESULTS_DIR, "timing.json"), timing)

    print(f"\n{'='*60}")
    print(f"Ablation complete — {total_wall:.0f}s total")
    for cid in conditions:
        m = results_by_condition.get(cid)
        if m:
            print(f"  {cid}: {m['accuracy_pct']:.2f}%")
        else:
            print(f"  {cid}: no results")
    print(f"Results: {RESULTS_DIR}")
    return 0


def _ablation_to_latex(table_rows):
    """Generate a simple LaTeX tabular from ablation table rows."""
    lines = [
        r"\begin{tabular}{l l r r}",
        r"\toprule",
        r"ID & Condition & Overall (\%) & $\Delta$ \\",
        r"\midrule",
    ]
    for row in table_rows:
        cid = row["id"]
        label = row["label"]
        overall = f'{row["overall"]:.2f}'
        delta = row["delta"]
        lines.append(f"{cid} & {label} & {overall} & {delta} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main() or 0)
