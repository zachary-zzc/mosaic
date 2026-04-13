#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalability Experiment — Measure MOSAIC build time and QA accuracy as
conversation length increases.

Truncates existing LoCoMo conversations to varying message counts
(e.g. 50, 100, 200, 300, 400, 500, 600) and measures:
  - Graph construction wall time
  - Graph size (nodes, edges)
  - QA accuracy on the full question set

Usage (from project root):
  python experiments/locomo/scalability/run.py                     # all lengths, all conversations
  python experiments/locomo/scalability/run.py --conv conv0         # single conversation
  python experiments/locomo/scalability/run.py --lengths 100 200 400
  python experiments/locomo/scalability/run.py --max-questions 10   # limit QA questions
  python experiments/locomo/scalability/run.py --skip-qa            # build timing only
"""
import os
import sys
import json
import time
import pickle
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

CATEGORY_NAMES = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop"}
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
RUNS_DIR = os.path.join(EXPERIMENT_DIR, "runs")

DEFAULT_LENGTHS = [50, 100, 200, 300, 400, 500, 600]


def truncate_conversation(conv_json_path, max_messages, out_path):
    """Write a truncated copy of a LoCoMo conversation JSON.
    Returns the actual message count written.

    Handles the LoCoMo session-based format (dict with session_1, session_2, ...)
    as well as flat list and {conversation: [...]} formats.
    """
    import re as _re

    with open(conv_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --- flat list ---
    if isinstance(data, list):
        truncated = data[:max_messages]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(truncated, f, ensure_ascii=False)
        return len(truncated)

    # --- {conversation: [...]} wrapper ---
    if isinstance(data, dict) and "conversation" in data:
        truncated = data["conversation"][:max_messages]
        out_data = dict(data)
        out_data["conversation"] = truncated
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False)
        return len(truncated)

    # --- LoCoMo session-based format ---
    session_keys = sorted(
        [k for k in data if _re.match(r"session_\d+$", k)],
        key=lambda k: int(k.split("_")[1]),
    )
    total = 0
    out_data = {}
    for k, v in data.items():
        if _re.match(r"session_\d+$", k):
            continue  # handled below
        out_data[k] = v  # copy metadata (speaker_a, session_X_date_time, ...)

    for sk in session_keys:
        msgs = data[sk]
        remaining = max_messages - total
        if remaining <= 0:
            break
        kept = msgs[:remaining]
        out_data[sk] = kept
        # also copy associated date_time key
        dt_key = f"{sk}_date_time"
        if dt_key in data:
            out_data[dt_key] = data[dt_key]
        total += len(kept)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    return total


def build_graph(conv_json_path, conv_name, art_dir, log_dir):
    """Build graph via mosaic CLI. Returns (graph_pkl, tags_json, elapsed)."""
    setup_mosaic_path()

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

    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        return None, None, elapsed
    return graph_pkl, tags_json, elapsed


def count_graph_elements(graph_pkl_path):
    """Count nodes and edges in a pickled graph. Returns (nodes, edges)."""
    try:
        with open(graph_pkl_path, "rb") as f:
            G = pickle.load(f)
        return G.number_of_nodes(), G.number_of_edges()
    except Exception:
        return 0, 0


def run_qa_eval(qa_json_path, graph_pkl, tags_json, conv_name, length_label,
                results_dir, max_questions=None):
    """Run QA evaluation. Returns (summary_dict, elapsed)."""
    setup_mosaic_path()
    t0 = time.time()
    try:
        from query import process_single_qa

        qa_idx = conv_name.replace("conv", "")
        out_path = os.path.join(results_dir, f"qa_{qa_idx}_eval_full_{length_label}.json")
        sum_path = os.path.join(results_dir, f"qa_{qa_idx}_eval_summary_{length_label}.json")

        process_single_qa(
            qa_json_path, graph_pkl, tags_json,
            out_path, sum_path,
            max_questions=max_questions,
        )
        elapsed = time.time() - t0
        summary = load_json_safe(sum_path)
        return summary, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [qa] FAILED: {e}")
        traceback.print_exc()
        return None, elapsed


def main():
    parser = argparse.ArgumentParser(description="MOSAIC Scalability Experiment on LoCoMo")
    parser.add_argument("--conv", type=str, default=None,
                        help="Single conversation ID (e.g., conv0)")
    parser.add_argument("--lengths", type=int, nargs="+", default=None,
                        help=f"Message counts to test (default: {DEFAULT_LENGTHS})")
    parser.add_argument("--skip-qa", action="store_true",
                        help="Skip QA evaluation (measure build time and graph size only)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of QA questions per run")
    args = parser.parse_args()

    lengths = args.lengths or DEFAULT_LENGTHS
    sessions = load_sessions()
    if args.conv:
        sessions = [s for s in sessions if s["conv_id"] == args.conv]
        if not sessions:
            print(f"[error] Conversation '{args.conv}' not found")
            sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"{'='*60}")
    print(f"Scalability Experiment — {run_timestamp}")
    print(f"  conversations: {[s['conv_id'] for s in sessions]}")
    print(f"  lengths: {lengths}")
    print(f"  skip_qa: {args.skip_qa}")
    print(f"{'='*60}\n")

    # Collect per-run metrics: {conv_id: [{length, actual_msgs, build_s, nodes, edges, accuracy}, ...]}
    all_metrics = {}
    t_start = time.time()

    for session in sessions:
        conv_id = session["conv_id"]
        conv_json = os.path.join(DATASET_LOCOMO_DIR, session["conv_json"])
        qa_json = os.path.join(DATASET_LOCOMO_DIR, session["qa_json"])

        if not os.path.isfile(conv_json):
            print(f"[warn] conversation file not found: {conv_json}")
            continue

        conv_metrics = []

        for length in lengths:
            length_label = f"S{length}"
            label = f"{conv_id}_{length_label}"
            run_dir = os.path.join(RUNS_DIR, conv_id, length_label)
            log_dir = os.path.join(run_dir, "log")
            art_dir = os.path.join(run_dir, "artifacts")
            results_dir = os.path.join(run_dir, "results")
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)

            print(f"\n--- {label} ---")

            # Truncate conversation
            trunc_path = os.path.join(run_dir, f"{conv_id}_S{length}.json")
            actual_msgs = truncate_conversation(conv_json, length, trunc_path)
            print(f"  [truncate] {actual_msgs} messages (requested {length})")

            # Build graph
            graph_pkl, tags_json, build_time = build_graph(
                trunc_path, conv_id, art_dir, log_dir,
            )
            metric = {
                "length": length,
                "actual_msgs": actual_msgs,
                "build_s": round(build_time, 1),
            }

            if graph_pkl is None:
                print(f"  [build] FAILED in {build_time:.0f}s")
                conv_metrics.append(metric)
                continue
            print(f"  [build] done in {build_time:.0f}s")

            # Graph size
            nodes, edges = count_graph_elements(graph_pkl)
            metric["nodes"] = nodes
            metric["edges"] = edges
            print(f"  [graph] {nodes} nodes, {edges} edges")

            # QA evaluation
            if not args.skip_qa and os.path.isfile(qa_json):
                summary, qa_time = run_qa_eval(
                    qa_json, graph_pkl, tags_json,
                    conv_id, length_label, results_dir,
                    max_questions=args.max_questions,
                )
                metric["qa_s"] = round(qa_time, 1)
                if summary and "summary" in summary:
                    s = summary["summary"]
                    metric["accuracy_pct"] = round(
                        s.get("overall_accuracy", 0) * 100
                        if s.get("overall_accuracy", 0) <= 1
                        else s.get("overall_accuracy", 0),
                        2,
                    )
                    metric["total_questions"] = s.get("total_questions", 0)
                    metric["total_correct"] = s.get("total_correct", 0)
                    print(f"  [qa] accuracy={metric['accuracy_pct']:.2f}% in {qa_time:.0f}s")

            conv_metrics.append(metric)

        all_metrics[conv_id] = conv_metrics

    # --- Save results ---
    total_wall = time.time() - t_start
    output = {
        "timestamp": run_timestamp,
        "lengths": lengths,
        "total_wall_s": round(total_wall, 1),
        "by_conversation": all_metrics,
    }
    save_json(os.path.join(RESULTS_DIR, "scalability_metrics.json"), output)

    # --- Generate LaTeX table ---
    tex = _scalability_to_latex(all_metrics, lengths)
    tex_path = os.path.join(RESULTS_DIR, "tab_scalability.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"\nLaTeX table written: {tex_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Scalability complete — {total_wall:.0f}s total")
    for conv_id, metrics in all_metrics.items():
        print(f"  {conv_id}:")
        for m in metrics:
            acc_str = f"  acc={m.get('accuracy_pct', '?')}%" if "accuracy_pct" in m else ""
            print(f"    S{m['length']}: {m.get('actual_msgs', '?')} msgs, "
                  f"build={m['build_s']}s, "
                  f"nodes={m.get('nodes', '?')}, edges={m.get('edges', '?')}"
                  f"{acc_str}")
    print(f"Results: {RESULTS_DIR}")
    return 0


def _scalability_to_latex(all_metrics, lengths):
    """Generate LaTeX table: rows=lengths, columns=metrics averaged across conversations."""
    lines = [
        r"\begin{tabular}{r r r r r}",
        r"\toprule",
        r"Messages & Build (s) & $|V|$ & $|E|$ & Accuracy (\%) \\",
        r"\midrule",
    ]
    for length in lengths:
        build_times, nodes_list, edges_list, acc_list = [], [], [], []
        for conv_id, metrics in all_metrics.items():
            for m in metrics:
                if m["length"] == length:
                    build_times.append(m["build_s"])
                    if "nodes" in m:
                        nodes_list.append(m["nodes"])
                    if "edges" in m:
                        edges_list.append(m["edges"])
                    if "accuracy_pct" in m:
                        acc_list.append(m["accuracy_pct"])
        avg_build = sum(build_times) / len(build_times) if build_times else 0
        avg_nodes = sum(nodes_list) / len(nodes_list) if nodes_list else 0
        avg_edges = sum(edges_list) / len(edges_list) if edges_list else 0
        avg_acc = sum(acc_list) / len(acc_list) if acc_list else 0
        acc_str = f"{avg_acc:.1f}" if acc_list else "---"
        lines.append(
            f"{length} & {avg_build:.1f} & {avg_nodes:.0f} & {avg_edges:.0f} & {acc_str} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main() or 0)
