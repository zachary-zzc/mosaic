#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HaluMem Scalability Experiment — measure how MOSAIC handles increasing
numbers of sessions per user.

For each user, progressively includes more sessions (e.g. 1, 2, 4, 8, ...
up to all sessions) and measures:
  - Graph construction wall time (build_duration_s)
  - Graph size (graph_pkl_bytes)
  - QA accuracy at each scale point (if --run-qa)

Usage:
  python experiments/halumem/scalability/start_experiment.py \\
      --data test.jsonl --api-keys sk-aaa

  # Custom session counts:
  python experiments/halumem/scalability/start_experiment.py \\
      --data test.jsonl --session-counts 1 2 4 8 --api-keys sk-aaa

  # Build timing only (skip QA):
  python experiments/halumem/scalability/start_experiment.py \\
      --data test.jsonl --skip-qa --api-keys sk-aaa

  # Single user for quick test:
  python experiments/halumem/scalability/start_experiment.py \\
      --data test.jsonl --max-users 1 --api-keys sk-aaa
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import time
import traceback
from string import Template

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.halumem._utils import (
    DATASET_HALUMEM_DIR,
    build_graph_for_user,
    extract_user_name,
    load_jsonl,
    save_json,
    setup_mosaic_path,
    _search_memory,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(EXPERIMENT_DIR, "runs")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")

DEFAULT_SESSION_COUNTS = [1, 2, 4, 8, 16, 32]


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------
def _read_api_keys(args) -> list[str]:
    keys: list[str] = []
    if args.api_keys:
        keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    if args.api_key_file and os.path.isfile(args.api_key_file):
        with open(args.api_key_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    keys.append(line)
    if not keys:
        env_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if env_key:
            keys.append(env_key)
    return keys


def truncate_user_sessions(user_data: dict, max_sessions: int) -> dict:
    """Return a copy of *user_data* with at most *max_sessions* sessions."""
    truncated = copy.deepcopy(user_data)
    truncated["sessions"] = truncated["sessions"][:max_sessions]
    return truncated


def run_qa_for_user(
    memory,
    new_user_data: dict,
    original_user_data: dict,
    api_key: str,
) -> dict:
    """Run QA queries for one user (single-threaded)."""
    os.environ["DASHSCOPE_API_KEY"] = api_key

    setup_mosaic_path()
    from src.assist import llm_request_for_json
    from src.prompts_en import PROMPT_QUERY_TEMPLATE_EVAL

    qa_count = 0
    correct_count = 0  # placeholder — actual scoring done later

    for session_data, orig_session in zip(
        new_user_data["sessions"], original_user_data["sessions"]
    ):
        if session_data.get("is_generated_qa_session"):
            continue
        orig_questions = orig_session.get("questions", [])
        if not orig_questions:
            continue

        new_questions = []
        for qa in orig_questions:
            context_str, dur_ms = _search_memory(memory, qa["question"])
            new_qa = copy.deepcopy(qa)
            new_qa["context"] = context_str
            new_qa["search_duration_ms"] = dur_ms

            prompt = Template(PROMPT_QUERY_TEMPLATE_EVAL).substitute(
                INFORMATION=context_str, QUESTION=qa["question"]
            )
            t0 = time.time()
            try:
                resp = llm_request_for_json(prompt)
                answer = resp.get("response", "")
            except Exception:
                answer = ""
            new_qa["system_response"] = answer
            new_qa["response_duration_ms"] = (time.time() - t0) * 1000
            new_questions.append(new_qa)
            qa_count += 1

        session_data["questions"] = new_questions

    return {"qa_count": qa_count}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="HaluMem Scalability Experiment")
    parser.add_argument("--data", default="test.jsonl",
                        help="JSONL filename inside dataset/halumem/data/")
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--session-counts", type=int, nargs="+", default=None,
                        help=f"Session counts to test (default: {DEFAULT_SESSION_COUNTS})")
    parser.add_argument("--skip-qa", action="store_true",
                        help="Skip QA evaluation (measure build time only)")
    parser.add_argument("--api-keys", type=str, default=None)
    parser.add_argument("--api-key-file", type=str, default=None)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()

    api_keys = _read_api_keys(args)
    if not api_keys and not args.skip_qa:
        print("[error] No API keys provided (needed for QA). Use --skip-qa to skip QA.")
        return 1

    data_path = os.path.join(DATASET_HALUMEM_DIR, args.data)
    if not os.path.isfile(data_path):
        print(f"[error] Data file not found: {data_path}")
        return 1

    users = load_jsonl(data_path)
    if args.max_users:
        users = users[: args.max_users]

    session_counts = args.session_counts or DEFAULT_SESSION_COUNTS

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f"HaluMem Scalability — {len(users)} user(s), session counts: {session_counts}")
    t_start = time.time()

    # metrics[uuid][session_count] = {...}
    all_metrics: list[dict] = []

    for user_idx, user_data in enumerate(users, 1):
        uuid = user_data["uuid"]
        total_sessions = len(user_data["sessions"])
        uname = extract_user_name(user_data["persona_info"])

        print(f"\n{'='*60}")
        print(f"User [{user_idx}/{len(users)}]: {uname} ({uuid}), {total_sessions} session(s)")
        print(f"{'='*60}")

        for sc in session_counts:
            if sc > total_sessions:
                print(f"  [skip] S{sc}: only {total_sessions} sessions available")
                continue

            label = f"S{sc}"
            run_dir = os.path.join(RUNS_DIR, args.run_id, uuid, label)
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n  --- {label} ({sc} sessions) ---")

            # Truncate
            truncated = truncate_user_sessions(user_data, sc)

            # Build
            setup_mosaic_path()
            t_build = time.time()
            memory, new_user_data = build_graph_for_user(truncated, run_dir)
            build_s = time.time() - t_build

            # Graph size
            graph_pkl = os.path.join(run_dir, "graphs", f"{uname}_graph.pkl")
            pkl_bytes = os.path.getsize(graph_pkl) if os.path.isfile(graph_pkl) else 0

            metric = {
                "uuid": uuid,
                "user_name": uname,
                "session_count": sc,
                "total_sessions": total_sessions,
                "build_duration_s": round(build_s, 2),
                "graph_pkl_bytes": pkl_bytes,
            }

            print(f"    build: {build_s:.1f}s, graph: {pkl_bytes/1024:.1f}KB")

            # Save build result
            tmp_dir = os.path.join(run_dir, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            save_json(os.path.join(tmp_dir, f"{uuid}.json"), new_user_data)

            # QA (optional)
            if not args.skip_qa and api_keys:
                api_key = api_keys[user_idx % len(api_keys)]
                t_qa = time.time()
                qa_stats = run_qa_for_user(memory, new_user_data, truncated, api_key)
                qa_s = time.time() - t_qa
                metric["qa_duration_s"] = round(qa_s, 2)
                metric["qa_count"] = qa_stats["qa_count"]
                print(f"    qa: {qa_s:.1f}s, {qa_stats['qa_count']} questions")

                # Save eval result
                results_dir = os.path.join(run_dir, "eval_results")
                os.makedirs(results_dir, exist_ok=True)
                save_json(os.path.join(results_dir, f"{uuid}_eval.json"), new_user_data)

            all_metrics.append(metric)

    # --- Summary ---
    summary_path = os.path.join(RESULTS_DIR, f"scalability_{args.run_id}.json")
    save_json(summary_path, {
        "run_id": args.run_id,
        "session_counts": session_counts,
        "total_users": len(users),
        "metrics": all_metrics,
    })

    total_s = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Scalability experiment done in {total_s:.1f}s")
    print(f"  Summary: {summary_path}")

    # Print table
    print(f"\n{'Sessions':>10} {'Build(s)':>10} {'Graph(KB)':>10}", end="")
    if not args.skip_qa:
        print(f" {'QA(s)':>10} {'Qs':>6}", end="")
    print()
    for m in all_metrics:
        print(f"  {m['session_count']:>8} {m['build_duration_s']:>10.1f} {m['graph_pkl_bytes']/1024:>10.1f}", end="")
        if "qa_duration_s" in m:
            print(f" {m['qa_duration_s']:>10.1f} {m['qa_count']:>6}", end="")
        print(f"  ({m.get('user_name', m['uuid'])})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
