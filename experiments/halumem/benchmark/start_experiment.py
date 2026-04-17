#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HaluMem Benchmark — full pipeline orchestrator.

Phase 1: Build graphs for all users (sequential).
Phase 2: Query QA + LLM-judge evaluation (parallel, one thread per API key).
Phase 3: Aggregate evaluation scores via dataset/halumem/evaluation.py.

Usage (from project root):
  python experiments/halumem/benchmark/start_experiment.py \\
      --data test.jsonl \\
      --api-keys sk-aaa,sk-bbb,sk-ccc

  # Or read keys from a file (one per line):
  python experiments/halumem/benchmark/start_experiment.py \\
      --data test.jsonl --api-key-file keys.txt

  # Skip build phase (reuse existing graphs):
  python experiments/halumem/benchmark/start_experiment.py \\
      --data test.jsonl --api-keys sk-aaa --skip-build

  # Limit to N users (for quick testing):
  python experiments/halumem/benchmark/start_experiment.py \\
      --data test.jsonl --api-keys sk-aaa --max-users 2
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# ---------------------------------------------------------------------------
# Phase 1 — Build graphs
# ---------------------------------------------------------------------------
def phase_build(users: list[dict], run_dir: str) -> None:
    setup_mosaic_path()
    print(f"\n{'='*60}")
    print(f"Phase 1: Building graphs for {len(users)} user(s)")
    print(f"{'='*60}")
    t0 = time.time()

    tmp_dir = os.path.join(run_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    for idx, user_data in enumerate(users, 1):
        uuid = user_data.get("uuid", f"user_{idx}")
        print(f"[{idx}/{len(users)}] Building graph for {uuid} …")
        t_user = time.time()
        _memory, new_user_data = build_graph_for_user(user_data, run_dir)
        save_json(os.path.join(tmp_dir, f"{uuid}.json"), new_user_data)
        print(f"  done in {time.time() - t_user:.1f}s")

    print(f"\nPhase 1 complete in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Phase 2 — Parallel QA query + evaluation (one thread per API key)
# ---------------------------------------------------------------------------
def _qa_worker(
    user_json_path: str,
    original_user_data: dict,
    graph_pkl_path: str,
    api_key: str,
    results_dir: str,
) -> dict:
    """Per-user: load graph, search memory for each question, generate answer."""
    os.environ["DASHSCOPE_API_KEY"] = api_key

    setup_mosaic_path()
    from src.assist import llm_request_for_json
    from src.prompts_en import PROMPT_QUERY_TEMPLATE_EVAL

    with open(user_json_path, "r", encoding="utf-8") as f:
        new_user_data = json.load(f)

    uuid = new_user_data["uuid"]

    # Load the full ClassGraph pickle
    with open(graph_pkl_path, "rb") as f:
        memory = pickle.load(f)

    # If it's a bare nx graph (legacy), wrap
    from src.data.graph import ClassGraph
    if not isinstance(memory, ClassGraph):
        cg = ClassGraph()
        cg.graph = memory
        memory = cg
    elif hasattr(memory, "_rebuild_dual_nx_from_edges"):
        memory._rebuild_dual_nx_from_edges()

    # QA query — questions come from the original data
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

        session_data["questions"] = new_questions

    out_path = os.path.join(results_dir, f"{uuid}_eval.json")
    save_json(out_path, new_user_data)
    return {"uuid": uuid, "status": "ok", "path": out_path}


def phase_qa(users: list[dict], run_dir: str, api_keys: list[str]) -> list[str]:
    """Dispatch QA across threads. Returns list of result file paths."""
    tmp_dir = os.path.join(run_dir, "tmp")
    graph_dir = os.path.join(run_dir, "graphs")
    results_dir = os.path.join(run_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase 2: QA query + eval — {len(users)} user(s), {len(api_keys)} thread(s)")
    print(f"{'='*60}")

    # Collect valid tasks
    tasks = []
    for user_data in users:
        uuid = user_data["uuid"]
        user_json = os.path.join(tmp_dir, f"{uuid}.json")
        if not os.path.isfile(user_json):
            print(f"  [skip] {uuid}: build artifact missing")
            continue
        uname = extract_user_name(user_data["persona_info"])
        graph_pkl = os.path.join(graph_dir, f"{uname}_graph.pkl")
        if not os.path.isfile(graph_pkl):
            print(f"  [skip] {uuid}: graph pickle missing")
            continue
        tasks.append((user_json, user_data, graph_pkl))

    if not tasks:
        print("[error] No valid tasks — run build phase first")
        return []

    n_threads = len(api_keys)
    result_paths: list[str] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {}
        for i, (user_json, user_data, graph_pkl) in enumerate(tasks):
            key = api_keys[i % n_threads]
            fut = pool.submit(
                _qa_worker, user_json, user_data, graph_pkl, key, results_dir
            )
            futures[fut] = user_data["uuid"]

        for i, fut in enumerate(as_completed(futures), 1):
            uid = futures[fut]
            try:
                result = fut.result()
                print(f"  [{i}/{len(tasks)}] {uid}: {result['status']}")
                result_paths.append(result["path"])
            except Exception as e:
                print(f"  [{i}/{len(tasks)}] {uid}: FAILED — {e}")
                traceback.print_exc()

    print(f"\nPhase 2 complete in {time.time() - t0:.1f}s  ({len(result_paths)}/{len(tasks)} succeeded)")
    return result_paths


# ---------------------------------------------------------------------------
# Phase 3 — Merge + aggregate evaluation
# ---------------------------------------------------------------------------
def phase_aggregate(result_paths: list[str], run_dir: str) -> None:
    """Merge per-user eval JSONs → combined JSONL, then run scoring."""
    if not result_paths:
        print("\n[skip] Phase 3: no results to aggregate")
        return

    print(f"\n{'='*60}")
    print(f"Phase 3: Aggregating {len(result_paths)} user result(s)")
    print(f"{'='*60}")

    combined_jsonl = os.path.join(run_dir, "halumem_eval_results.jsonl")
    with open(combined_jsonl, "w", encoding="utf-8") as out:
        for path in sorted(result_paths):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"  Combined JSONL: {combined_jsonl}")

    # Run evaluation scoring (from dataset/halumem/evaluation.py)
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "dataset", "halumem"))
        from evaluation import process_user, aggregate_eval_results

        eval_tmp_dir = os.path.join(run_dir, "eval_tmp")
        os.makedirs(eval_tmp_dir, exist_ok=True)

        eval_results = {
            "overall_score": {
                "memory_integrity": {},
                "memory_accuracy": {},
                "memory_extraction_f1": 0,
                "memory_update": {},
                "question_answering": {},
                "memory_type_accuracy": {
                    "Event Memory": {"memory_integrity_acc": 0, "memory_update_acc": 0, "total_num": 0},
                    "Persona Memory": {"memory_integrity_acc": 0, "memory_update_acc": 0, "total_num": 0},
                    "Relationship Memory": {"memory_integrity_acc": 0, "memory_update_acc": 0, "total_num": 0},
                },
            },
            "memory_integrity_records": [],
            "memory_accuracy_records": [],
            "memory_update_records": [],
            "question_answering_records": [],
        }

        with open(combined_jsonl, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                user_data = json.loads(line.strip())
                uuid = user_data.get("uuid", f"user_{idx}")
                tmp_file = os.path.join(eval_tmp_dir, f"{uuid}.json")

                if os.path.exists(tmp_file):
                    print(f"  [{idx}] {uuid}: cached")
                    with open(tmp_file, "r", encoding="utf-8") as tf:
                        user_result = json.load(tf)
                else:
                    print(f"  [{idx}] {uuid}: evaluating …")
                    user_result = process_user(idx, user_data, 1)
                    save_json(tmp_file, user_result)

                eval_results["memory_integrity_records"].extend(user_result.get("memory_integrity_records", []))
                eval_results["memory_accuracy_records"].extend(user_result.get("memory_accuracy_records", []))
                eval_results["memory_update_records"].extend(user_result.get("memory_update_records", []))
                eval_results["question_answering_records"].extend(user_result.get("question_answering_records", []))

        eval_results = aggregate_eval_results(eval_results)

        score_path = os.path.join(run_dir, "halumem_eval_scores.json")
        save_json(score_path, eval_results)
        print(f"\n  Scores: {score_path}")

        # Print key metrics
        overall = eval_results.get("overall_score", {})
        mi = overall.get("memory_integrity", {})
        ma = overall.get("memory_accuracy", {})
        qa = overall.get("question_answering", {})
        print(f"\n  Memory Integrity recall(valid):  {mi.get('recall(valid)', 'N/A')}")
        print(f"  Memory Accuracy  target(valid):  {ma.get('target_accuracy(valid)', 'N/A')}")
        print(f"  Memory F1:                       {overall.get('memory_extraction_f1', 'N/A')}")
        print(f"  QA correct(valid):               {qa.get('correct_qa_ratio(valid)', 'N/A')}")

    except Exception as e:
        print(f"\n  [warn] Evaluation scoring failed: {e}")
        traceback.print_exc()
        print("  You can run scoring manually with dataset/halumem/evaluation.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="HaluMem Benchmark — build → QA → eval pipeline"
    )
    parser.add_argument("--data", default="test.jsonl",
                        help="JSONL filename inside dataset/halumem/data/")
    parser.add_argument("--run-id", default="default",
                        help="Run identifier (subdirectory under runs/)")
    parser.add_argument("--api-keys", type=str, default=None,
                        help="Comma-separated API keys for parallel QA")
    parser.add_argument("--api-key-file", type=str, default=None,
                        help="File with one API key per line")
    parser.add_argument("--max-users", type=int, default=None,
                        help="Limit number of users processed")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip Phase 1 (reuse existing graphs)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip Phase 3 (no LLM-judge scoring)")
    args = parser.parse_args()

    api_keys = _read_api_keys(args)
    if not api_keys:
        print("[error] No API keys — provide --api-keys, --api-key-file, or set DASHSCOPE_API_KEY")
        return 1

    data_path = os.path.join(DATASET_HALUMEM_DIR, args.data)
    if not os.path.isfile(data_path):
        print(f"[error] Data file not found: {data_path}")
        return 1

    users = load_jsonl(data_path)
    if args.max_users:
        users = users[: args.max_users]

    run_dir = os.path.join(RUNS_DIR, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"HaluMem Benchmark — {len(users)} user(s), {len(api_keys)} API key(s)")
    print(f"  data: {data_path}")
    print(f"  run_dir: {run_dir}")
    t_start = time.time()

    # Phase 1
    if not args.skip_build:
        phase_build(users, run_dir)
    else:
        print("\n[skip] Phase 1: --skip-build")

    # Phase 2
    result_paths = phase_qa(users, run_dir, api_keys)

    # Phase 3
    if not args.skip_eval:
        phase_aggregate(result_paths, run_dir)
    else:
        print("\n[skip] Phase 3: --skip-eval")

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All done in {total:.1f}s")
    print(f"  Results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
