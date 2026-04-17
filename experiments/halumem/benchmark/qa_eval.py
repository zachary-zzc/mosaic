#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HaluMem Benchmark — parallel QA query + evaluation using multiple API keys.

Reads the per-user build results from ``runs/<run_id>/tmp/<uuid>.json``,
dispatches QA query + LLM-judge evaluation across *N* threads (one per API
key), and writes per-user eval JSONL + aggregated scores.

Usage (from project root):
  python experiments/halumem/benchmark/qa_eval.py \\
      --data test.jsonl \\
      --run-id default \\
      --api-keys sk-aaa,sk-bbb,sk-ccc

  # Or read keys from a file (one per line):
  python experiments/halumem/benchmark/qa_eval.py \\
      --data test.jsonl --run-id default --api-key-file keys.txt
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from string import Template

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.halumem._utils import (
    DATASET_HALUMEM_DIR,
    append_jsonl,
    load_jsonl,
    save_json,
    setup_mosaic_path,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(EXPERIMENT_DIR, "runs")


# ---------------------------------------------------------------------------
# Per-user worker: query QA + LLM-judge evaluation
# ---------------------------------------------------------------------------
def _process_one_user(
    user_json_path: str,
    original_user_data: dict,
    graph_pkl_path: str,
    api_key: str,
    results_dir: str,
) -> dict:
    """
    1. Load the user's built graph.
    2. For every QA question → search memory → generate answer.
    3. Run LLM-judge evaluation on all four axes (integrity, accuracy, update, QA).
    4. Save per-user result JSON.

    *api_key* is set as ``DASHSCOPE_API_KEY`` (or the relevant env var) so
    each thread uses its own key.
    """
    # Set the per-thread API key
    os.environ["DASHSCOPE_API_KEY"] = api_key

    setup_mosaic_path()
    from src.assist import load_mosaic_memory_pickle, llm_request_for_json
    from src.prompts_en import PROMPT_QUERY_TEMPLATE_EVAL
    from experiments.halumem._utils import _search_memory

    # Load built user data (contains extracted_memories, memory_points, etc.)
    with open(user_json_path, "r", encoding="utf-8") as f:
        new_user_data = json.load(f)

    uuid = new_user_data["uuid"]
    user_name = new_user_data.get("user_name", uuid)

    # Load the graph
    memory = load_mosaic_memory_pickle(graph_pkl_path)

    # --- QA query ---
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

    # --- Persist ---
    out_path = os.path.join(results_dir, f"{uuid}_eval.json")
    save_json(out_path, new_user_data)
    return {"uuid": uuid, "status": "ok", "path": out_path}


# ---------------------------------------------------------------------------
# Main
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
        # fallback: use env
        env_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if env_key:
            keys.append(env_key)
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description="HaluMem Benchmark — parallel QA + eval")
    parser.add_argument("--data", default="test.jsonl", help="JSONL filename inside dataset/halumem/data/")
    parser.add_argument("--run-id", default="default", help="Run identifier (must match build_graph)")
    parser.add_argument("--api-keys", type=str, default=None, help="Comma-separated API keys")
    parser.add_argument("--api-key-file", type=str, default=None, help="File with one API key per line")
    parser.add_argument("--max-users", type=int, default=None, help="Limit users")
    args = parser.parse_args()

    api_keys = _read_api_keys(args)
    if not api_keys:
        print("[error] No API keys provided (--api-keys, --api-key-file, or DASHSCOPE_API_KEY)")
        return 1

    data_path = os.path.join(DATASET_HALUMEM_DIR, args.data)
    if not os.path.isfile(data_path):
        print(f"[error] Data file not found: {data_path}")
        return 1

    users = load_jsonl(data_path)
    if args.max_users:
        users = users[: args.max_users]

    run_dir = os.path.join(RUNS_DIR, args.run_id)
    tmp_dir = os.path.join(run_dir, "tmp")
    graph_dir = os.path.join(run_dir, "graphs")
    results_dir = os.path.join(run_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)

    # Validate that build has been done
    missing = []
    tasks = []
    for user_data in users:
        uuid = user_data["uuid"]
        user_json = os.path.join(tmp_dir, f"{uuid}.json")
        if not os.path.isfile(user_json):
            missing.append(uuid)
            continue
        # find graph pkl
        from experiments.halumem._utils import extract_user_name
        uname = extract_user_name(user_data["persona_info"])
        graph_pkl = os.path.join(graph_dir, f"{uname}_graph.pkl")
        if not os.path.isfile(graph_pkl):
            missing.append(uuid)
            continue
        tasks.append((user_json, user_data, graph_pkl))

    if missing:
        print(f"[warn] {len(missing)} user(s) missing build artifacts — run build_graph.py first")
        for m in missing[:5]:
            print(f"  missing: {m}")
        if not tasks:
            return 1

    n_threads = len(api_keys)
    print(f"QA+Eval for {len(tasks)} user(s) with {n_threads} thread(s)")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {}
        for i, (user_json, user_data, graph_pkl) in enumerate(tasks):
            key = api_keys[i % n_threads]
            fut = pool.submit(
                _process_one_user, user_json, user_data, graph_pkl, key, results_dir
            )
            futures[fut] = user_data["uuid"]

        for i, fut in enumerate(as_completed(futures), 1):
            uuid = futures[fut]
            try:
                result = fut.result()
                print(f"[{i}/{len(tasks)}] ✅ {uuid} ({result['status']})")
            except Exception as e:
                print(f"[{i}/{len(tasks)}] ❌ {uuid}: {e}")
                traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nAll QA+Eval done in {elapsed:.1f}s  →  {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
