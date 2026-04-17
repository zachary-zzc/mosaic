#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HaluMem Ablation Experiment — disable individual MOSAIC components and
measure impact on memory extraction and QA.

Ablation conditions:
  C0: Full MOSAIC (reuse benchmark results)
  C1: hash_only build (benchmark default — reuse)
  C2: No relationship edges  (MOSAIC_EDGE_SEMANTIC_A=0, MOSAIC_EDGE_PREREQ_LLM=0)
  C3: No graph traversal      (MOSAIC_QUERY_NEIGHBOR_HOPS=0)
  C4: Prereq-only traversal   (MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS=P)
  C5: Assoc-only traversal    (MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS=A)
  C6: No edges + no traversal (C2 + C3)

For C0/C1 the script copies results from benchmark/runs/<run_id>/.
For C2–C6 it rebuilds the graph (if needed) and re-runs QA + eval with
the specified environment overrides.

Usage:
  python experiments/halumem/ablation/start_experiment.py \\
      --data test.jsonl --benchmark-run-id default \\
      --api-keys sk-aaa,sk-bbb

  python experiments/halumem/ablation/start_experiment.py \\
      --data test.jsonl --condition C3 --api-keys sk-aaa
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import shutil
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
BENCHMARK_RUNS = os.path.join(EXPERIMENT_DIR, "..", "benchmark", "runs")

# ---------------------------------------------------------------------------
# Ablation conditions
# ---------------------------------------------------------------------------
ABLATION_CONDITIONS = {
    "C0": {
        "label": "Full MOSAIC (benchmark)",
        "reuse": True,
    },
    "C1": {
        "label": "hash_only baseline (benchmark)",
        "reuse": True,
    },
    "C2": {
        "label": "- Relationship edges",
        "env": {
            "MOSAIC_EDGE_SEMANTIC_A": "0",
            "MOSAIC_EDGE_PREREQ_LLM": "0",
        },
        "needs_build": True,
    },
    "C3": {
        "label": "- Graph traversal",
        "env": {
            "MOSAIC_QUERY_NEIGHBOR_HOPS": "0",
        },
        "needs_build": False,  # query-only ablation
    },
    "C4": {
        "label": "- Association traversal (prereq only)",
        "env": {
            "MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS": "P",
        },
        "needs_build": False,
    },
    "C5": {
        "label": "- Prerequisite traversal (assoc only)",
        "env": {
            "MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS": "A",
        },
        "needs_build": False,
    },
    "C6": {
        "label": "- Edge enhancement + traversal",
        "env": {
            "MOSAIC_EDGE_SEMANTIC_A": "0",
            "MOSAIC_EDGE_PREREQ_LLM": "0",
            "MOSAIC_QUERY_NEIGHBOR_HOPS": "0",
        },
        "needs_build": True,
    },
}


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
# Reuse benchmark results for C0/C1
# ---------------------------------------------------------------------------
def _reuse_benchmark(cond_id: str, benchmark_run_dir: str, cond_run_dir: str) -> int:
    """Copy eval_results/ and scores from benchmark. Returns count of files copied."""
    src_dir = os.path.join(benchmark_run_dir, "eval_results")
    if not os.path.isdir(src_dir):
        print(f"  [warn] benchmark eval_results not found: {src_dir}")
        return 0
    dst_dir = os.path.join(cond_run_dir, "eval_results")
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for f in os.listdir(src_dir):
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        count += 1
    # Also copy aggregate scores if present
    for fname in ("halumem_eval_scores.json", "halumem_eval_results.jsonl"):
        src = os.path.join(benchmark_run_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(cond_run_dir, fname))
    return count


# ---------------------------------------------------------------------------
# Build + QA for one condition
# ---------------------------------------------------------------------------
def _qa_worker_ablation(
    user_json_path: str,
    original_user_data: dict,
    graph_pkl_path: str,
    api_key: str,
    results_dir: str,
    env_overrides: dict,
) -> dict:
    """QA worker with environment overrides for ablation."""
    os.environ["DASHSCOPE_API_KEY"] = api_key
    for k, v in env_overrides.items():
        os.environ[k] = v

    setup_mosaic_path()
    from src.assist import llm_request_for_json
    from src.prompts_en import PROMPT_QUERY_TEMPLATE_EVAL

    with open(user_json_path, "r", encoding="utf-8") as f:
        new_user_data = json.load(f)

    uuid = new_user_data["uuid"]

    with open(graph_pkl_path, "rb") as f:
        memory = pickle.load(f)

    from src.data.graph import ClassGraph
    if not isinstance(memory, ClassGraph):
        cg = ClassGraph()
        cg.graph = memory
        memory = cg
    elif hasattr(memory, "_rebuild_dual_nx_from_edges"):
        memory._rebuild_dual_nx_from_edges()

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


def run_condition(
    cond_id: str,
    cond: dict,
    users: list[dict],
    api_keys: list[str],
    benchmark_run_dir: str,
    run_id: str,
) -> None:
    cond_run_dir = os.path.join(RUNS_DIR, run_id, cond_id)
    os.makedirs(cond_run_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"#  {cond_id}: {cond['label']}")
    print(f"{'#'*60}")

    # --- Reuse ---
    if cond.get("reuse"):
        n = _reuse_benchmark(cond_id, benchmark_run_dir, cond_run_dir)
        print(f"  Reused {n} file(s) from benchmark")
        return

    env_overrides = cond.get("env", {})
    needs_build = cond.get("needs_build", True)

    graph_dir = os.path.join(cond_run_dir, "graphs")
    tmp_dir = os.path.join(cond_run_dir, "tmp")
    results_dir = os.path.join(cond_run_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)

    # --- Build (if needed) ---
    if needs_build:
        print(f"  Rebuilding graphs with env overrides: {env_overrides}")
        old_env = {}
        for k, v in env_overrides.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = v

        for idx, user_data in enumerate(users, 1):
            uuid = user_data["uuid"]
            print(f"  [{idx}/{len(users)}] Building {uuid} …")
            _memory, new_user_data = build_graph_for_user(user_data, cond_run_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            save_json(os.path.join(tmp_dir, f"{uuid}.json"), new_user_data)

        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    else:
        # Query-only: reuse benchmark graphs + tmp
        bm_graph = os.path.join(benchmark_run_dir, "graphs")
        bm_tmp = os.path.join(benchmark_run_dir, "tmp")
        if os.path.isdir(bm_graph):
            os.makedirs(graph_dir, exist_ok=True)
            for f in os.listdir(bm_graph):
                src, dst = os.path.join(bm_graph, f), os.path.join(graph_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        if os.path.isdir(bm_tmp):
            os.makedirs(tmp_dir, exist_ok=True)
            for f in os.listdir(bm_tmp):
                src, dst = os.path.join(bm_tmp, f), os.path.join(tmp_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        print(f"  Reusing benchmark graphs (query-only ablation)")

    # --- QA ---
    tasks = []
    for user_data in users:
        uuid = user_data["uuid"]
        user_json = os.path.join(tmp_dir, f"{uuid}.json")
        if not os.path.isfile(user_json):
            continue
        uname = extract_user_name(user_data["persona_info"])
        graph_pkl = os.path.join(graph_dir, f"{uname}_graph.pkl")
        if not os.path.isfile(graph_pkl):
            continue
        tasks.append((user_json, user_data, graph_pkl))

    if not tasks:
        print(f"  [warn] No valid tasks for {cond_id}")
        return

    n_threads = len(api_keys)
    print(f"  QA eval: {len(tasks)} user(s), {n_threads} thread(s)")

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {}
        for i, (uj, ud, gp) in enumerate(tasks):
            key = api_keys[i % n_threads]
            fut = pool.submit(
                _qa_worker_ablation, uj, ud, gp, key, results_dir, env_overrides
            )
            futures[fut] = ud["uuid"]

        for i, fut in enumerate(as_completed(futures), 1):
            uid = futures[fut]
            try:
                r = fut.result()
                print(f"    [{i}/{len(tasks)}] {uid}: {r['status']}")
            except Exception as e:
                print(f"    [{i}/{len(tasks)}] {uid}: FAILED — {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="HaluMem Ablation Experiment")
    parser.add_argument("--data", default="test.jsonl",
                        help="JSONL filename inside dataset/halumem/data/")
    parser.add_argument("--run-id", default="default",
                        help="Run identifier")
    parser.add_argument("--benchmark-run-id", default="default",
                        help="Benchmark run ID to reuse for C0/C1 and query-only conditions")
    parser.add_argument("--condition", type=str, default=None,
                        help="Single condition (e.g. C3). Default: all")
    parser.add_argument("--api-keys", type=str, default=None)
    parser.add_argument("--api-key-file", type=str, default=None)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()

    api_keys = _read_api_keys(args)
    if not api_keys:
        print("[error] No API keys provided")
        return 1

    data_path = os.path.join(DATASET_HALUMEM_DIR, args.data)
    if not os.path.isfile(data_path):
        print(f"[error] Data file not found: {data_path}")
        return 1

    users = load_jsonl(data_path)
    if args.max_users:
        users = users[: args.max_users]

    benchmark_run_dir = os.path.abspath(
        os.path.join(BENCHMARK_RUNS, args.benchmark_run_id)
    )

    conditions = (
        {args.condition: ABLATION_CONDITIONS[args.condition]}
        if args.condition
        else ABLATION_CONDITIONS
    )

    print(f"HaluMem Ablation — {len(users)} user(s), conditions: {list(conditions.keys())}")
    t0 = time.time()

    for cid, cond in conditions.items():
        run_condition(cid, cond, users, api_keys, benchmark_run_dir, args.run_id)

    print(f"\nAll conditions done in {time.time() - t0:.1f}s")
    print(f"  Results: {os.path.join(RUNS_DIR, args.run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
