#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoCoMo benchmark experiment pipeline — cross-platform replacement for start_experiment.sh.

For each conversation, runs graph construction + full QA evaluation with two strategies:
  1. hybrid     — TF-IDF + LLM joint graph construction
  2. hash_only  — TF-IDF/hash baseline graph construction

Uses ``python -m mosaic build`` and ``python -m mosaic query`` directly.

Directory layout (per conversation):
  runs/<conv_id>/artifacts/{hybrid,hash_only}/  — graph pkl, tags, snapshots
  runs/<conv_id>/results/                        — QA eval full & summary JSONs
  runs/<conv_id>/log/                            — telemetry, llm_io, progress

Usage:
  python experiments/locomo/benchmark/start_experiment.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main() -> int:
    repo = _repo_root()
    locomo_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(repo, "dataset", "locomo")
    sessions_json = os.path.join(dataset_dir, "experiment_sessions.json")

    if not os.path.isfile(sessions_json):
        print(f"[error] {sessions_json} not found.")
        return 1

    with open(sessions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions = data["sessions"]
    strategies = ["hybrid", "hash_only"]
    runs_dir = os.path.join(locomo_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    env = {**os.environ, "PYTHONPATH": repo}

    print(f"[check] strategies: {' '.join(strategies)}")
    print(f"[check] conversations: {' '.join(s['conv_id'] for s in sessions)}")
    print(f"[check] repo root: {repo}")
    print(f"[check] python: {sys.executable}")
    print()

    # Validate all input files exist
    for s in sessions:
        conv_json = os.path.join(dataset_dir, s["conv_json"])
        qa_json = os.path.join(dataset_dir, s["qa_json"])
        if not os.path.isfile(conv_json):
            print(f"[error] conversation file not found: {conv_json}")
            return 1
        if not os.path.isfile(qa_json):
            print(f"[error] QA file not found: {qa_json}")
            return 1
        print(f"[ok] {s['conv_id']}: conv={conv_json}  qa={qa_json}")
    print()

    timings: dict[str, dict[str, int]] = {}
    t_pipeline_start = time.time()

    for s in sessions:
        conv = s["conv_id"]
        conv_json = os.path.join(dataset_dir, s["conv_json"])
        qa_json = os.path.join(dataset_dir, s["qa_json"])
        qa_idx = conv.replace("conv", "")

        run_dir = os.path.join(locomo_dir, "runs", conv)
        log_dir = os.path.join(run_dir, "log")
        artifacts_dir = os.path.join(run_dir, "artifacts")
        results_dir = os.path.join(run_dir, "results")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        run_env = {
            **env,
            "MOSAIC_LLM_IO_LOG": os.path.join(log_dir, "llm_io.jsonl"),
            "MOSAIC_INGEST_JSONL": os.path.join(log_dir, "ingest.jsonl"),
            "MOSAIC_NCS_TRACE_JSONL": os.path.join(log_dir, "ncs_trace.jsonl"),
        }
        run_env.setdefault("MOSAIC_CONTROL_PROFILE", "static")

        for strategy in strategies:
            label = f"{conv}_{strategy}"
            art_dir = os.path.join(artifacts_dir, strategy)
            snapshot_dir = os.path.join(art_dir, "graph_snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)

            print()
            print("#" * 60)
            print(f"#  {label}")
            print("#" * 60)

            hash_flag = ["--hash"] if strategy == "hash_only" else []
            g_pkl = os.path.join(art_dir, f"graph_network_{conv}.pkl")
            t_json = os.path.join(art_dir, f"{conv}_tags.json")

            timing: dict[str, int] = {}

            # --- 1. Build graph ---
            print(f"========== [{label} 1/3] build {strategy} ==========")
            t0 = time.time()
            cmd = [
                sys.executable, "-m", "mosaic", "build",
                "--conv-json", conv_json,
                "--conv-name", conv,
                "--graph-save-dir", snapshot_dir,
                "--graph-out", g_pkl,
                "--tags-out", t_json,
                "--progress-file", os.path.join(log_dir, f"{conv}_progress.txt"),
                "--log-prompt", os.path.join(log_dir, "llm_io.jsonl"),
            ] + hash_flag
            result = subprocess.run(cmd, env=run_env)
            timing["graph"] = int(time.time() - t0)
            if result.returncode != 0:
                print(f"[error] {label} build failed (exit {result.returncode})")

            # --- 2. Smoke query ---
            print(f"========== [{label} 2/3] smoke query {strategy} method=hash ==========")
            t0 = time.time()
            cmd = [
                sys.executable, "-m", "mosaic", "query",
                "--graph-pkl", g_pkl,
                "--tags-json", t_json,
                "--method", "hash",
                "--question", "Who are the people in this conversation?",
            ]
            result = subprocess.run(cmd, env=run_env)
            if result.returncode != 0:
                print(f"[warn] {label} smoke query failed")
            timing["smoke"] = int(time.time() - t0)

            # --- 3. Full QA evaluation ---
            print(f"========== [{label} 3/3] QA eval method=hash results-tag={strategy} ==========")
            t0 = time.time()
            cmd = [
                sys.executable, "-m", "mosaic", "query",
                "--graph-pkl", g_pkl,
                "--tags-json", t_json,
                "--qa-json", qa_json,
                "--method", "hash",
                "--output", os.path.join(results_dir, f"qa_{qa_idx}_eval_full_{strategy}.json"),
                "--summary-out", os.path.join(results_dir, f"qa_{qa_idx}_eval_summary_{strategy}.json"),
                "--resume",
            ]
            result = subprocess.run(cmd, env=run_env)
            timing["qa"] = int(time.time() - t0)

            print(f"========== [{label}] timing ==========")
            print(f"  graph={timing['graph']}  smoke={timing['smoke']}  qa={timing['qa']}")

            timings[label] = timing

    # Timing summary
    total_wall = int(time.time() - t_pipeline_start)
    print()
    print("========== Timing Summary (seconds) ==========")
    for label, t in timings.items():
        print(f"  {label}: graph={t.get('graph', '?')} smoke={t.get('smoke', '?')} qa={t.get('qa', '?')}")
    print(f"  total_wall={total_wall}")

    print()
    print("========== Done (all conversations × strategies) ==========")
    print(f"Results: {os.path.join(locomo_dir, 'runs', '*', 'results')}")
    print(f"Dataset: {dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
