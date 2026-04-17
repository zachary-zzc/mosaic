#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HaluMem Experiments — Full Pipeline Runner

Runs: benchmark → ablation → scalability

Usage:
  python experiments/halumem/start_experiment.py
  python experiments/halumem/start_experiment.py --skip ablation scalability
"""
import argparse
import os
import subprocess
import sys
import time

def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def _run(label, cmd, env):
    print(f">>> {label}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[error] {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print()

def main():
    parser = argparse.ArgumentParser(description="HaluMem Experiments — Full Pipeline")
    parser.add_argument("--skip", nargs="*", default=[], choices=["benchmark", "ablation", "scalability"], help="Sub-experiments to skip")
    args, extra = parser.parse_known_args()
    repo = _repo_root()
    env = {**os.environ, "PYTHONPATH": repo}
    t_start = time.time()
    print("=" * 60)
    print("  HaluMem Experiments — Full Pipeline")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    steps = [
        ("benchmark", "[1/3] Benchmark", [sys.executable, os.path.join(repo, "experiments", "halumem", "benchmark", "run.py")] + extra),
        ("ablation", "[2/3] Ablation", [sys.executable, os.path.join(repo, "experiments", "halumem", "ablation", "run.py")]),
        ("scalability", "[3/3] Scalability", [sys.executable, os.path.join(repo, "experiments", "halumem", "scalability", "run.py")]),
    ]
    for key, label, cmd in steps:
        if key in args.skip:
            print(f">>> Skipping {label}")
            print()
            continue
        _run(label, cmd, env)
    elapsed = int(time.time() - t_start)
    print("=" * 60)
    print(f"  All HaluMem experiments complete — {elapsed}s total")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
