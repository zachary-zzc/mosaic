#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoCoMo dataset — run all sub-experiments in sequence.

Cross-platform replacement for start_experiment.sh.

Runs: benchmark → ablation → scalability, then exports LaTeX tables.

Usage:
  python experiments/locomo/start_experiment.py
  python experiments/locomo/start_experiment.py --skip ablation scalability
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _run(label: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f">>> {label}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[error] {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="LoCoMo Experiments — Full Pipeline")
    parser.add_argument("--skip", nargs="*", default=[], choices=["benchmark", "ablation", "scalability", "export"],
                        help="Sub-experiments to skip")
    args, extra = parser.parse_known_args()

    repo = _repo_root()
    env = {**os.environ, "PYTHONPATH": repo}

    t_start = time.time()

    print("=" * 60)
    print("  LoCoMo Experiments — Full Pipeline")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    steps = [
        ("benchmark", "[1/4] Benchmark (hybrid + hash_only)",
         [sys.executable, os.path.join(repo, "experiments", "locomo", "benchmark", "run.py")] + extra),
        ("ablation", "[2/4] Ablation (C0-C6)",
         [sys.executable, os.path.join(repo, "experiments", "locomo", "ablation", "run.py")]),
        ("scalability", "[3/4] Scalability",
         [sys.executable, os.path.join(repo, "experiments", "locomo", "scalability", "run.py")]),
        ("export", "[4/4] Export tables to Manuscript/generated/",
         [sys.executable, os.path.join(repo, "experiments", "export_to_manuscript.py")]),
    ]

    for key, label, cmd in steps:
        if key in args.skip:
            print(f">>> Skipping {label}")
            print()
            continue
        _run(label, cmd, env)

    elapsed = int(time.time() - t_start)
    print("=" * 60)
    print(f"  All LoCoMo experiments complete — {elapsed}s total")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
