#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalability experiment launcher — cross-platform replacement for start_experiment.sh.

Truncates LoCoMo conversations to varying lengths and runs build + QA.

Usage:
  python experiments/locomo/scalability/start_experiment.py
  python experiments/locomo/scalability/start_experiment.py --conv conv0 --lengths 100 200 400
"""
from __future__ import annotations

import os
import subprocess
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main() -> int:
    repo = _repo_root()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(script_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    env = {**os.environ, "PYTHONPATH": repo}

    print("=" * 10 + " Scalability Experiment " + "=" * 10)
    print(f"  repo: {repo}")
    print(f"  python: {sys.executable}")
    print()

    cmd = [sys.executable, os.path.join(repo, "experiments", "locomo", "scalability", "run.py")] + sys.argv[1:]
    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
