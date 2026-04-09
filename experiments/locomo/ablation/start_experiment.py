#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation experiment launcher — cross-platform replacement for start_experiment.sh.

Each condition disables one MOSAIC component and re-evaluates on LoCoMo.
C0/C1 reuse benchmark results; C2-C6 run with env var overrides.

Usage:
  python experiments/locomo/ablation/start_experiment.py
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

    print("=" * 10 + " Ablation Experiment " + "=" * 10)
    print(f"  repo: {repo}")
    print(f"  python: {sys.executable}")
    print()

    cmd = [sys.executable, os.path.join(repo, "experiments", "locomo", "ablation", "run.py")] + sys.argv[1:]
    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
