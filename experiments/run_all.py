#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 experiments/README.md 顺序执行三项实验，并可选导出 LaTeX。

  python experiments/run_all.py                 # 01 → 02 → 03 → export
  python experiments/run_all.py --locomo-only # 仅 01
  python experiments/run_all.py --export-only # 仅 export_to_manuscript
"""
import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS = [
    ("01_locomo_benchmark", "run.py"),
    ("02_scalability", "run.py"),
    ("03_ablation", "run.py"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomo-only", action="store_true", help="仅运行 01_locomo_benchmark")
    parser.add_argument("--export-only", action="store_true", help="仅运行 export_to_manuscript.py")
    parser.add_argument("--no-export", action="store_true", help="不运行导出")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if args.export_only:
        return subprocess.call([sys.executable, "experiments/export_to_manuscript.py"])

    to_run = [("01_locomo_benchmark", "run.py")] if args.locomo_only else EXPERIMENTS

    for name, script in to_run:
        run_py = os.path.join("experiments", name, script)
        if not os.path.exists(run_py):
            print("Skip (missing):", run_py)
            continue
        print("\n---", name, "---")
        ret = subprocess.call([sys.executable, run_py])
        if ret != 0:
            print("Exit", ret, "from", name)
            return ret

    if not args.no_export:
        subprocess.call([sys.executable, "experiments/export_to_manuscript.py"])
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
