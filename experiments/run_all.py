#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整构建：按顺序执行 01–09 与导出。无占位符，全部填满。
  python experiments/run_all.py              # 完整跑 01–09（09 可 --skip-run 若已有 results）
  python experiments/run_all.py --locomo-only [--max-questions N]  # 只跑 09
  python experiments/run_all.py --export-only # 只从现有 results 导出 LaTeX 到 Manuscript/generated
"""
import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS = [
    "01_memory_gap",
    "02_baselines",
    "03_evolving_dag",
    "04_ncs_mechanism",
    "05_ablation",
    "06_graph_construction",
    "07_downstream",
    "08_pilot",
    "09_locomo",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomo-only", action="store_true", help="Only run 09_locomo")
    parser.add_argument("--max-questions", type=int, default=None, help="For 09: max questions per QA file")
    parser.add_argument("--skip-locomo-run", action="store_true", help="For 09: skip QA run, only aggregate and export")
    parser.add_argument("--export-only", action="store_true", help="Only run export_to_manuscript.py")
    parser.add_argument("--no-export", action="store_true", help="Do not run export after experiments")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if args.export_only:
        return subprocess.call([sys.executable, "experiments/export_to_manuscript.py"])

    if args.locomo_only:
        cmd = [sys.executable, "experiments/09_locomo/run_locomo.py"]
        if args.max_questions is not None:
            cmd.extend(["--max-questions", str(args.max_questions)])
        if args.skip_locomo_run:
            cmd.append("--skip-run")
        ret = subprocess.call(cmd)
        if ret == 0 and not args.no_export:
            subprocess.call([sys.executable, "experiments/export_to_manuscript.py"])
        return ret

    for name in EXPERIMENTS:
        run_py = "experiments/09_locomo/run_locomo.py" if name == "09_locomo" else f"experiments/{name}/run.py"
        if not os.path.exists(run_py) and name != "09_locomo":
            run_py = os.path.join("experiments", name, "run.py")
        if not os.path.exists(run_py):
            print("Skip (no script):", name)
            continue
        cmd = [sys.executable, run_py]
        if name == "09_locomo":
            if args.skip_locomo_run:
                cmd.append("--skip-run")
            if args.max_questions is not None:
                cmd.extend(["--max-questions", str(args.max_questions)])
        print("\n---", name, "---")
        ret = subprocess.call(cmd)
        if ret != 0:
            print("Exit", ret, "from", name)
            return ret

    if not args.no_export:
        subprocess.call([sys.executable, "experiments/export_to_manuscript.py"])
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
