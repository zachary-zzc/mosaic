#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Copy generated tab_*.tex from experiment results to Manuscript/generated/.

Run experiments first, then run this script:
  python experiments/export_to_manuscript.py

Or use the dataset-level orchestrator:
  python experiments/locomo/start_experiment.py
  experiments/locomo/start_experiment.sh  (Unix/macOS only)
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GENERATED_DIR = os.path.join(PROJECT_ROOT, "Manuscript", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

# (experiment path relative to experiments/, source tex filename)
EXPERIMENT_TABLES = [
    ("locomo/benchmark", "tab_locomo.tex"),
    ("locomo/ablation", "tab_ablation_mosaic.tex"),
    ("locomo/scalability", "tab_scalability.tex"),
]


def main():
    copied = []
    for exp_name, filename in EXPERIMENT_TABLES:
        src = os.path.join(PROJECT_ROOT, "experiments", exp_name, "results", filename)
        if os.path.exists(src):
            dst = os.path.join(GENERATED_DIR, filename)
            with open(src, "r", encoding="utf-8") as f:
                content = f.read()
            with open(dst, "w", encoding="utf-8") as f:
                f.write(content)
            copied.append(filename)
            print("Written:", dst)
        else:
            print("Skip (not found):", src)

    readme = os.path.join(GENERATED_DIR, "README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("Generated tables from experiment results.\n")
        f.write("Files: " + ", ".join(copied) + "\n")
    print("Written:", readme)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
