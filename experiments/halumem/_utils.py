# -*- coding: utf-8 -*-
"""
Shared utilities for HaluMem experiments: path resolution, mosaic environment setup, JSONL helpers, and dataset access.
"""
import os
import sys
import json
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MOSAIC_DIR = os.path.join(PROJECT_ROOT, "mosaic")
MOSAIC_SRC = os.path.join(MOSAIC_DIR, "src")
DATASET_HALUMEM_DIR = os.path.join(PROJECT_ROOT, "dataset", "halumem", "data")


def setup_mosaic_path():
    """Add mosaic and mosaic/src to sys.path."""
    for p in (MOSAIC_DIR, MOSAIC_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_jsonl(path):
    """Load a JSONL file as a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(path, data, indent=2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def get_halumem_jsonl_files():
    """Return all main HaluMem JSONL files (Medium, Long, test)."""
    files = []
    for name in ["HaluMem-Medium.jsonl", "HaluMem-Long.jsonl", "test.jsonl"]:
        path = os.path.join(DATASET_HALUMEM_DIR, name)
        if os.path.isfile(path):
            files.append(path)
    return files
