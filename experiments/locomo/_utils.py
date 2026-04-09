# -*- coding: utf-8 -*-
"""
Shared utilities for LoCoMo experiments: path resolution, mosaic environment
setup, JSON helpers, and dataset access.
"""
import os
import sys
import json
import glob

# Repository root (grandparent of experiments/locomo/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MOSAIC_DIR = os.path.join(PROJECT_ROOT, "mosaic")
MOSAIC_SRC = os.path.join(MOSAIC_DIR, "src")
DATASET_LOCOMO_DIR = os.path.join(PROJECT_ROOT, "dataset", "locomo")


def setup_mosaic_path():
    """Add mosaic and mosaic/src to sys.path."""
    for p in (MOSAIC_DIR, MOSAIC_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_json_safe(path, default=None):
    """Load JSON file, returning *default* on any error."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data, indent=2):
    """Write *data* as JSON to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def get_locomo_example_pairs():
    """
    Return (conv_path, qa_path) pairs from dataset/locomo/.
    Convention: locomo_conv0.json matches qa_0.json by numeric suffix.
    """
    pairs = []
    if not os.path.isdir(DATASET_LOCOMO_DIR):
        return pairs
    conv_pattern = os.path.join(DATASET_LOCOMO_DIR, "locomo_conv*.json")
    for conv_path in sorted(glob.glob(conv_pattern)):
        base = os.path.basename(conv_path)
        try:
            num = base.replace("locomo_conv", "").replace(".json", "").strip()
            if not num:
                num = "0"
            qa_path = os.path.join(DATASET_LOCOMO_DIR, f"qa_{num}.json")
            if os.path.isfile(qa_path):
                pairs.append((conv_path, qa_path))
        except Exception:
            continue
    return pairs


def build_locomo_graph_hash(conv_path, out_dir, conv_name=None):
    """
    Build graph from conv JSON using hash method, outputting to out_dir.
    Returns (graph_pkl_path, tags_json_path).
    Must be called after setup_mosaic_path().
    """
    from src.save import save_hash

    with open(conv_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name = conv_name or os.path.basename(conv_path).replace("locomo_", "").replace(".json", "conv")
    os.makedirs(out_dir, exist_ok=True)
    graph_path = os.path.join(out_dir, "graph.pkl")
    tags_path = os.path.join(out_dir, "tags.json")
    save_hash(
        data,
        conv_name=name,
        graph_save_dir=out_dir,
        final_graph_path=graph_path,
        final_tags_path=tags_path,
    )
    return graph_path, tags_path


def load_sessions():
    """Load experiment session registry from dataset/locomo/experiment_sessions.json."""
    sessions_path = os.path.join(DATASET_LOCOMO_DIR, "experiment_sessions.json")
    if not os.path.isfile(sessions_path):
        print(f"[error] Session registry not found: {sessions_path}")
        sys.exit(1)
    data = load_json_safe(sessions_path)
    return data.get("sessions", [])
