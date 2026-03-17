# -*- coding: utf-8 -*-
"""
实验运行工具：解析路径、设置 mosaic 运行环境，供各 experiment 脚本调用。
"""
import os
import sys
import json
import glob

# 仓库根目录（experiments/ 的上级）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOSAIC_DIR = os.path.join(PROJECT_ROOT, "mosaic")
MOSAIC_SRC = os.path.join(MOSAIC_DIR, "src")
# LoCoMo 数据在 mosaic/src 下的相对路径（与 save.py / query.py 一致）
LOCOMO_BASE = "locomo results"
LOCOMO_CONV_DIR = os.path.join(MOSAIC_SRC, LOCOMO_BASE, "conv")
LOCOMO_QA_DIR = os.path.join(MOSAIC_SRC, LOCOMO_BASE, "qa")
LOCOMO_GRAPH_DIR = os.path.join(MOSAIC_SRC, LOCOMO_BASE, "graph")
LOCOMO_TAGS_DIR = os.path.join(MOSAIC_SRC, LOCOMO_BASE, "tags")


def setup_mosaic_path():
    """将 mosaic 与 mosaic/src 加入 sys.path（满足 data.* 与 src.data.* 导入），并 chdir 到 mosaic/src。"""
    for p in (MOSAIC_DIR, MOSAIC_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(MOSAIC_SRC)


def get_locomo_qa_graph_tag_pairs():
    """
    返回 (qa_file, graph_file, tag_file) 列表，与 mosaic query 使用的约定一致。
    qa_i.json 对应 pkl_con{i} 下唯一 .pkl 与 tags/conv{i}{i}_tag.json。
    """
    pairs = []
    for i in range(10):
        qa_path = os.path.join(LOCOMO_QA_DIR, f"qa_{i}.json")
        tag_path = os.path.join(LOCOMO_TAGS_DIR, f"conv{i}{i}_tag.json")
        pkl_dir = os.path.join(LOCOMO_GRAPH_DIR, f"pkl_con{i}")
        if not os.path.isdir(pkl_dir):
            continue
        pkls = glob.glob(os.path.join(pkl_dir, "*.pkl"))
        if not pkls:
            continue
        graph_path = max(pkls, key=os.path.getmtime)
        if os.path.exists(qa_path) and os.path.exists(tag_path):
            pairs.append((qa_path, graph_path, tag_path))
    return pairs


def get_locomo_conv_files():
    """返回 locomo conv JSON 列表（用于 save 流程）。"""
    pattern = os.path.join(LOCOMO_CONV_DIR, "locomo_conv*.json")
    files = sorted(glob.glob(pattern))
    return files


def ensure_results_dir(experiment_name: str) -> str:
    """确保 experiments/<name>/results 存在并返回其路径。"""
    path = os.path.join(PROJECT_ROOT, "experiments", experiment_name, "results")
    os.makedirs(path, exist_ok=True)
    return path


def load_json_safe(path: str, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data, indent=2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
