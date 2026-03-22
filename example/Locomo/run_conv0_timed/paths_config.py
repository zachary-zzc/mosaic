"""run_conv0_timed 与 run_qa_eval 共用的路径解析与环境变量（mosaic）。"""
from __future__ import annotations

import json
import os
from pathlib import Path


def repo_root_from_file(this_file: Path) -> Path:
    """`*_timed/*.py` -> 仓库根 LongtermMemory。"""
    return this_file.resolve().parent.parent.parent.parent


def run_dir_from_file(this_file: Path) -> Path:
    return this_file.resolve().parent


def load_paths_raw(paths_file: Path) -> dict:
    with open(paths_file, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def resolve_auto(repo: Path, run_dir: Path, raw: str, key: str) -> Path:
    s = (raw or "").strip()
    su = s.upper()
    # 极小冒烟集：data_mini 来自 example/Locomo/locomo_conv0.json 与 qa_0.json 的前若干条
    if su == "AUTO_MINI":
        if key == "locomo_conv_json":
            return run_dir / "data_mini" / "locomo_conv0_mini.json"
        if key == "qa_json":
            return run_dir / "data_mini" / "qa_0_mini.json"
        if key == "log_dir":
            return run_dir / "log_mini"
        if key == "results_dir":
            return run_dir / "results_mini"
        if key == "artifacts_dir":
            return run_dir / "artifacts_mini"
    if su == "AUTO" or not s:
        if key == "mosaic_root":
            return repo / "mosaic"
        if key == "locomo_conv_json":
            return run_dir.parent / "locomo_conv0.json"
        if key == "qa_json":
            return run_dir.parent / "qa_0.json"
        if key == "log_dir":
            return run_dir / "log"
        if key == "results_dir":
            return run_dir / "results"
        if key == "mosaic_config_path":
            return repo / "mosaic" / "config" / "config.cfg"
        if key == "artifacts_dir":
            return run_dir / "artifacts"
    p = Path(s)
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def build_resolved_paths(repo: Path, run_dir: Path, raw: dict) -> dict[str, Path | str]:
    paths: dict[str, Path | str] = {}
    for key in (
        "mosaic_root",
        "locomo_conv_json",
        "qa_json",
        "log_dir",
        "results_dir",
        "mosaic_config_path",
        "artifacts_dir",
    ):
        paths[key] = resolve_auto(repo, run_dir, str(raw.get(key, "AUTO")), key)
    paths["embedding_model_override"] = (raw.get("embedding_model_override") or "").strip()
    return paths


def default_graph_and_tags(artifacts_dir: Path) -> tuple[Path, Path]:
    """与 run.py / mosaic build 约定一致。"""
    return (
        artifacts_dir / "graph_network_conv0.pkl",
        artifacts_dir / "conv0_tags.json",
    )


def apply_mosaic_env(
    paths: dict[str, Path | str],
    *,
    log_basename: str = "mosaic_server.log",
    progress_relpath: str | None = "conv0_progress.txt",
    set_graph_snapshots_dir: bool = True,
) -> None:
    """设置 MOSAIC_*；评测时可用 log_basename=qa_eval.log 且 set_graph_snapshots_dir=False。"""
    log_dir = paths["log_dir"]
    assert isinstance(log_dir, Path)
    log_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MOSAIC_CONFIG_PATH"] = str(paths["mosaic_config_path"])
    os.environ["MOSAIC_LOG_DIR"] = str(log_dir)
    os.environ["MOSAIC_SERVER_LOG_BASENAME"] = log_basename
    if progress_relpath:
        os.environ["MOSAIC_PROGRESS_FILE"] = str(log_dir / progress_relpath)
    else:
        os.environ.pop("MOSAIC_PROGRESS_FILE", None)

    os.environ["MOSAIC_CONSOLE_MIN_LEVEL"] = "ERROR"

    override = (paths.get("embedding_model_override") or "").strip()
    if override:
        os.environ["MOSAIC_EMBEDDING_MODEL"] = str(Path(override).resolve())
    else:
        os.environ.pop("MOSAIC_EMBEDDING_MODEL", None)

    if set_graph_snapshots_dir:
        artifacts = paths["artifacts_dir"]
        assert isinstance(artifacts, Path)
        graph_snapshots = artifacts / "graph_snapshots"
        graph_snapshots.mkdir(parents=True, exist_ok=True)
        os.environ["GRAPH_SAVE_DIR"] = str(graph_snapshots)
    else:
        os.environ.pop("GRAPH_SAVE_DIR", None)
