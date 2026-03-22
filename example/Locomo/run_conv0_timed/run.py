#!/usr/bin/env python3
"""
Locomo conv0 构图入口（独立于 mosaic/ 源码目录，产物写入本目录 artifacts/ 与 log/）。

使用仓库内 **mosaic CLI**（`python -m mosaic build`）构图，与当前 mosaic 主流程一致。

stdout：任务控制信息 + 子进程继承的 tqdm/输出。
详细运行参数与环境：使用 --verbose-log 写入 log/run_verbose.log

用法（在仓库根目录或本目录）:
  conda activate mosaic
  python run.py
  python run.py --verbose-log
  python run.py --hash
  python run.py --paths /path/to/paths.json

后台:
  nohup python run.py --verbose-log >> log/task_stdout.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import platform
from datetime import datetime, timezone
from pathlib import Path

from paths_config import (
    apply_mosaic_env,
    build_resolved_paths,
    default_graph_and_tags,
    load_paths_raw,
    repo_root_from_file,
    run_dir_from_file,
)


def _write_verbose_log(log_dir: Path, paths: dict, args: argparse.Namespace) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "run_verbose.log"
    lines = [
        f"utc={datetime.now(timezone.utc).isoformat()}",
        f"platform={platform.platform()}",
        f"python={sys.version}",
        f"executable={sys.executable}",
        f"cwd={os.getcwd()}",
        f"argv={sys.argv!r}",
        "--- resolved paths ---",
    ]
    for k, v in sorted(paths.items()):
        lines.append(f"{k}={v}")
    lines.append("--- relevant env ---")
    for k in sorted(os.environ):
        if k.startswith("MOSAIC_") or k in ("PATH", "PYTHONPATH", "GRAPH_SAVE_DIR"):
            val = os.environ[k]
            if "key" in k.lower() or "secret" in k.lower() or "token" in k.lower():
                val = "***"
            lines.append(f"{k}={val}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _print_control(msg: str) -> None:
    print(msg, flush=True)


def _conv_name_from_json(conv_json: Path) -> str:
    return conv_json.name.replace("locomo_", "").replace(".json", "").replace(".JSON", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Locomo conv0 构图（mosaic CLI build，日志在本目录 log/）")
    parser.add_argument(
        "--paths",
        type=Path,
        default=None,
        help="paths.json 路径（默认为本目录 paths.json）",
    )
    parser.add_argument(
        "--verbose-log",
        action="store_true",
        help="将详细运行参数写入 log/run_verbose.log；并传给 mosaic build -v",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="仅 TF-IDF/hash 构图（mosaic build --hash），不调用 LLM 建新类",
    )
    args = parser.parse_args()

    this = Path(__file__)
    run_dir = run_dir_from_file(this)
    repo = repo_root_from_file(this)
    paths_file = args.paths or (run_dir / "paths.json")
    if not paths_file.is_file():
        _print_control(f"[错误] 找不到配置文件: {paths_file}")
        sys.exit(1)

    raw = load_paths_raw(paths_file)
    paths = build_resolved_paths(repo, run_dir, raw)

    conv_json = paths["locomo_conv_json"]
    assert isinstance(conv_json, Path)
    if not conv_json.is_file():
        _print_control(f"[错误] 对话数据不存在: {conv_json}")
        sys.exit(1)
    mcfg = paths["mosaic_config_path"]
    assert isinstance(mcfg, Path)
    if not mcfg.is_file():
        _print_control(f"[警告] mosaic 配置文件不存在: {mcfg}（API 等可能无法读取）")

    artifacts = paths["artifacts_dir"]
    assert isinstance(artifacts, Path)
    artifacts.mkdir(parents=True, exist_ok=True)
    graph_pkl, tags_json = default_graph_and_tags(artifacts)

    log_dir = paths["log_dir"]
    assert isinstance(log_dir, Path)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose_log:
        vpath = _write_verbose_log(log_dir, paths, args)
        _print_control(f"[任务] 详细参数已写入: {vpath}")

    apply_mosaic_env(
        paths,
        log_basename="mosaic_server.log",
        progress_relpath="conv0_progress.txt",
        set_graph_snapshots_dir=True,
    )

    mosaic_root = paths["mosaic_root"]
    assert isinstance(mosaic_root, Path)
    conv_name = _conv_name_from_json(conv_json)

    _print_control("[任务] Locomo conv0 构图（mosaic: python -m mosaic build）")
    _print_control(f"[配置] mosaic 根目录: {mosaic_root}")
    _print_control(f"[配置] 对话 JSON: {conv_json}")
    _print_control(f"[配置] 日志目录: {log_dir}")
    _print_control(f"[配置] 构图产物目录: {artifacts}")
    _print_control(f"[配置] 最终图 pkl: {graph_pkl}")
    _print_control(f"[配置] tags JSON: {tags_json}")
    _print_control(f"[配置] mosaic 配置: {mcfg}")
    _print_control("[日志] 构图进度: " + str(log_dir / "conv0_progress.txt"))
    _print_control("[日志] 详细运行(mosaic): " + str(log_dir / "mosaic_server.log"))

    cmd: list[str] = [
        sys.executable,
        "-m",
        "mosaic",
    ]
    if args.verbose_log:
        cmd.append("-v")
    cmd.extend(
        [
            "build",
            "--conv-json",
            str(conv_json),
            "--conv-name",
            conv_name,
            "--graph-save-dir",
            str(artifacts / "graph_snapshots"),
            "--graph-out",
            str(graph_pkl),
            "--tags-out",
            str(tags_json),
            "--progress-file",
            str(log_dir / "conv0_progress.txt"),
        ]
    )
    if args.hash:
        cmd.append("--hash")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)

    _print_control("[启动] 执行: " + " ".join(cmd))
    _print_control(f"[启动] cwd={repo} PYTHONPATH={repo}")
    start = time.perf_counter()
    try:
        subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            check=True,
        )
        elapsed = time.perf_counter() - start
        _print_control(f"[完成] 图与 tags 已写入 {artifacts}，耗时 {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - start
        _print_control(f"[失败] 进程退出码 {e.returncode}，已运行 {elapsed:.1f} 秒")
        sys.exit(e.returncode)
    except Exception as e:
        elapsed = time.perf_counter() - start
        _print_control(f"[失败] 已运行 {elapsed:.1f} 秒: {e}")
        raise


if __name__ == "__main__":
    main()
