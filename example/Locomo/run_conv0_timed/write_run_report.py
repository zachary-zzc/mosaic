#!/usr/bin/env python3
"""
根据流水线日志与 qa_0_eval_summary.json 生成 Markdown 评测报告。

由 start_background.sh / start_background_full.sh 在流水线末尾调用。
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def _parse_graph_seconds_from_task_log(text: str) -> float | None:
    """匹配 run.py 输出: [完成] 图与 tags 已写入 ...，耗时 12.3 秒"""
    pat = re.compile(r"\[完成\] 图与 tags 已写入 .+，耗时 ([0-9.]+) 秒")
    matches = list(pat.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group(1))


def _fmt_sec(s: float | int | None) -> str:
    if s is None:
        return "—"
    if isinstance(s, int):
        return f"{s} 秒 ({s / 60:.2f} 分钟)"
    return f"{s:.1f} 秒 ({s / 60:.2f} 分钟)"


def _category_name(cat: str) -> str:
    # 与 Locomo 常见约定一致（若不符仅作展示标签）
    names = {
        "1": "单跳 / 类别 1",
        "2": "多跳 / 类别 2",
        "3": "时序 / 类别 3",
        "4": "开放 / 类别 4",
        "5": "跳过类别",
    }
    return names.get(str(cat), f"类别 {cat}")


def main() -> int:
    p = argparse.ArgumentParser(description="生成 Locomo conv0 流水线 Markdown 报告")
    p.add_argument("--mode", choices=("mini", "full"), required=True)
    p.add_argument("--run-dir", type=Path, default=None, help="run_conv0_timed 目录，默认脚本所在目录")
    p.add_argument("--wall-graph", type=int, default=None, help="构图阶段墙钟秒数（bash 传入）")
    p.add_argument("--wall-smoke", type=int, default=None, help="单条冒烟 query 墙钟秒数")
    p.add_argument("--wall-qa-eval", type=int, default=None, help="run_qa_eval 整段墙钟秒数")
    p.add_argument("--wall-total", type=int, default=None, help="流水线总墙钟秒数")
    args = p.parse_args()

    run_dir = args.run_dir or Path(__file__).resolve().parent
    run_dir = run_dir.resolve()

    if args.mode == "mini":
        summary_path = run_dir / "results_mini" / "qa_0_eval_summary.json"
        task_log = run_dir / "log" / "task_stdout.log"
        artifacts = run_dir / "artifacts_mini"
        results_dir = run_dir / "results_mini"
        log_dir = run_dir / "log_mini"
        report_path = run_dir / "log" / "run_report_mini.md"
        label = "mini（data_mini）"
    else:
        summary_path = run_dir / "results" / "qa_0_eval_summary.json"
        task_log = run_dir / "log" / "task_stdout.log"
        artifacts = run_dir / "artifacts"
        results_dir = run_dir / "results"
        log_dir = run_dir / "log"
        report_path = run_dir / "log" / "run_report_full.md"
        label = "全量 conv0"

    task_text = task_log.read_text(encoding="utf-8", errors="replace") if task_log.is_file() else ""
    graph_runpy_sec = _parse_graph_seconds_from_task_log(task_text)

    summary: dict = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inner = summary.get("summary") or {}

    total_q = inner.get("total_questions", 0)
    total_ok = inner.get("total_correct", 0)
    total_bad = inner.get("total_wrong", 0)
    acc = inner.get("overall_accuracy")
    cat_stats = inner.get("category_stats") or {}
    errors = inner.get("errors") or []

    utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = [
        f"# Locomo conv0 流水线评测报告（{label}）",
        "",
        f"- **生成时间**: {utc}",
        f"- **运行目录**: `{run_dir}`",
        f"- **检索方式**: `{summary.get('query_method', '—')}`",
        "",
        "## 计时说明（读报告前请看）",
        "",
        "### 构图为什么可以很快？统计有没有错？",
        "",
        "- **没有错**：`run.py` 里对 `python -m mosaic build` 子进程使用的是 `time.perf_counter()`，"
        "表格中「run.py 报告」一行解析自日志里的 `[完成] 图与 tags 已写入 …，耗时 X 秒`，与子进程真实耗时一致。",
        "- **本仓库后台脚本（`start_background.sh` / `start_background_full.sh`）固定传入 `run.py --hash`**："
        "构图走 **TF-IDF/hash 启发式**，**不会**像默认 LLM 路径那样对每批消息调用大模型做分类/建类，因此全量 conv0 也可能在较短时间内完成。",
        "- 若要与论文/实验中的 **「LLM 构图」** 对比耗时，请使用 **不带 `--hash` 的 `run.py`**（或 mosaic `build` 的 LLM 模式）；其墙钟会显著更长。",
        "",
        "### QA / query 为什么慢？能否优化？",
        "",
        "- **主要瓶颈在网络 API**：即使检索为 `hash`，每条题仍需要 **(1) LLM 根据检索片段生成答案**、**(2) LLM 评判 CORRECT/WRONG**，"
        "约 **2 次** chat 调用/题；`method=llm` 时检索阶段还会额外调用 LLM（类感知、关键词补充等），更慢。",
        "- **脚本里的「单条 query」** 同样是 hash 检索 + **一次**作答 LLM，若网络延迟高，冒烟一步也会显得久。",
        "- **已做的小优化**：`fetch_default_llm_model()` 在进程内 **复用同一 ChatModel 实例**（避免重复构造包装类；HTTP 连接仍按 endpoint 复用）。",
        "- **若仍要加速（需改实验协议或工程）**：提高并发（多题并行评判，注意配额）、"
        "批处理 API、或在允许时合并/简化评判流程；检索侧对大图可再做 TF-IDF 向量缓存（当前每题会重算相关矩阵，图很大时 CPU 也会占时间）。",
        "",
        "- **墙钟行（bash `date`）** 为整秒，可能与子进程精确耗时差 0～1 秒。",
        "",
        "## 计时",
        "",
        "| 阶段 | 说明 | 时间 |",
        "|------|------|------|",
    ]

    lines.append(
        f"| 构图 | run.py → mosaic build（`--hash`）；以下为 run.py 子进程耗时 | {_fmt_sec(graph_runpy_sec)} |"
    )
    if args.wall_graph is not None:
        lines.append(f"| 构图（墙钟） | 自 bash 包裹的整段 run.py | {_fmt_sec(args.wall_graph)} |")
    lines.append(
        f"| 单条 query | `python -m mosaic query` 冒烟一条 | {_fmt_sec(args.wall_smoke)} |"
    )
    lines.append(
        f"| QA 评测 | `run_qa_eval.py` 全量逐题检索+作答+评判 | {_fmt_sec(args.wall_qa_eval)} |"
    )
    lines.append(f"| **合计（墙钟）** | 自流水线开始到评测结束 | {_fmt_sec(args.wall_total)} |")
    lines.append("")
    lines.append(
        "说明：构图「run.py 报告」为 mosaic 子进程的 `perf_counter` 耗时；bash 墙钟为整秒且含进程启动。"
        " 后台流水线构图阶段 **始终带 `--hash`**，见上文「计时说明」。"
    )
    lines.append("")

    lines.extend(
        [
            "## QA 准确率与题量",
            "",
            f"- **QA 文件**: `{summary.get('qa_file', '—')}`",
            f"- **图**: `{summary.get('graph_file', '—')}`",
            f"- **tags**: `{summary.get('tag_file', '—')}`",
            f"- **总题数**: {total_q}",
            f"- **正确 / 错误**: {total_ok} / {total_bad}",
        ]
    )
    if acc is not None and total_q:
        lines.append(f"- **整体准确率**: {acc:.4f}（{acc * 100:.2f}%）")
    else:
        lines.append("- **整体准确率**: —")
    lines.append("")

    lines.append("### 按类别（category）")
    lines.append("")
    lines.append("| 类别 | 名称（示意） | 正确 | 错误 | 小题数 | 准确率 |")
    lines.append("|------|----------------|------|------|--------|--------|")

    for cat in sorted(cat_stats.keys(), key=lambda x: (len(str(x)), str(x))):
        st = cat_stats[cat]
        c, w, t = st.get("correct", 0), st.get("wrong", 0), st.get("total", 0)
        a = (c / t) if t else 0.0
        lines.append(
            f"| `{cat}` | {_category_name(cat)} | {c} | {w} | {t} | {a:.4f} ({a * 100:.2f}%) |"
        )
    lines.append("")

    lines.append("### 评测错误记录（errors 字段）")
    lines.append("")
    if not errors:
        lines.append("无（列表为空）。")
    else:
        lines.append(f"共 {len(errors)} 条，详见 `{summary_path}` 内 `summary.errors`。")
    lines.append("")

    lines.extend(
        [
            "## 产物与日志",
            "",
            f"- **图 / tags 目录**: `{artifacts}`",
            f"- **评测 JSON**: `{results_dir}/qa_0_eval_summary.json`、`{results_dir}/qa_0_eval_full.json`",
            f"- **流水线 stdout**: `{run_dir / 'log' / 'task_stdout.log'}`",
            f"- **mosaic 详情日志**: `{log_dir / 'mosaic_server.log'}`",
            f"- **QA 评测日志**: `{log_dir / 'qa_eval.log'}`",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[报告] 已写入: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
