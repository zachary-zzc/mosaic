"""
Mosaic 命令行：子命令 build（构图）、query（读图查询）、run（构图后查询）、chat（交互）。

使用前请将 mosaic 目录加入 PYTHONPATH，例如::

    cd mosaic && PYTHONPATH=. python -m mosaic --help
    cd mosaic && PYTHONPATH=. python cli.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _apply_logging(verbose: bool) -> None:
    """与 logger.setup_logger 配合：--verbose 时控制台 DEBUG；否则 WARNING（进度靠 tqdm）。"""
    os.environ["MOSAIC_VERBOSE"] = "1" if verbose else "0"
    if verbose:
        os.environ.pop("MOSAIC_CONSOLE_MIN_LEVEL", None)
    else:
        os.environ.setdefault("MOSAIC_CONSOLE_MIN_LEVEL", "WARNING")


def _conv_name_from_path(path: str) -> str:
    name = os.path.basename(path)
    return name.replace("locomo_", "").replace(".json", "").replace(".JSON", "")


def _apply_log_prompt_arg(args: argparse.Namespace, *, default_beside: str | None) -> None:
    """--log-prompt [PATH]：设置 MOSAIC_LLM_IO_LOG（完整 prompt/response JSONL）。"""
    v = getattr(args, "log_prompt", None)
    if v is None:
        return
    if v == "__AUTO__":
        if default_beside:
            path = str(Path(default_beside).resolve().parent / "llm_io.jsonl")
        else:
            path = str(Path.cwd() / "llm_io.jsonl")
    else:
        path = os.path.abspath(v)
    os.environ["MOSAIC_LLM_IO_LOG"] = path


def _default_build_checkpoint_path(graph_out: str | None) -> str | None:
    """与 --graph-out 同目录、同主名的构图状态 pickle（含 .meta.json）。"""
    if not graph_out:
        return None
    return os.path.abspath(graph_out) + ".build_state.pkl"


def cmd_build(args: argparse.Namespace) -> int:
    _apply_logging(args.verbose)
    _apply_log_prompt_arg(
        args,
        default_beside=args.graph_out or os.environ.get("GRAPH_SAVE_DIR"),
    )
    if getattr(args, "progress_file", None):
        os.environ["MOSAIC_PROGRESS_FILE"] = os.path.abspath(args.progress_file)
    if getattr(args, "graph_save_dir", None):
        os.environ["GRAPH_SAVE_DIR"] = os.path.abspath(args.graph_save_dir)

    from src.config_loader import get_mosaic_build_mode
    from src.save import remove_build_checkpoint, save, save_hash

    with open(args.conv_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    conv_name = args.conv_name or _conv_name_from_path(args.conv_json)

    cfg_mode = get_mosaic_build_mode()
    hash_only = bool(args.hash) or cfg_mode == "hash_only"
    os.environ["MOSAIC_BUILD_EFFECTIVE"] = "hash_only" if hash_only else "hybrid"

    ck = getattr(args, "checkpoint", None) or os.environ.get("MOSAIC_BUILD_CHECKPOINT", "").strip() or None
    if not ck and args.graph_out:
        ck = _default_build_checkpoint_path(args.graph_out)
    resume = bool(getattr(args, "resume", False))
    src_path = os.path.abspath(args.conv_json)

    if hash_only:
        save_hash(
            data,
            conv_name,
            graph_save_dir=args.graph_save_dir,
            final_graph_path=args.graph_out,
            final_tags_path=args.tags_out,
            checkpoint_path=ck,
            resume=resume,
            source_path=src_path,
        )
    else:
        memory = save(
            data,
            conv_name,
            checkpoint_path=ck,
            resume=resume,
            source_path=src_path,
        )
        if args.graph_out:
            outp = os.path.abspath(args.graph_out)
            os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
            with open(outp, "wb") as f:
                pickle.dump(memory, f, protocol=pickle.HIGHEST_PROTOCOL)
        if args.tags_out:
            memory.generate_tags_tfidf(args.tags_out)
    remove_build_checkpoint(ck)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    _apply_logging(args.verbose)
    _apply_log_prompt_arg(args, default_beside=args.graph_pkl)

    from src.assist import load_mosaic_memory_pickle
    from src.query import process_single_qa, query_with_telemetry

    method = args.method or "hash"

    if args.question:
        memory = load_mosaic_memory_pickle(args.graph_pkl)
        memory.process_kw(args.tags_json)
        out = query_with_telemetry(args.question, memory, method=method)
        print(out.get("answer", ""))
        show = not bool(getattr(args, "no_show_retrieval", False))
        if show:
            import json as _json

            print("\n--- retrieved_context (摘要) ---", flush=True)
            print(_json.dumps(out.get("retrieved_context"), ensure_ascii=False, indent=2), flush=True)
            print("\n--- timing_ms ---", flush=True)
            print(_json.dumps(out.get("timing_ms"), ensure_ascii=False, indent=2), flush=True)
            print("\n--- graph_stats ---", flush=True)
            print(_json.dumps(out.get("graph_stats"), ensure_ascii=False, indent=2), flush=True)
        return 0

    if args.qa_json:
        out = args.output or os.devnull
        summ = args.summary_out or (args.qa_json + ".summary.json")
        process_single_qa(
            qa_file_path=args.qa_json,
            graph_file_path=args.graph_pkl,
            tag_file_path=args.tags_json,
            output_file_path=out,
            summary_file_path=summ,
            max_questions=args.max_questions,
            method=method,
            resume=bool(getattr(args, "resume", False)),
        )
        return 0

    print("请指定 --question 或 --qa-json", file=sys.stderr)
    return 2


def cmd_run(args: argparse.Namespace) -> int:
    """构图后使用同一 --graph-out / --tags-out 做查询（若提供问题）。"""
    rc = cmd_build(args)
    if rc != 0:
        return rc
    if not args.question and not args.qa_json:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        msg = "构图结束；未指定 --question / --qa-json，已跳过查询。"
        if tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg)
        return 0

    if not args.graph_out or not args.tags_out:
        print("run 模式在查询前需要构图产物：请同时指定 --graph-out 与 --tags-out", file=sys.stderr)
        return 2

    qargs = argparse.Namespace(
        verbose=args.verbose,
        graph_pkl=args.graph_out,
        tags_json=args.tags_out,
        method=args.method,
        question=args.question,
        qa_json=args.qa_json,
        summary_out=args.summary_out,
        output=args.output,
        max_questions=args.max_questions,
        resume=getattr(args, "resume", False),
        log_prompt=getattr(args, "log_prompt", None),
        no_show_retrieval=getattr(args, "no_show_retrieval", False),
    )
    return cmd_query(qargs)


def cmd_chat(args: argparse.Namespace) -> int:
    _apply_logging(args.verbose)
    _apply_log_prompt_arg(args, default_beside=args.graph_pkl)
    from src.assist import load_mosaic_memory_pickle
    from src.query import query

    memory = load_mosaic_memory_pickle(args.graph_pkl)
    memory.process_kw(args.tags_json)
    method = args.method or "hash"

    print("已加载图与 tags；输入问题回车提交，空行退出。")
    while True:
        try:
            line = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        ans = query(line, memory, method=method)
        print(ans)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mosaic",
        description="Mosaic：构图（save 批处理）/ 读图查询 / 整套流水线 / 交互问答",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细日志：控制台输出 DEBUG（含 TF-IDF/检索细节）；默认仅 WARNING，构图进度见 tqdm",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="从 LoCoMo 对话 JSON 构图（与 src/save.py 一致，每批最多 10 条消息）")
    pb.add_argument("--conv-json", required=True, help="对话 JSON 路径")
    pb.add_argument("--conv-name", default=None, help="会话标识（默认从文件名推断）")
    pb.add_argument("--graph-save-dir", default=None, help="图快照目录（默认环境变量 GRAPH_SAVE_DIR 或 results/graph）")
    pb.add_argument(
        "--hash",
        action="store_true",
        help="强制 hash_only：仅用 TF-IDF/hash 构图（覆盖 [BUILD] / MOSAIC_BUILD_MODE）",
    )
    pb.add_argument("--graph-out", default=None, help="将最终图 pickle 到此路径（LLM 构图时可选）")
    pb.add_argument("--tags-out", default=None, help="构图后生成 TF-IDF tags JSON")
    pb.add_argument("--progress-file", default=None, help="写入批次进度（同 MOSAIC_PROGRESS_FILE）")
    pb.add_argument(
        "--checkpoint",
        default=None,
        help="构图断点 pickle 路径（默认: 与 --graph-out 同名的 .build_state.pkl；可用 MOSAIC_BUILD_CHECKPOINT）",
    )
    pb.add_argument(
        "--resume",
        action="store_true",
        help="从断点继续构图（需对话 JSON 与断点指纹一致；每批自动落盘）",
    )
    pb.add_argument(
        "--log-prompt",
        nargs="?",
        const="__AUTO__",
        default=None,
        metavar="PATH",
        help="LLM I/O JSONL（省略 PATH 时写入 --graph-out 或 GRAPH_SAVE_DIR 上级目录 llm_io.jsonl）",
    )
    pb.set_defaults(_fn=cmd_build)

    pq = sub.add_parser(
        "query",
        help="加载 graph_network*.pkl 与 tags JSON 后查询（需先构图或提供文件）",
    )
    pq.add_argument("--graph-pkl", required=True, help="构图导出的 .pkl")
    pq.add_argument("--tags-json", required=True, help="tags JSON（generate_tags_tfidf 产出）")
    pq.add_argument("--method", choices=("llm", "hash"), default="hash", help="检索方式")
    g = pq.add_mutually_exclusive_group(required=True)
    g.add_argument("--question", default=None, help="单个问题")
    g.add_argument("--qa-json", dest="qa_json", default=None, help="LoCoMo QA JSON 批量评测")
    pq.add_argument("--summary-out", default=None, help="批量评测汇总 JSON 路径")
    pq.add_argument("--output", default=None, help="批量评测占位输出路径（可选）")
    pq.add_argument("--max-questions", type=int, default=None, help="最多评测题数")
    pq.add_argument(
        "--resume",
        action="store_true",
        help="QA 断点续跑：若存在 <output>.partial.json 则跳过已评测题并追加",
    )
    pq.add_argument(
        "--log-prompt",
        nargs="?",
        const="__AUTO__",
        default=None,
        metavar="PATH",
        help="LLM I/O JSONL（省略 PATH 时写入 graph-pkl 同目录 llm_io.jsonl）",
    )
    pq.add_argument(
        "--no-show-retrieval",
        action="store_true",
        help="单题 query 时不打印 retrieved_context / timing / graph_stats",
    )
    pq.set_defaults(_fn=cmd_query)

    pr = sub.add_parser("run", help="执行 build 后立刻用 --graph-out / --tags-out 做 query")
    pr.add_argument("--conv-json", required=True)
    pr.add_argument("--conv-name", default=None)
    pr.add_argument("--graph-save-dir", default=None)
    pr.add_argument("--hash", action="store_true")
    pr.add_argument("--graph-out", required=True, help="构图后 pickle 路径（供查询加载）")
    pr.add_argument("--tags-out", required=True, help="构图后 tags 路径（供查询加载）")
    pr.add_argument("--progress-file", default=None)
    pr.add_argument("--method", choices=("llm", "hash"), default="hash")
    pr.add_argument("--question", default=None)
    pr.add_argument("--qa-json", default=None)
    pr.add_argument("--summary-out", default=None)
    pr.add_argument("--output", default=None)
    pr.add_argument("--max-questions", type=int, default=None)
    pr.add_argument("--checkpoint", default=None, help="构图断点路径（默认由 --graph-out 推导）")
    pr.add_argument("--resume", action="store_true", help="构图与（若指定 qa）QA 均尝试断点续跑")
    pr.add_argument(
        "--no-show-retrieval",
        action="store_true",
        help="单题 query 时不打印检索摘要（与 query 子命令一致）",
    )
    pr.add_argument(
        "--log-prompt",
        nargs="?",
        const="__AUTO__",
        default=None,
        metavar="PATH",
        help="LLM I/O JSONL（省略 PATH 时写入 --graph-out 同目录 llm_io.jsonl）",
    )
    pr.set_defaults(_fn=cmd_run)

    pc = sub.add_parser("chat", help="加载图与 tags 后交互式问答（readline）")
    pc.add_argument("--graph-pkl", required=True)
    pc.add_argument("--tags-json", required=True)
    pc.add_argument("--method", choices=("llm", "hash"), default="hash")
    pc.add_argument(
        "--log-prompt",
        nargs="?",
        const="__AUTO__",
        default=None,
        metavar="PATH",
        help="LLM I/O JSONL",
    )
    pc.set_defaults(_fn=cmd_chat)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    fn = getattr(args, "_fn", None)
    if fn is None:
        parser.print_help()
        return 2
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
