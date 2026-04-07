"""Shared QA evaluation logic for class-graph and instance-graph query scripts."""
from __future__ import annotations

import re
import time
from collections import defaultdict
from string import Template
from typing import Any, Callable, Dict, List

from src.assist import fetch_default_llm_model
from src.logger import setup_logger
from src.prompts_en import JUDGE_ANSWER
from src.utils.io_utils import parse_llm_json_object

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

_logger = setup_logger("qa_common")


def _label_from_judge_response(content: str | None) -> str:
    """Parse judge JSON or infer CORRECT/WRONG from text; default WRONG if ambiguous."""
    raw = (content or "").strip()
    parsed = parse_llm_json_object(raw)
    if parsed:
        label = (parsed.get("label") or "").strip().upper()
        if label in ("CORRECT", "WRONG"):
            return label

    m = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 纯文本：整段或最后一行恰好为 CORRECT / WRONG（避免 “not correct” 误匹配）
    one = raw.strip().upper()
    if one in ("CORRECT", "WRONG"):
        return one
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        u = line.upper()
        if u in ("CORRECT", "WRONG"):
            return u

    _logger.warning(
        "无法从 judge 回复解析 label（非 JSON 或为空），记为 WRONG。原文前 500 字: %r",
        raw[:500],
    )
    return "WRONG"


def category_stats_from_qa_results(
    qa_results: List[Dict[str, Any]],
    *,
    skip_category: int = 5,
) -> Dict[int, Dict[str, int]]:
    """从已有评测结果重建 category_stats（用于断点续跑）。"""
    category_stats: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "wrong": 0, "total": 0}
    )
    for r in qa_results:
        category = r.get("category", 0)
        if category == skip_category:
            continue
        label = (r.get("judgment") or "WRONG").strip().upper()
        category_stats[category]["total"] += 1
        if label == "CORRECT":
            category_stats[category]["correct"] += 1
        elif label == "WRONG":
            category_stats[category]["wrong"] += 1
    return dict(category_stats)


def judge_answer_llm_timed(question: str, gold_answer: str, generated_answer: str) -> tuple[str, float]:
    """LLM 评判 CORRECT/WRONG，返回 (label, judge_llm_ms)。"""
    from src.llm.telemetry import llm_call_scope

    prompt = Template(JUDGE_ANSWER).substitute(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    llm = fetch_default_llm_model()
    t0 = time.perf_counter()
    with llm_call_scope("qa.judge"):
        response = llm.invoke(prompt)
    ms = (time.perf_counter() - t0) * 1000.0
    _logger.debug("JUDGE_ANSWER_RESPONSE %s", response.content)
    return _label_from_judge_response(
        getattr(response, "content", None) or str(response)
    ), ms


def judge_answer_llm(question: str, gold_answer: str, generated_answer: str) -> str:
    """Use LLM to label generated answer as CORRECT or WRONG."""
    label, _ms = judge_answer_llm_timed(question, gold_answer, generated_answer)
    return label


def run_qa_loop(
    questions: List[Dict[str, Any]],
    memory: Any,
    query_fn: Callable[[str, Any], Any],
    *,
    skip_category: int = 5,
    max_questions: int | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
    initial_qa_results: List[Dict[str, Any]] | None = None,
    initial_error_records: List[Dict[str, Any]] | None = None,
    partial_save: Callable[[List[Dict], Dict[int, Dict[str, int]], List[Dict]], None] | None = None,
) -> tuple[List[Dict], Dict, List[Dict]]:
    """
    Run QA over a list of questions. Returns (qa_results, category_stats, error_records).

    ``query_fn`` 可返回 ``str``，或含 ``answer`` 键的 ``dict``（可带 ``retrieved_context``、``graph_stats``，E-1）。

    ``initial_qa_results``：断点续跑时传入已完成的条目（须含 ``qa_source_index``，与当前 ``questions`` 枚举下标一致）。
    ``partial_save``：每完成一题或写入一条 error 后回调，便于落盘临时结果。
    """
    if max_questions is not None:
        questions = questions[:max_questions]

    qa_results: List[Dict] = list(initial_qa_results or [])
    done_indices = {
        int(r["qa_source_index"])
        for r in qa_results
        if r.get("qa_source_index") is not None
    }
    category_stats: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "wrong": 0, "total": 0}
    )
    merged_stats = category_stats_from_qa_results(qa_results, skip_category=skip_category)
    for cat, st in merged_stats.items():
        category_stats[cat] = {
            "correct": st["correct"],
            "wrong": st["wrong"],
            "total": st["total"],
        }
    total_correct = sum(1 for r in qa_results if r.get("judgment") == "CORRECT")
    total_wrong = sum(1 for r in qa_results if r.get("judgment") == "WRONG")
    total_count = total_correct + total_wrong
    error_records: List[Dict] = list(initial_error_records or [])

    iterator = tqdm(questions, desc="QA", unit="q")
    for i, qa_item in enumerate(iterator):
        try:
            question = qa_item.get("question", "")
            category = qa_item.get("category", 0)
            if category == skip_category:
                continue

            if i in done_indices:
                continue

            expected = qa_item.get("adversarial_answer") or qa_item.get("answer", "")
            if not isinstance(expected, str):
                expected = str(expected).strip()

            raw_out = query_fn(question, memory)
            if isinstance(raw_out, dict) and "answer" in raw_out:
                answer = raw_out["answer"]
                rctx = raw_out.get("retrieved_context")
                gstats = raw_out.get("graph_stats")
                qtimings = raw_out.get("timing_ms")
            else:
                answer = raw_out
                rctx = None
                gstats = None
                qtimings = None
            _logger.debug("expected_answer: %s; answer: %s", expected, answer)

            label, judge_ms = judge_answer_llm_timed(question, expected, answer)
            qa_item = {
                **qa_item,
                "generated_answer": answer,
                "judgment": label,
                "qa_source_index": i,
            }
            if rctx is not None:
                qa_item["retrieved_context"] = rctx
            if gstats is not None:
                qa_item["graph_stats"] = gstats
            timing: dict = {}
            if isinstance(qtimings, dict):
                timing.update(qtimings)
            timing["judge_llm_ms"] = round(judge_ms, 3)
            timing["qa_total_ms"] = round(
                float(timing.get("total_ms", 0.0)) + judge_ms,
                3,
            )
            qa_item["timing_ms"] = timing
            qa_results.append(qa_item)
            done_indices.add(i)

            total_count += 1
            if label == "CORRECT":
                total_correct += 1
            elif label == "WRONG":
                total_wrong += 1
            category_stats[category]["total"] += 1
            if label == "CORRECT":
                category_stats[category]["correct"] += 1
            elif label == "WRONG":
                category_stats[category]["wrong"] += 1

            if progress_callback and total_count % 10 == 0:
                acc = total_correct / total_count if total_count > 0 else 0.0
                progress_callback(total_count, total_correct, acc)
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(acc=f"{total_correct}/{total_count}")
            if partial_save is not None:
                partial_save(qa_results, dict(category_stats), error_records)
        except Exception as e:
            error_records.append({
                "question_index": i + 1,
                "question": qa_item.get("question", "未知问题"),
                "error_type": type(e).__name__,
                "error_message": str(e),
            })
            _logger.exception("处理问题 %s 时出错", i + 1)
            if partial_save is not None:
                partial_save(qa_results, dict(category_stats), error_records)

    return qa_results, dict(category_stats), error_records
