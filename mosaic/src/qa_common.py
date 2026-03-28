"""Shared QA evaluation logic for class-graph and instance-graph query scripts."""
from __future__ import annotations

import re
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


def judge_answer_llm(question: str, gold_answer: str, generated_answer: str) -> str:
    """Use LLM to label generated answer as CORRECT or WRONG."""
    prompt = Template(JUDGE_ANSWER).substitute(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    llm = fetch_default_llm_model()
    response = llm.invoke(prompt)
    _logger.debug("JUDGE_ANSWER_RESPONSE %s", response.content)
    return _label_from_judge_response(
        getattr(response, "content", None) or str(response)
    )


def run_qa_loop(
    questions: List[Dict[str, Any]],
    memory: Any,
    query_fn: Callable[[str, Any], Any],
    *,
    skip_category: int = 5,
    max_questions: int | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> tuple[List[Dict], Dict, List[Dict]]:
    """
    Run QA over a list of questions. Returns (qa_results, category_stats, error_records).

    ``query_fn`` 可返回 ``str``，或含 ``answer`` 键的 ``dict``（可带 ``retrieved_context``、``graph_stats``，E-1）。
    """
    if max_questions is not None:
        questions = questions[:max_questions]

    qa_results: List[Dict] = []
    category_stats: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "wrong": 0, "total": 0}
    )
    total_correct = 0
    total_wrong = 0
    total_count = 0
    error_records: List[Dict] = []

    iterator = tqdm(questions, desc="QA", unit="q")
    for i, qa_item in enumerate(iterator):
        try:
            question = qa_item.get("question", "")
            category = qa_item.get("category", 0)
            if category == skip_category:
                continue

            expected = qa_item.get("adversarial_answer") or qa_item.get("answer", "")
            if not isinstance(expected, str):
                expected = str(expected).strip()

            raw_out = query_fn(question, memory)
            if isinstance(raw_out, dict) and "answer" in raw_out:
                answer = raw_out["answer"]
                rctx = raw_out.get("retrieved_context")
                gstats = raw_out.get("graph_stats")
            else:
                answer = raw_out
                rctx = None
                gstats = None
            _logger.debug("expected_answer: %s; answer: %s", expected, answer)

            label = judge_answer_llm(question, expected, answer)
            qa_item = {**qa_item, "generated_answer": answer, "judgment": label}
            if rctx is not None:
                qa_item["retrieved_context"] = rctx
            if gstats is not None:
                qa_item["graph_stats"] = gstats
            qa_results.append(qa_item)

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
        except Exception as e:
            error_records.append({
                "question_index": i + 1,
                "question": qa_item.get("question", "未知问题"),
                "error_type": type(e).__name__,
                "error_message": str(e),
            })
            _logger.exception("处理问题 %s 时出错", i + 1)

    return qa_results, dict(category_stats), error_records
