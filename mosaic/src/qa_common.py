"""Shared QA evaluation logic for class-graph and instance-graph query scripts."""
from __future__ import annotations

import json
from collections import defaultdict
from string import Template
from typing import Any, Callable, Dict, List

from src.assist import fetch_default_llm_model
from src.logger import setup_logger
from src.prompts_en import JUDGE_ANSWER

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

_logger = setup_logger("qa_common")


def judge_answer_llm(question: str, gold_answer: str, generated_answer: str) -> str:
    """Use LLM to label generated answer as CORRECT or WRONG."""
    prompt = Template(JUDGE_ANSWER).substitute(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    llm = fetch_default_llm_model()
    response = llm.invoke(prompt)
    _logger.info("JUDGE_ANSWER_RESPONSE %s", response.content)
    judge_ans = json.loads(response.content)
    return (judge_ans.get("label") or "").strip().upper()


def run_qa_loop(
    questions: List[Dict[str, Any]],
    memory: Any,
    query_fn: Callable[[str, Any], str],
    *,
    skip_category: int = 5,
    max_questions: int | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> tuple[List[Dict], Dict, List[Dict]]:
    """
    Run QA over a list of questions. Returns (qa_results, category_stats, error_records).
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

            answer = query_fn(question, memory)
            _logger.info("expected_answer: %s; answer: %s", expected, answer)

            label = judge_answer_llm(question, expected, answer)
            qa_item = {**qa_item, "generated_answer": answer, "judgment": label}
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
