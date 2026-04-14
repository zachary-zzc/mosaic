#!/usr/bin/env python3
"""
Targeted test: run QA only on multi-hop (cat 3) and open-domain (cat 4) questions
using the patched graph vs original graph, to verify the cross-class edge fix.
"""
import json
import os
import sys
import pickle
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic", "src"))

from src.data.graph import ClassGraph
from src.assist import fetch_default_llm_model, load_mosaic_memory_pickle, query_question, read_to_file_json
from src.prompts_en import PROMPT_QUERY_TEMPLATE
from src.qa_common import judge_answer_llm_timed
from src.logger import setup_logger

_logger = setup_logger("targeted_test")

QA_PATH = os.path.join(PROJECT_ROOT, "dataset/locomo/qa_0.json")
TAGS_PATH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/conv0_tags.json")
ORIGINAL_GRAPH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0.pkl")
PATCHED_GRAPH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0_patched.pkl")

TARGET_CATEGORIES = {3, 4}  # multi-hop and open-domain
CAT_NAMES = {1: "Single-hop", 2: "Temporal", 3: "Multi-hop", 4: "Open-domain"}


def load_target_questions(qa_path):
    """Load only multi-hop and open-domain questions."""
    questions = read_to_file_json(qa_path)
    filtered = [q for q in questions if q.get("category") in TARGET_CATEGORIES]
    return filtered


def run_single_question(memory, question_text, method="hash"):
    """Retrieve context and generate answer for a single question."""
    ctx, trace = memory._search_by_sub_hash(question_text)
    llm = fetch_default_llm_model()
    answer = query_question(llm, question_text, ctx, PROMPT_QUERY_TEMPLATE)
    return answer, trace


def evaluate_questions(graph_path, tags_path, questions, label):
    """Run QA + judge on a list of questions using the given graph."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Graph: {os.path.basename(graph_path)}")
    print(f"Questions: {len(questions)} (multi-hop + open-domain)")
    print(f"{'='*60}")

    memory = load_mosaic_memory_pickle(graph_path)
    memory.process_kw(tags_path)

    results = []
    cat_stats = {3: {"correct": 0, "total": 0}, 4: {"correct": 0, "total": 0}}

    for i, q in enumerate(questions):
        cat = q["category"]
        question_text = q["question"]
        gold = q["answer"]

        print(f"\n[{i+1}/{len(questions)}] Cat={CAT_NAMES[cat]} | Q: {question_text[:80]}")

        try:
            answer, trace = run_single_question(memory, question_text)
            judgment, judge_ms = judge_answer_llm_timed(question_text, gold, answer)
        except Exception as e:
            print(f"  ERROR: {e}")
            answer = ""
            judgment = "WRONG"
            trace = {}
            judge_ms = 0

        cat_stats[cat]["total"] += 1
        if judgment == "CORRECT":
            cat_stats[cat]["correct"] += 1

        tfidf = trace.get("tfidf_hits", {}).get("count", "?") if isinstance(trace, dict) else "?"
        neighbor = trace.get("neighbor_expansion", {}).get("count", "?") if isinstance(trace, dict) else "?"

        print(f"  Gold: {gold[:80]}")
        print(f"  Pred: {answer[:80]}")
        print(f"  Judge: {judgment} | TF-IDF: {tfidf}, Neighbors: {neighbor}")

        results.append({
            "question": question_text,
            "category": cat,
            "gold": gold,
            "predicted": answer,
            "judgment": judgment,
            "tfidf_hits": tfidf,
            "neighbor_expansion": neighbor,
        })

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {label}")
    print(f"{'='*60}")
    total_correct = 0
    total_all = 0
    for cat in [3, 4]:
        c = cat_stats[cat]["correct"]
        t = cat_stats[cat]["total"]
        acc = c / t * 100 if t else 0
        total_correct += c
        total_all += t
        print(f"  {CAT_NAMES[cat]}: {c}/{t} = {acc:.1f}%")
    overall = total_correct / total_all * 100 if total_all else 0
    print(f"  Overall (cat 3+4): {total_correct}/{total_all} = {overall:.1f}%")

    return results, cat_stats


def main():
    questions = load_target_questions(QA_PATH)
    print(f"Loaded {len(questions)} target questions")
    for cat in [3, 4]:
        n = sum(1 for q in questions if q["category"] == cat)
        print(f"  {CAT_NAMES[cat]}: {n}")

    # Test both graphs
    orig_results, orig_stats = evaluate_questions(ORIGINAL_GRAPH, TAGS_PATH, questions, "ORIGINAL (no patch)")
    patch_results, patch_stats = evaluate_questions(PATCHED_GRAPH, TAGS_PATH, questions, "PATCHED (cross-class edges)")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON: Original vs Patched")
    print(f"{'='*60}")
    for cat in [3, 4]:
        o_c = orig_stats[cat]["correct"]
        o_t = orig_stats[cat]["total"]
        p_c = patch_stats[cat]["correct"]
        p_t = patch_stats[cat]["total"]
        o_acc = o_c / o_t * 100 if o_t else 0
        p_acc = p_c / p_t * 100 if p_t else 0
        delta = p_acc - o_acc
        print(f"  {CAT_NAMES[cat]}: {o_acc:.1f}% -> {p_acc:.1f}% ({'+' if delta >= 0 else ''}{delta:.1f}pp)")

    o_total_c = sum(orig_stats[c]["correct"] for c in [3, 4])
    o_total_t = sum(orig_stats[c]["total"] for c in [3, 4])
    p_total_c = sum(patch_stats[c]["correct"] for c in [3, 4])
    p_total_t = sum(patch_stats[c]["total"] for c in [3, 4])
    o_overall = o_total_c / o_total_t * 100 if o_total_t else 0
    p_overall = p_total_c / p_total_t * 100 if p_total_t else 0
    delta = p_overall - o_overall
    print(f"  Overall: {o_overall:.1f}% -> {p_overall:.1f}% ({'+' if delta >= 0 else ''}{delta:.1f}pp)")

    # Save comparison
    out_path = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/results/targeted_comparison.json")
    comparison = {
        "original": orig_stats,
        "patched": patch_stats,
        "original_results": orig_results,
        "patched_results": patch_results,
    }
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
