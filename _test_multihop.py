#!/usr/bin/env python3
"""
Multi-hop ONLY accuracy test: original vs patched graph.
Only 13 questions — fast enough to finish reliably.
"""
import json, os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic", "src"))

from src.data.graph import ClassGraph
from src.assist import fetch_default_llm_model, load_mosaic_memory_pickle, query_question, read_to_file_json
from src.prompts_en import PROMPT_QUERY_TEMPLATE
from src.qa_common import judge_answer_llm_timed

QA_PATH = os.path.join(PROJECT_ROOT, "dataset/locomo/qa_0.json")
TAGS_PATH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/conv0_tags.json")
ORIGINAL_GRAPH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0.pkl")
PATCHED_GRAPH  = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0_patched.pkl")

def evaluate(graph_path, questions, label):
    print(f"\n{'='*60}")
    print(f"  {label}  ({os.path.basename(graph_path)})")
    print(f"  {len(questions)} multi-hop questions")
    print(f"{'='*60}")

    memory = load_mosaic_memory_pickle(graph_path)
    memory.process_kw(TAGS_PATH)
    llm = fetch_default_llm_model()

    correct = 0
    details = []
    for i, q in enumerate(questions):
        question = q["question"]
        gold = q["answer"]
        try:
            ctx, trace = memory._search_by_sub_hash(question)
            answer = query_question(llm, question, ctx, PROMPT_QUERY_TEMPLATE)
            judgment, ms = judge_answer_llm_timed(question, gold, answer)
        except Exception as e:
            answer = f"ERROR: {e}"
            judgment = "ERROR"
            trace = {}

        is_correct = judgment == "CORRECT"
        if is_correct:
            correct += 1

        n_tfidf = trace.get("tfidf_hits", {}).get("count", "?") if isinstance(trace, dict) else "?"
        n_neigh = trace.get("neighbor_expansion", {}).get("count", "?") if isinstance(trace, dict) else "?"

        mark = "OK" if is_correct else "XX"
        print(f"  [{mark}] Q{i+1}: {question[:70]}")
        print(f"       Gold: {gold[:70]}")
        print(f"       Pred: {str(answer)[:70]}")
        print(f"       Judge={judgment}  TF-IDF={n_tfidf}  Neighbors={n_neigh}")

        details.append({"q": question, "gold": gold, "pred": answer,
                        "judgment": judgment, "tfidf": n_tfidf, "neigh": n_neigh})

    acc = correct / len(questions) * 100 if questions else 0
    print(f"\n  >> {label}: {correct}/{len(questions)} = {acc:.1f}%\n")
    return correct, len(questions), details

def main():
    questions = read_to_file_json(QA_PATH)
    mh = [q for q in questions if q.get("category") == 3]
    print(f"Multi-hop questions: {len(mh)}")

    o_c, o_t, o_det = evaluate(ORIGINAL_GRAPH, mh, "ORIGINAL")
    p_c, p_t, p_det = evaluate(PATCHED_GRAPH,  mh, "PATCHED")

    print("=" * 60)
    print("FINAL COMPARISON  (Multi-hop only)")
    print("=" * 60)
    print(f"  Original: {o_c}/{o_t} = {o_c/o_t*100:.1f}%")
    print(f"  Patched:  {p_c}/{p_t} = {p_c/p_t*100:.1f}%")
    delta = (p_c - o_c)
    print(f"  Delta:    {'+' if delta >= 0 else ''}{delta} questions")

    # Per-question diff
    changed = []
    for i in range(len(mh)):
        oj = o_det[i]["judgment"]
        pj = p_det[i]["judgment"]
        if oj != pj:
            changed.append((i+1, oj, pj, mh[i]["question"][:60]))
    if changed:
        print("\n  Questions that changed:")
        for idx, oj, pj, q in changed:
            print(f"    Q{idx}: {oj} -> {pj}  | {q}")
    else:
        print("\n  No questions changed judgment.")

    # Save results
    out = {"original": {"correct": o_c, "total": o_t, "details": o_det},
           "patched":  {"correct": p_c, "total": p_t, "details": p_det}}
    outpath = os.path.join(PROJECT_ROOT, "_multihop_accuracy.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {outpath}")

if __name__ == "__main__":
    main()
