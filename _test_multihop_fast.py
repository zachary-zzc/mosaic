#!/usr/bin/env python3
"""
Multi-hop accuracy test using direct OpenAI calls (faster than LangChain wrapper).
Tests both original and patched graph on 13 multi-hop questions.
"""
import json, os, sys, time
from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic", "src"))

from src.data.graph import ClassGraph
from src.assist import load_mosaic_memory_pickle, read_to_file_json
from src.config_loader import get_api_key_and_base_url

QA_PATH = os.path.join(PROJECT_ROOT, "dataset/locomo/qa_0.json")
TAGS_PATH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/conv0_tags.json")
ORIGINAL_GRAPH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0.pkl")
PATCHED_GRAPH  = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0_patched.pkl")

api_key, base_url = get_api_key_and_base_url()
client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
MODEL = "qwen-plus"

def llm_call(prompt, timeout=30):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
        timeout=timeout,
    )
    return r.choices[0].message.content.strip()

ANSWER_PROMPT = """Based on the following retrieved memory snippets, answer the question.
If the snippets do not contain enough information, answer based on what is available.

[Retrieved Memory Snippets]
{context}

[Question]
{question}

Answer concisely."""

JUDGE_PROMPT = """You are a judge evaluating if a generated answer is correct compared to the gold answer.
The generated answer may use synonyms, paraphrases, or different wording.
As long as the core factual content or sentiment is equivalent, it should be CORRECT.

Question: {question}
Gold Answer: {gold}
Generated Answer: {pred}

Reply with exactly one word: CORRECT or WRONG"""

def evaluate(graph_path, questions, label):
    print(f"\n{'='*60}")
    print(f"  {label}  ({os.path.basename(graph_path)})")
    print(f"  {len(questions)} multi-hop questions")
    print(f"{'='*60}")
    sys.stdout.flush()

    memory = load_mosaic_memory_pickle(graph_path)
    memory.process_kw(TAGS_PATH)

    correct = 0
    details = []
    for i, q in enumerate(questions):
        question = q["question"]
        gold = q["answer"]
        t0 = time.time()
        try:
            ctx, trace = memory._search_by_sub_hash(question)
            prompt = ANSWER_PROMPT.format(context=ctx[:8000], question=question)
            answer = llm_call(prompt)
            judge_prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=answer)
            judgment = llm_call(judge_prompt, timeout=15)
            judgment = "CORRECT" if "CORRECT" in judgment.upper() else "WRONG"
        except Exception as e:
            answer = f"ERROR: {e}"
            judgment = "ERROR"
            trace = {}

        elapsed = time.time() - t0
        is_correct = judgment == "CORRECT"
        if is_correct:
            correct += 1

        n_neigh = trace.get("neighbor_expansion", {}).get("count", "?") if isinstance(trace, dict) else "?"
        mark = "OK" if is_correct else "XX"
        print(f"  [{mark}] Q{i+1}: {question[:65]}  ({elapsed:.1f}s)")
        print(f"       Gold: {gold[:70]}")
        print(f"       Pred: {str(answer)[:70]}")
        print(f"       Judge={judgment}  Neighbors={n_neigh}")
        sys.stdout.flush()

        details.append({"q": question, "gold": gold, "pred": answer,
                        "judgment": judgment, "neigh": n_neigh})

    acc = correct / len(questions) * 100 if questions else 0
    print(f"\n  >> {label}: {correct}/{len(questions)} = {acc:.1f}%\n")
    sys.stdout.flush()
    return correct, len(questions), details

def main():
    questions = read_to_file_json(QA_PATH)
    mh = [q for q in questions if q.get("category") == 3]
    print(f"Multi-hop questions: {len(mh)}")
    print(f"Model: {MODEL}")
    sys.stdout.flush()

    o_c, o_t, o_det = evaluate(ORIGINAL_GRAPH, mh, "ORIGINAL")
    p_c, p_t, p_det = evaluate(PATCHED_GRAPH,  mh, "PATCHED")

    print("=" * 60)
    print("FINAL COMPARISON  (Multi-hop only)")
    print("=" * 60)
    print(f"  Original: {o_c}/{o_t} = {o_c/o_t*100:.1f}%")
    print(f"  Patched:  {p_c}/{p_t} = {p_c/p_t*100:.1f}%")
    delta = p_c - o_c
    print(f"  Delta:    {'+' if delta >= 0 else ''}{delta} questions")

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

    out = {"original": {"correct": o_c, "total": o_t, "details": o_det},
           "patched":  {"correct": p_c, "total": p_t, "details": p_det}}
    outpath = os.path.join(PROJECT_ROOT, "_multihop_accuracy.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {outpath}")

if __name__ == "__main__":
    main()
