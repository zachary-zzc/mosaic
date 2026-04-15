#!/usr/bin/env python3
"""
Background test for conv5 multi-hop QA — rebuild graph with new features
(Option 1: inference enrichment + Option 3: BGE class sensing) then evaluate.

Usage (from project root):
  # Full pipeline: rebuild + evaluate
  python experiments/temp/test_conv5.py

  # Evaluate only (reuse existing graph)
  python experiments/temp/test_conv5.py --skip-build

  # Build only
  python experiments/temp/test_conv5.py --skip-qa

  # Evaluate existing graph at a custom path
  python experiments/temp/test_conv5.py --skip-build --graph PATH

Output goes to experiments/temp/conv5_results/
"""
import argparse
import json
import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic", "src"))

CONV_NAME = "conv5"
CONV_JSON = os.path.join(PROJECT_ROOT, "dataset", "locomo", "locomo_conv5.json")
QA_JSON = os.path.join(PROJECT_ROOT, "dataset", "locomo", "qa_5.json")

# Existing artifacts (baseline)
EXISTING_RUN = os.path.join(
    PROJECT_ROOT, "experiments", "locomo", "benchmark", "runs", "runs", CONV_NAME
)
EXISTING_GRAPH_HYBRID = os.path.join(
    EXISTING_RUN, "artifacts", "hybrid", f"graph_network_{CONV_NAME}.pkl"
)
EXISTING_TAGS_HYBRID = os.path.join(
    EXISTING_RUN, "artifacts", "hybrid", f"{CONV_NAME}_tags.json"
)
EXISTING_GRAPH_HASH = os.path.join(
    EXISTING_RUN, "artifacts", "hash_only", f"graph_network_{CONV_NAME}.pkl"
)
EXISTING_TAGS_HASH = os.path.join(
    EXISTING_RUN, "artifacts", "hash_only", f"{CONV_NAME}_tags.json"
)

# New output
OUT_DIR = os.path.join(os.path.dirname(__file__), "conv5_results")
LOG_DIR = os.path.join(OUT_DIR, "logs")

from openai import OpenAI
from src.data.graph import ClassGraph
from src.assist import load_mosaic_memory_pickle, read_to_file_json
from src.config_loader import get_api_key_and_base_url

api_key, base_url = get_api_key_and_base_url()
client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
MODEL = os.environ.get("MOSAIC_EVAL_MODEL", "qwen-plus")

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


def llm_call(prompt, timeout=30):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
        timeout=timeout,
    )
    return r.choices[0].message.content.strip()


def build_graph(strategy):
    """Build graph for conv5 with the current code (includes Option 1 + 3)."""
    import subprocess

    art_dir = os.path.join(OUT_DIR, "artifacts", strategy)
    os.makedirs(os.path.join(art_dir, "graph_snapshots"), exist_ok=True)

    graph_pkl = os.path.join(art_dir, f"graph_network_{CONV_NAME}.pkl")
    tags_json = os.path.join(art_dir, f"{CONV_NAME}_tags.json")

    os.environ["MOSAIC_LLM_IO_LOG"] = os.path.join(LOG_DIR, f"llm_io_{strategy}.jsonl")
    os.environ["MOSAIC_INGEST_JSONL"] = os.path.join(LOG_DIR, f"ingest_{strategy}.jsonl")

    cmd = [
        sys.executable, "-m", "mosaic", "build",
        "--conv-json", CONV_JSON,
        "--conv-name", CONV_NAME,
        "--graph-save-dir", os.path.join(art_dir, "graph_snapshots"),
        "--graph-out", graph_pkl,
        "--tags-out", tags_json,
        "--progress-file", os.path.join(LOG_DIR, f"{CONV_NAME}_progress_{strategy}.txt"),
        "--log-prompt", os.path.join(LOG_DIR, f"llm_io_{strategy}.jsonl"),
    ]
    if strategy == "hash_only":
        cmd.append("--hash")

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    print(f"\n  [build] {CONV_NAME} strategy={strategy}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [build] FAILED (exit {result.returncode})")
        return None, None, elapsed

    print(f"  [build] done in {elapsed:.0f}s")
    return graph_pkl, tags_json, elapsed


def evaluate(graph_pkl, tags_json, questions, label, search_method="hash"):
    """Evaluate multi-hop questions against a graph."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Graph: {os.path.basename(graph_pkl)}")
    print(f"  Questions: {len(questions)} (multi-hop)")
    print(f"  Search: {search_method}")
    print(f"{'='*60}")
    sys.stdout.flush()

    memory = load_mosaic_memory_pickle(graph_pkl)
    memory.process_kw(tags_json)

    correct = 0
    details = []
    for i, q in enumerate(questions):
        question = q["question"]
        gold = q["answer"]
        t0 = time.time()
        try:
            if search_method == "llm":
                ctx, trace = memory._search_by_sub_llm(question)
            else:
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
        bge_cls = trace.get("bge_class_sensing", "?") if isinstance(trace, dict) else "?"
        mark = "OK" if is_correct else "XX"
        print(f"  [{mark}] Q{i+1}: {question[:65]}  ({elapsed:.1f}s)")
        print(f"       Gold: {gold[:70]}")
        print(f"       Pred: {str(answer)[:70]}")
        print(f"       Judge={judgment}  Neighbors={n_neigh}")
        sys.stdout.flush()

        details.append({
            "question": question, "gold": gold, "pred": answer,
            "judgment": judgment, "neighbors": n_neigh,
            "elapsed": round(elapsed, 1),
        })

    acc = correct / len(questions) * 100 if questions else 0
    print(f"\n  >> {label}: {correct}/{len(questions)} = {acc:.1f}%\n")
    sys.stdout.flush()
    return correct, len(questions), details


def main():
    parser = argparse.ArgumentParser(description=f"Test conv5 with new features")
    parser.add_argument("--skip-build", action="store_true", help="Skip graph building, evaluate only")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA evaluation, build only")
    parser.add_argument("--graph", type=str, default=None, help="Custom graph path for eval")
    parser.add_argument("--tags", type=str, default=None, help="Custom tags path for eval")
    parser.add_argument("--strategy", choices=["hybrid", "hash_only", "both"], default="both",
                        help="Which strategy to build/eval")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print(f"  Conv5 Test — Option 1 (Inference) + Option 3 (BGE Class Sensing)")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: {MODEL}")
    print("=" * 60)

    # Load QA and filter multi-hop
    questions = read_to_file_json(QA_JSON)
    mh = [q for q in questions if q.get("category") == 3]
    all_q = questions
    print(f"  Total questions: {len(all_q)}, Multi-hop: {len(mh)}")

    strategies = ["hybrid", "hash_only"] if args.strategy == "both" else [args.strategy]
    results = {}

    for strategy in strategies:
        print(f"\n{'#'*60}")
        print(f"  STRATEGY: {strategy}")
        print(f"{'#'*60}")

        # --- Build ---
        if not args.skip_build:
            graph_pkl, tags_json, build_time = build_graph(strategy)
            if graph_pkl is None:
                print(f"  [SKIP] Build failed for {strategy}")
                continue
            results[f"{strategy}_build_time"] = build_time
        else:
            if args.graph:
                graph_pkl = args.graph
                tags_json = args.tags or EXISTING_TAGS_HYBRID
            elif strategy == "hybrid":
                graph_pkl = EXISTING_GRAPH_HYBRID
                tags_json = EXISTING_TAGS_HYBRID
            else:
                graph_pkl = EXISTING_GRAPH_HASH
                tags_json = EXISTING_TAGS_HASH
            print(f"  [skip-build] Using existing graph: {graph_pkl}")

        if not os.path.isfile(graph_pkl):
            print(f"  [SKIP] Graph not found: {graph_pkl}")
            continue

        # --- Evaluate ---
        if not args.skip_qa:
            # Multi-hop only
            c, t, det = evaluate(graph_pkl, tags_json, mh,
                                 f"{strategy.upper()} — Multi-hop", "hash")
            results[f"{strategy}_multihop"] = {"correct": c, "total": t, "acc": c/t*100 if t else 0}

            # Save detailed results
            detail_path = os.path.join(OUT_DIR, f"{CONV_NAME}_{strategy}_multihop_details.json")
            with open(detail_path, "w") as f:
                json.dump({"correct": c, "total": t, "acc": c/t*100 if t else 0,
                           "details": det}, f, indent=2, ensure_ascii=False)
            print(f"  Details saved: {detail_path}")

            # All categories
            c_all, t_all, det_all = evaluate(graph_pkl, tags_json, all_q,
                                              f"{strategy.upper()} — All categories", "hash")
            results[f"{strategy}_all"] = {"correct": c_all, "total": t_all, "acc": c_all/t_all*100 if t_all else 0}

            all_path = os.path.join(OUT_DIR, f"{CONV_NAME}_{strategy}_all_details.json")
            with open(all_path, "w") as f:
                json.dump({"correct": c_all, "total": t_all, "acc": c_all/t_all*100 if t_all else 0,
                           "details": det_all}, f, indent=2, ensure_ascii=False)

    # --- Compare with baseline ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for key, val in sorted(results.items()):
        if isinstance(val, dict):
            print(f"  {key}: {val['correct']}/{val['total']} = {val['acc']:.1f}%")
        else:
            print(f"  {key}: {val:.1f}s")

    summary_path = os.path.join(OUT_DIR, f"{CONV_NAME}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
