"""Fair regression check: original 1-hop vs our 2-hop on the 17 correct queries."""
import os, sys, json, pickle
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

from src.assist import build_instance_fragments
from src.data.graph import ClassGraph

BASE = "/Users/zachary/Workspace/LongtermMemory"
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg = pickle.load(f)
cg.process_kw(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json")

with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/results/qa_5_eval_full_hybrid.json") as f:
    eval_data = json.load(f)

inst_text = {}
for node in cg.graph.nodes:
    cid = getattr(node, "class_id", "")
    for inst in getattr(node, "_instances", []) or []:
        iid = inst.get("instance_id", "")
        eid = f"{cid}:{iid}"
        frags = build_instance_fragments(inst)
        text = "\n".join(t for _ty, t in frags if t and t.strip()).lower()
        inst_text[eid] = text

cat1_correct = [r for r in eval_data["results"] if r["category"] == 1 and r["judgment"] == "CORRECT"]

def count_answer(answer, trace):
    gold_words = [w.strip(".,!?()[]\"'") for w in answer.lower().split() if len(w.strip(".,!?()[]\"'")) > 3]
    answer_eids = set()
    for eid, text in inst_text.items():
        found = [w for w in gold_words if w in text]
        if len(found) >= max(1, len(gold_words) * 0.3):
            answer_eids.add(eid)
    retrieved = set(trace.get("retrieved_entity_ids", []))
    neigh = set(trace.get("neighbor_expansion", {}).get("entity_ids", []))
    return len(answer_eids & (retrieved | neigh))

# Config A: original 1-hop
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "1"
os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = "16"
results_a = {}
for r in cat1_correct:
    cg._adj_cache = None
    _, trace = cg._search_by_sub_hash(r["question"])
    results_a[r["qa_source_index"]] = count_answer(r["answer"], trace)

# Config B: our 2-hop
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "2"
results_b = {}
for r in cat1_correct:
    cg._adj_cache = None
    _, trace = cg._search_by_sub_hash(r["question"])
    results_b[r["qa_source_index"]] = count_answer(r["answer"], trace)

better = worse = same = 0
for idx in sorted(results_a):
    a, b = results_a[idx], results_b[idx]
    delta = b - a
    mark = "+" if delta > 0 else ("-" if delta < 0 else "=")
    if delta > 0: better += 1
    elif delta < 0: worse += 1
    else: same += 1
    if delta != 0:
        print(f"  Q{idx+1}: 1-hop={a}, 2-hop={b} [{mark}]")
print(f"\nBetter: {better}, Worse: {worse}, Same: {same}")
