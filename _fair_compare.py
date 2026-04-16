"""Fair A/B comparison: original MMR (1-hop) vs our round-robin (2-hop).
Both run live, no comparison to stored results."""
import os, sys, json, pickle
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

from src.assist import build_instance_fragments
from src.data.graph import ClassGraph
import copy

BASE = "/Users/zachary/Workspace/LongtermMemory"

# Load graph
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg = pickle.load(f)
tag_path = f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json"
cg.process_kw(tag_path)

# Load eval to get failing questions
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/results/qa_5_eval_full_hybrid.json") as f:
    eval_data = json.load(f)

# Build instance text index
inst_text = {}
for node in cg.graph.nodes:
    cid = getattr(node, "class_id", "")
    for inst in getattr(node, "_instances", []) or []:
        iid = inst.get("instance_id", "")
        eid = f"{cid}:{iid}"
        frags = build_instance_fragments(inst)
        text = "\n".join(t for _ty, t in frags if t and t.strip()).lower()
        inst_text[eid] = text

cat1 = [r for r in eval_data["results"] if r["category"] == 1]
cat1_wrong = [r for r in cat1 if r["judgment"] == "WRONG"]

def count_answer_entities(question, answer, trace):
    gold_lower = answer.lower()
    gold_words = [w.strip(".,!?()[]\"'") for w in gold_lower.split() if len(w.strip(".,!?()[]\"'")) > 3]
    answer_eids = set()
    for eid, text in inst_text.items():
        found = [w for w in gold_words if w in text]
        if len(found) >= max(1, len(gold_words) * 0.3):
            answer_eids.add(eid)
    
    retrieved = set(trace.get("retrieved_entity_ids", []))
    neigh = set(trace.get("neighbor_expansion", {}).get("entity_ids", []))
    return len(answer_eids & (retrieved | neigh)), len(answer_eids)

# Run with ORIGINAL config (1-hop, which is the committed default)
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "1"
os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = "16"
# Clear adjacency cache
cg._adj_cache = None

print("=== Config A: Original 1-hop MMR ===")
results_a = {}
for r in cat1_wrong:
    q = r["question"]
    cg._adj_cache = None
    ctx, trace = cg._search_by_sub_hash(q)
    found, total = count_answer_entities(q, r["answer"], trace)
    results_a[r["qa_source_index"]] = found
    print(f"  Q{r['qa_source_index']+1}: {found}/{total} answer entities")

# Run with NEW config (2-hop, round-robin)
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "2"
cg._adj_cache = None

print("\n=== Config B: Our 2-hop round-robin ===")
results_b = {}
for r in cat1_wrong:
    q = r["question"]
    cg._adj_cache = None
    ctx, trace = cg._search_by_sub_hash(q)
    found, total = count_answer_entities(q, r["answer"], trace)
    results_b[r["qa_source_index"]] = found
    print(f"  Q{r['qa_source_index']+1}: {found}/{total} answer entities")

print("\n=== Comparison ===")
better = worse = same = 0
for idx in results_a:
    a, b = results_a[idx], results_b[idx]
    delta = b - a
    mark = "+" if delta > 0 else ("-" if delta < 0 else "=")
    if delta > 0: better += 1
    elif delta < 0: worse += 1
    else: same += 1
    print(f"  Q{idx+1}: A={a}, B={b} [{mark}]")
print(f"\nBetter: {better}, Worse: {worse}, Same: {same}")
