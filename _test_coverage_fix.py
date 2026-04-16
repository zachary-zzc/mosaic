"""Quick test: run the new coverage-aware neighbor expansion on conv5 multi-hop failures.
Checks whether answer entities now appear in the expanded neighbor set."""
import os, sys, json, pickle
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

# Force 2-hop, 16 extra
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "2"
os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = "16"

from src.assist import build_instance_fragments, calculate_tfidf_similarity
from src.data.graph import ClassGraph

BASE = "/Users/zachary/Workspace/LongtermMemory"

# Load graph
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg: ClassGraph = pickle.load(f)

# Load tags
tag_path = f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json"
cg.process_kw(tag_path)

# Load old eval results to identify failing queries
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

cat1_wrong = [r for r in eval_data["results"] if r["category"] == 1 and r["judgment"] == "WRONG"]
cat1_correct = [r for r in eval_data["results"] if r["category"] == 1 and r["judgment"] == "CORRECT"]

print(f"Conv5 multi-hop: {len(cat1_correct)} correct, {len(cat1_wrong)} wrong (old)")
print(f"Testing new coverage-aware neighbor expansion on {len(cat1_wrong)} failing queries\n")

improved = 0
for i, r in enumerate(cat1_wrong):
    q = r["question"]
    gold = r["answer"]
    old_retrieved_eids = set(r.get("retrieved_context", {}).get("retrieved_entity_ids", []))
    old_neigh_eids = set(r.get("retrieved_context", {}).get("neighbor_expansion", {}).get("entity_ids", []))

    # Run the NEW retrieval
    ctx, trace = cg._search_by_sub_hash(q)

    new_retrieved_eids = set(trace.get("retrieved_entity_ids", []))
    new_neigh_eids = set(trace.get("neighbor_expansion", {}).get("entity_ids", []))
    all_new = new_retrieved_eids | new_neigh_eids

    # Find answer entities
    gold_lower = gold.lower()
    gold_words = [w.strip(".,!?()[]\"'") for w in gold_lower.split() if len(w.strip(".,!?()[]\"'")) > 3]
    answer_eids = set()
    for eid, text in inst_text.items():
        found = [w for w in gold_words if w in text]
        if len(found) >= max(1, len(gold_words) * 0.3):
            answer_eids.add(eid)

    old_answer_in_ctx = answer_eids & (old_retrieved_eids | old_neigh_eids)
    new_answer_in_ctx = answer_eids & all_new
    new_answer_in_neigh = answer_eids & new_neigh_eids

    delta = len(new_answer_in_ctx) - len(old_answer_in_ctx)
    marker = "IMPROVED" if delta > 0 else ("SAME" if delta == 0 else "WORSE")
    if delta > 0:
        improved += 1

    print(f"[{marker}] Q{r['qa_source_index']+1}: {q[:70]}")
    print(f"  Gold: {gold[:70]}")
    print(f"  Answer entities: {len(answer_eids)}")
    print(f"  OLD: {len(old_answer_in_ctx)} answer entities in context (tfidf+neigh)")
    print(f"  NEW: {len(new_answer_in_ctx)} answer entities in context (tfidf={len(answer_eids & new_retrieved_eids)}, neigh={len(new_answer_in_neigh)})")
    print(f"  New neighbor count: {trace.get('neighbor_expansion', {}).get('count', 0)}")
    if new_answer_in_neigh - old_neigh_eids:
        print(f"  Newly found via neighbors: {sorted(new_answer_in_neigh - old_neigh_eids)}")
    print()

print(f"{'='*60}")
print(f"Improved: {improved}/{len(cat1_wrong)} failing queries now have more answer entities in context")
