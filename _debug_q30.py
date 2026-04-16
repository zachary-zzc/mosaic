"""Debug Q30 and Q42: what neighbors did old vs new select?"""
import os, sys, json, pickle
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "2" 
os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = "16"

from src.assist import build_instance_fragments
from src.data.graph import ClassGraph

BASE = "/Users/zachary/Workspace/LongtermMemory"
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg = pickle.load(f)
tag_path = f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json"
cg.process_kw(tag_path)

with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/results/qa_5_eval_full_hybrid.json") as f:
    eval_data = json.load(f)

# Q30: "What are the names of Audrey's dogs?"
r30 = [r for r in eval_data["results"] if r["qa_source_index"] == 29][0]
old_neigh = set(r30.get("retrieved_context", {}).get("neighbor_expansion", {}).get("entity_ids", []))

# Run new retrieval
q = r30["question"]
ctx, trace = cg._search_by_sub_hash(q)
new_neigh = set(trace.get("neighbor_expansion", {}).get("entity_ids", []))

print("Q30: What are the names of Audrey's dogs?")
print(f"  Old neighbor IDs: {sorted(old_neigh)}")
print(f"  New neighbor IDs: {sorted(new_neigh)}")
print(f"  Lost: {sorted(old_neigh - new_neigh)}")
print(f"  Gained: {sorted(new_neigh - old_neigh)}")

# Check lost entities
inst_text = {}
for node in cg.graph.nodes:
    cid = getattr(node, "class_id", "")
    for inst in getattr(node, "_instances", []) or []:
        iid = inst.get("instance_id", "")
        eid = f"{cid}:{iid}"
        frags = build_instance_fragments(inst)
        text = "\n".join(t for _ty, t in frags if t and t.strip())[:200]
        inst_text[eid] = text

print("\nLost entities content:")
for eid in sorted(old_neigh - new_neigh):
    print(f"  {eid}: {inst_text.get(eid, '?')[:150]}")

print("\nGained entities content:")
for eid in sorted(new_neigh - old_neigh):
    print(f"  {eid}: {inst_text.get(eid, '?')[:150]}")

# Also check Q42
print("\n" + "="*60)
r42 = [r for r in eval_data["results"] if r["qa_source_index"] == 41][0]
old_neigh42 = set(r42.get("retrieved_context", {}).get("neighbor_expansion", {}).get("entity_ids", []))
q42 = r42["question"]
ctx42, trace42 = cg._search_by_sub_hash(q42)
new_neigh42 = set(trace42.get("neighbor_expansion", {}).get("entity_ids", []))
print(f"Q42: {q42}")
print(f"  Old neighbor IDs: {sorted(old_neigh42)}")
print(f"  New neighbor IDs: {sorted(new_neigh42)}")
print(f"  Lost: {sorted(old_neigh42 - new_neigh42)}")
