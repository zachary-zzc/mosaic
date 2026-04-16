"""Check that the 17 previously-correct multi-hop queries still get their answer entities."""
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

print(f"Testing {len(cat1_correct)} previously-correct multi-hop queries\n")
lost = 0
for r in cat1_correct:
    q = r["question"]
    old_eids = set(r.get("retrieved_context", {}).get("retrieved_entity_ids", []))
    old_neigh = set(r.get("retrieved_context", {}).get("neighbor_expansion", {}).get("entity_ids", []))
    old_all = old_eids | old_neigh

    cg._adj_cache = None
    ctx, trace = cg._search_by_sub_hash(q)
    new_eids = set(trace.get("retrieved_entity_ids", []))
    new_neigh = set(trace.get("neighbor_expansion", {}).get("entity_ids", []))
    new_all = new_eids | new_neigh

    gold_lower = r["answer"].lower()
    gold_words = [w.strip(".,!?()[]\"'") for w in gold_lower.split() if len(w.strip(".,!?()[]\"'")) > 3]
    answer_eids = set()
    for eid, text in inst_text.items():
        found = [w for w in gold_words if w in text]
        if len(found) >= max(1, len(gold_words) * 0.3):
            answer_eids.add(eid)

    old_found = len(answer_eids & old_all)
    new_found = len(answer_eids & new_all)
    
    if new_found < old_found:
        lost += 1
        print(f"[LOST] Q{r['qa_source_index']+1}: old={old_found}, new={new_found}  {q[:60]}")

if lost == 0:
    print("All previously-correct queries maintain or improve answer entity coverage!")
else:
    print(f"\n{lost} queries lost answer entities")
