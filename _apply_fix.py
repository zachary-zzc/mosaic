#!/usr/bin/env python3
"""
Apply the cross-class edge sweep fix to the existing conv0 hybrid graph,
save the patched graph, and run QA eval to test improvement.
"""
import json, os, sys, pickle, time, shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic", "src"))

GRAPH_PATH = "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0.pkl"
TAGS_PATH = "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/conv0_tags.json"
PATCHED_GRAPH = "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0_patched.pkl"
QA_PATH = "dataset/locomo/qa_0.json"
OUT_PATH = "experiments/locomo/benchmark/runs/conv0/results/qa_0_eval_full_hybrid_patched.json"
SUM_PATH = "experiments/locomo/benchmark/runs/conv0/results/qa_0_eval_summary_hybrid_patched.json"

# 1. Load the existing graph
print("Loading existing graph...")
with open(GRAPH_PATH, "rb") as f:
    memory = pickle.load(f)

# Stats before
edges_before = len(getattr(memory, "edges", []) or [])
from src.data.dual_graph import ALL_EDGE_LEGS
adj_before = memory._build_instance_adjacency(frozenset(ALL_EDGE_LEGS))
connected_before = sum(1 for v in adj_before.values() if len(v) > 0)
isolated_before = sum(
    len(getattr(n, "_instances", []) or [])
    for n in memory.graph.nodes
) - len(adj_before)

print(f"Before sweep: {edges_before} edge records, {connected_before} connected, {isolated_before} isolated")

# 2. Apply the sweep
print("\nApplying cross-class edge sweep...")
stats = memory.sweep_cross_class_cooccurrence_edges()
print(f"Sweep stats: {stats}")

# Stats after
memory._adj_cache = None  # clear cache
adj_after = memory._build_instance_adjacency(frozenset(ALL_EDGE_LEGS))
connected_after = sum(1 for v in adj_after.values() if len(v) > 0)
edges_after = len(getattr(memory, "edges", []) or [])
total_instances = sum(
    len(getattr(n, "_instances", []) or [])
    for n in memory.graph.nodes
)
isolated_after = total_instances - len(adj_after)

print(f"After sweep: {edges_after} edge records (+{edges_after - edges_before})")
print(f"Connected: {connected_after} (was {connected_before}), Isolated: {isolated_after} (was {isolated_before})")
print(f"G_p: {memory.G_p.number_of_nodes()} nodes, {memory.G_p.number_of_edges()} edges")
print(f"G_a: {memory.G_a.number_of_nodes()} nodes, {memory.G_a.number_of_edges()} edges")

# 3. Save patched graph
print(f"\nSaving patched graph to {PATCHED_GRAPH}...")
with open(PATCHED_GRAPH, "wb") as f:
    pickle.dump(memory, f, protocol=pickle.HIGHEST_PROTOCOL)

# 4. Verify retrieval improvement for previously failing questions
print("\n" + "="*60)
print("Verifying retrieval improvement")
print("="*60)

memory.process_kw(TAGS_PATH)

import re
all_instances_text = {}
for node in memory.graph.nodes:
    cid = getattr(node, "class_id", None) or ""
    for inst in getattr(node, "_instances", []) or []:
        iid = inst.get("instance_id")
        ikey = memory._instance_key(cid, iid)
        eid = memory._instance_key_to_entity_id(ikey)
        text = json.dumps(inst, ensure_ascii=False, default=str)
        all_instances_text[eid] = text

eid_to_ikey = {}
for node in memory.graph.nodes:
    cid = getattr(node, "class_id", None) or ""
    for inst in getattr(node, "_instances", []) or []:
        iid = inst.get("instance_id")
        ikey = memory._instance_key(cid, iid)
        eid = memory._instance_key_to_entity_id(ikey)
        eid_to_ikey[eid] = ikey

test_cases = [
    ("Would Caroline be considered religious?", ["religious", "church", "faith"], 3),
    ("What did Melanie and her family see during their camping trip last year?", ["Perseid", "meteor"], 4),
    ("Did Melanie make the black and white bowl in the photo?", ["black and white", "bowl"], 4),
]

for question, terms, cat in test_cases:
    print(f"\n--- Cat={cat}: {question}")
    gold_entities = set()
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for eid, text in all_instances_text.items():
            if pattern.search(text):
                gold_entities.add(eid)

    ctx, trace = memory._search_by_sub_hash(question)
    retrieved_eids = set(trace.get("retrieved_entity_ids", []))
    neighbor_eids = set(trace.get("neighbor_expansion", {}).get("entity_ids", []))
    all_retrieved = retrieved_eids | neighbor_eids
    found = gold_entities & all_retrieved
    missed = gold_entities - all_retrieved
    print(f"  Gold: {len(gold_entities)}, Retrieved+Neighbors: {len(all_retrieved)}, Found: {len(found)}, Missed: {len(missed)}")
    
    # Check if missed are still disconnected
    disconnected = 0
    for m in missed:
        m_ikey = eid_to_ikey.get(m)
        if not m_ikey:
            disconnected += 1
            continue
        neighbors_of_m = adj_after.get(m_ikey, set())
        retrieved_ikeys = {eid_to_ikey.get(e) for e in all_retrieved if e in eid_to_ikey}
        if not (neighbors_of_m & retrieved_ikeys):
            disconnected += 1
    print(f"  Still disconnected: {disconnected}")

print("\nDone. Patched graph saved.")
print(f"Graph: {PATCHED_GRAPH}")
print(f"Tags: {TAGS_PATH}")
