#!/usr/bin/env python3
"""Quick comparison of original vs patched graph: adjacency + retrieval for multi-hop questions."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mosaic"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mosaic", "src"))

from src.data.graph import ClassGraph
from src.data.dual_graph import ALL_EDGE_LEGS
from src.assist import load_mosaic_memory_pickle, read_to_file_json

ORIGINAL = "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0.pkl"
PATCHED  = "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0_patched.pkl"
TAGS     = "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/conv0_tags.json"

print("Loading original graph...")
orig = load_mosaic_memory_pickle(ORIGINAL)
orig.process_kw(TAGS)

print("Loading patched graph...")
patched = load_mosaic_memory_pickle(PATCHED)
patched.process_kw(TAGS)

# --- Adjacency comparison ---
print("\n=== ADJACENCY COMPARISON ===")
orig_adj = orig._build_instance_adjacency(frozenset(ALL_EDGE_LEGS))
patch_adj = patched._build_instance_adjacency(frozenset(ALL_EDGE_LEGS))

orig_total_edges = sum(len(v) for v in orig_adj.values()) // 2
patch_total_edges = sum(len(v) for v in patch_adj.values()) // 2
orig_isolated = sum(1 for k, v in orig_adj.items() if not v)
patch_isolated = sum(1 for k, v in patch_adj.items() if not v)

print(f"Original:  {len(orig_adj)} nodes, {orig_total_edges} edges, {orig_isolated} isolated")
print(f"Patched:   {len(patch_adj)} nodes, {patch_total_edges} edges, {patch_isolated} isolated")
print(f"New edges: +{patch_total_edges - orig_total_edges}")
print(f"Isolated fixed: {orig_isolated} -> {patch_isolated}")

# Find nodes with new neighbors
new_connections = 0
for k in patch_adj:
    orig_neighbors = orig_adj.get(k, set())
    patch_neighbors = patch_adj.get(k, set())
    if len(patch_neighbors) > len(orig_neighbors):
        new_connections += 1
print(f"Nodes with new neighbors: {new_connections}")

# --- Retrieval comparison on multi-hop + open-domain ---
print("\n=== RETRIEVAL COMPARISON ===")
qa = read_to_file_json("dataset/locomo/qa_0.json")
target_qs = [q for q in qa if q.get("category") in (3, 4)]
print(f"Testing {len(target_qs)} questions (multi-hop + open-domain)\n")

cat_names = {3: "Multi-hop", 4: "Open-domain"}
results = {3: {"orig_ctx": [], "patch_ctx": [], "new_entities": [], "questions": []},
           4: {"orig_ctx": [], "patch_ctx": [], "new_entities": [], "questions": []}}

for i, q in enumerate(target_qs):
    cat = q["category"]
    question = q["question"]
    gold = q["answer"]
    
    ctx_o, trace_o = orig._search_by_sub_hash(question)
    ctx_p, trace_p = patched._search_by_sub_hash(question)
    
    o_eids = set(trace_o["retrieved_entity_ids"]) | set(trace_o["neighbor_expansion"]["entity_ids"])
    p_eids = set(trace_p["retrieved_entity_ids"]) | set(trace_p["neighbor_expansion"]["entity_ids"])
    new = p_eids - o_eids
    lost = o_eids - p_eids
    
    results[cat]["orig_ctx"].append(len(ctx_o))
    results[cat]["patch_ctx"].append(len(ctx_p))
    results[cat]["new_entities"].append(len(new))
    
    if new:
        results[cat]["questions"].append({
            "q": question[:80], "gold": gold[:60],
            "new": len(new), "lost": len(lost),
            "orig_n": len(o_eids), "patch_n": len(p_eids)
        })
    
    print(f"[{i+1}/{len(target_qs)}] {cat_names[cat]} | orig={len(o_eids)} patch={len(p_eids)} new={len(new)} | {question[:60]}")

# --- Summary ---
print("\n=== SUMMARY ===")
for cat in [3, 4]:
    r = results[cat]
    n = len(r["orig_ctx"])
    if n == 0:
        continue
    avg_orig = sum(r["orig_ctx"]) / n
    avg_patch = sum(r["patch_ctx"]) / n
    total_new = sum(r["new_entities"])
    improved = sum(1 for x in r["new_entities"] if x > 0)
    
    print(f"\n{cat_names[cat]} ({n} questions):")
    print(f"  Avg context chars: {avg_orig:.0f} -> {avg_patch:.0f} ({'+' if avg_patch >= avg_orig else ''}{avg_patch - avg_orig:.0f})")
    print(f"  Questions with new entities found: {improved}/{n}")
    print(f"  Total new entities across all questions: {total_new}")
    
    if r["questions"]:
        print(f"  Improved questions:")
        for qr in r["questions"]:
            print(f"    [{qr['orig_n']}->{qr['patch_n']}] +{qr['new']} new | {qr['q']}")
            print(f"      Gold: {qr['gold']}")
