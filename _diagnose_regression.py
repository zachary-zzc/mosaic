#!/usr/bin/env python3
"""
Diagnose why the patched graph regressed on Q1 and Q6.
Show which neighbors (instance keys) each graph retrieves, and which are different.
"""
import os, sys, json
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "mosaic", "src"))

from src.data.graph import ClassGraph
from src.assist import load_mosaic_memory_pickle, read_to_file_json

QA_PATH = os.path.join(PROJECT_ROOT, "dataset/locomo/qa_0.json")
TAGS_PATH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/conv0_tags.json")
ORIGINAL_GRAPH = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0.pkl")
PATCHED_GRAPH  = os.path.join(PROJECT_ROOT, "experiments/locomo/benchmark/runs/conv0/artifacts/hybrid/graph_network_conv0_patched.pkl")

questions = read_to_file_json(QA_PATH)
mh = [q for q in questions if q.get("category") == 3]

# Q1 and Q6 are the ones that regressed
DIAG_INDICES = [0, 5]  # 0-based

for qi in DIAG_INDICES:
    question = mh[qi]["question"]
    gold = mh[qi]["answer"]
    print(f"\n{'='*80}")
    print(f"Q{qi+1}: {question}")
    print(f"Gold: {gold}")
    print(f"{'='*80}")

    for label, gpath in [("ORIGINAL", ORIGINAL_GRAPH), ("PATCHED", PATCHED_GRAPH)]:
        memory = load_mosaic_memory_pickle(gpath)
        memory.process_kw(TAGS_PATH)

        ctx, trace = memory._search_by_sub_hash(question)

        seeds_tfidf = set(trace.get("tfidf_hits", {}).get("entity_ids", []))
        neighbor_eids = trace.get("neighbor_expansion", {}).get("entity_ids", [])

        print(f"\n  --- {label} ---")
        print(f"  TF-IDF seeds ({len(seeds_tfidf)}): {sorted(seeds_tfidf)[:10]}...")
        print(f"  Neighbor expansion ({len(neighbor_eids)}):")
        for eid in neighbor_eids:
            print(f"    {eid}")

        # Show adjacency count for seeds
        from src.config_loader import get_query_neighbor_traversal_config
        _, _, legs = get_query_neighbor_traversal_config()
        adj = memory._build_instance_adjacency(legs)
        seed_keys = set()
        for eid in seeds_tfidf:
            # Convert entity_id (class_X:instance_Y) to instance_key (class_X_instance_Y)
            parts = eid.split(":")
            if len(parts) == 2:
                seed_keys.add(f"{parts[0]}_{parts[1]}")

        total_neighbors = 0
        for sk in seed_keys:
            n = len(adj.get(sk, set()))
            total_neighbors += n
            if n > 5:
                print(f"  Seed {sk} has {n} neighbors in adjacency")
        print(f"  Total seed neighbor count: {total_neighbors}")
        
        # Context size
        print(f"  Context length: {len(ctx)} chars")
        
        # Show first 500 chars of context
        print(f"  Context preview:")
        print(f"  {ctx[:600]}")
        print()
