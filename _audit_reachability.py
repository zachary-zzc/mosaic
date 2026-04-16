"""Part 3: Verify adjacency reachability with correct key format."""
import os, sys, json, pickle
from collections import defaultdict
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "2"
os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = "16"

from src.assist import build_instance_fragments
from src.data.graph import ClassGraph
from src.config_loader import get_query_neighbor_traversal_config

BASE = "/Users/zachary/Workspace/LongtermMemory"
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg = pickle.load(f)
cg.process_kw(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json")

with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/results/qa_5_eval_full_hybrid.json") as f:
    eval_data = json.load(f)

_, _, legs = get_query_neighbor_traversal_config()
adj = cg._build_instance_adjacency(legs)

# Get seeds for key queries
queries = {
    "Q30": "What are the names of Audrey's dogs?",
    "Q42": "What are the breeds of Audrey's dogs?",
    "Q13": "What outdoor activities has Andrew done other than hiking in nature?",
    "Q19": "Where did Audrey get Pixie from?",
}

# Answer entity keys (underscore format matching adj keys)
targets = {
    "Q30": ["class_3_instance_1", "class_144_instance_1", "class_81_instance_1", "class_7_instance_1", "class_14_instance_1"],
    "Q42": ["class_144_instance_1", "class_3_instance_1", "class_81_instance_1", "class_7_instance_1"],
    "Q13": ["class_36_instance_1", "class_89_instance_2"],
    "Q19": ["class_54_instance_2"],
}

for qid, query in queries.items():
    print(f"\n{'='*70}")
    print(f"{qid}: {query}")
    
    # Get seeds
    cg._adj_cache = None
    cg.selected_instance_keys.clear()
    ctx, trace = cg._search_by_sub_hash(query)
    seeds = set(cg.selected_instance_keys)  # This would be cleared already...
    # Use trace to reconstruct
    seed_eids = trace.get("retrieved_entity_ids", [])
    # Convert eid to key format
    seed_keys = set()
    for eid in seed_eids:
        # class_X:instance_Y -> class_X_instance_Y
        key = eid.replace(":", "_")
        seed_keys.add(key)
    
    print(f"  Seeds ({len(seed_keys)}): {sorted(seed_keys)[:5]}...")
    
    # Check each target
    for tgt in targets[qid]:
        in_adj = tgt in adj
        deg = len(adj.get(tgt, set()))
        
        # Is it reachable from any seed at hop 1 or 2?
        hop1_seeds = [s for s in seed_keys if tgt in adj.get(s, set())]
        hop2_seeds = []
        if not hop1_seeds:
            # Check hop 2: any seed -> intermediate -> target
            for s in seed_keys:
                for mid in adj.get(s, set()):
                    if tgt in adj.get(mid, set()):
                        hop2_seeds.append((s, mid))
                        break
        
        print(f"\n  Target {tgt}: in_adj={in_adj}, degree={deg}")
        if hop1_seeds:
            print(f"    REACHABLE at hop 1 from {len(hop1_seeds)} seeds: {hop1_seeds[:3]}")
        elif hop2_seeds:
            print(f"    REACHABLE at hop 2 via: {hop2_seeds[:3]}")
        else:
            print(f"    NOT REACHABLE from any seed within 2 hops!")
            # Show what this target IS connected to
            neighbors = adj.get(tgt, set())
            print(f"    Has {len(neighbors)} neighbors: {sorted(neighbors)[:5]}...")
            # How many of those neighbors are seeds?
            overlap = neighbors & seed_keys
            print(f"    Neighbors that are seeds: {len(overlap)}")
