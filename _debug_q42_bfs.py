"""Debug Q30/Q42: compare candidate pools and scores."""
import os, sys, json, pickle
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "2"
os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = "16"

from src.assist import build_instance_fragments, calculate_tfidf_similarity
from src.data.graph import ClassGraph
from src.config_loader import get_query_neighbor_traversal_config
from src.data.dual_graph import normalize_edge_leg
from collections import defaultdict, deque

BASE = "/Users/zachary/Workspace/LongtermMemory"
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg = pickle.load(f)
tag_path = f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json"
cg.process_kw(tag_path)

# Run retrieval for Q42 and dump internal state
q = "What are the breeds of Audrey's dogs?"
ctx, trace = cg._search_by_sub_hash(q)

# Get the TF-IDF seeds
retrieved_eids = trace.get("retrieved_entity_ids", [])
neigh_eids = trace.get("neighbor_expansion", {}).get("entity_ids", [])
print(f"Q42: {q}")
print(f"  TF-IDF retrieved: {len(retrieved_eids)}")
print(f"  Neighbor expansion: {len(neigh_eids)}")
print(f"  Neighbor IDs: {sorted(neigh_eids)}")

# Target: class_144:instance_1 (breed info), class_3:instance_1 (dog names)
targets = ["class_144:instance_1", "class_3:instance_1", "class_69:instance_1", "class_81:instance_1"]
for t in targets:
    if t in retrieved_eids:
        print(f"  {t}: in TF-IDF retrieved")
    elif t in neigh_eids:
        print(f"  {t}: in neighbor expansion")
    else:
        print(f"  {t}: MISSED")

# Now manually trace the BFS to check if targets are reachable
max_hops, max_extra, legs = get_query_neighbor_traversal_config()
adj = cg._build_instance_adjacency(legs)

# Get seeds from TF-IDF
seed_keys = set(cg.selected_instance_keys) if cg.selected_instance_keys else set()

# Actually, by the time we check, selected_instance_keys is already cleared.
# Let's re-derive: the trace has retrieved_entity_ids which we can map back.
# entity_id format is "class_X:instance_Y", instance_key format is "class_X_instance_Y"
def eid_to_key(eid):
    parts = eid.split(":")
    if len(parts) == 2:
        return f"{parts[0]}_{parts[1]}"
    return eid

seed_keys = set(eid_to_key(e) for e in retrieved_eids)
target_keys = [eid_to_key(t) for t in targets]

print(f"\n  Seeds ({len(seed_keys)}): {sorted(list(seed_keys))[:5]}...")
for tk, te in zip(target_keys, targets):
    # BFS from seeds to find distance
    visited = {s: 0 for s in seed_keys}
    frontier = deque((s, 0) for s in seed_keys)
    found_dist = -1
    while frontier:
        u, d = frontier.popleft()
        if d >= 3:
            continue
        for v in adj.get(u, ()):
            if v in visited:
                continue
            visited[v] = d + 1
            if v == tk:
                found_dist = d + 1
                break
            frontier.append((v, d + 1))
        if found_dist > 0:
            break
    # Check if it's a bridge node
    inst = None
    for node in cg.graph.nodes:
        cid = getattr(node, "class_id", "")
        for i in getattr(node, "_instances", []) or []:
            key = f"{cid}_{i.get('instance_id', '')}"
            if key == tk:
                inst = i
                break

    print(f"  {te}: dist={found_dist}, in_adj={'yes' if tk in adj else 'no'}, neighbors={len(adj.get(tk, set()))}")
