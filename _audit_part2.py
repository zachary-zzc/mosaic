"""Part 2: Check WHY classes have no theme/tags and why all answer entities have 0 edges."""
import os, sys, json, pickle
from collections import Counter
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

from src.data.graph import ClassGraph

BASE = "/Users/zachary/Workspace/LongtermMemory"
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg = pickle.load(f)
cg.process_kw(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json")

# 1. Class metadata status
has_theme = has_tags = has_neither = 0
empty_theme_examples = []
for node in cg.graph.nodes:
    cid = getattr(node, "class_id", "")
    theme = (getattr(node, "theme", "") or "").strip()
    tags = getattr(node, "tags", []) or []
    if theme:
        has_theme += 1
    if tags:
        has_tags += 1
    if not theme and not tags:
        has_neither += 1
        if len(empty_theme_examples) < 5:
            empty_theme_examples.append(cid)

total = sum(1 for _ in cg.graph.nodes)
print(f"=== CLASS METADATA ===")
print(f"Total classes: {total}")
print(f"Has theme: {has_theme} ({100*has_theme/total:.0f}%)")
print(f"Has tags: {has_tags} ({100*has_tags/total:.0f}%)")
print(f"Has neither: {has_neither}")
print(f"Examples with no theme/tags: {empty_theme_examples}")

# 2. Edge connectivity
# How many instances have 0 edges?
inst_edge_count = Counter()
for rec in cg.edges or []:
    conns = rec.get("connections") or []
    conn_keys = set()
    for c in conns:
        ocid = c.get("class_id")
        oid = c.get("instance_id")
        if ocid and oid is not None:
            conn_keys.add(f"class_{ocid}:instance_{oid}")
    for k in conn_keys:
        inst_edge_count[k] += 1

total_instances = 0
isolated = 0
for node in cg.graph.nodes:
    cid = getattr(node, "class_id", "")
    for inst in getattr(node, "_instances", []) or []:
        iid = inst.get("instance_id", "")
        eid = f"{cid}:{iid}"
        total_instances += 1
        if inst_edge_count[eid] == 0:
            isolated += 1

print(f"\n=== EDGE CONNECTIVITY ===")
print(f"Total instances: {total_instances}")
print(f"Isolated (0 edges): {isolated} ({100*isolated/total_instances:.0f}%)")
print(f"Connected: {total_instances - isolated}")
print(f"Total edge records: {len(cg.edges or [])}")

# 3. Check edge connection format
# Look at a few edges to understand the connections format
print(f"\n=== SAMPLE EDGES ===")
for rec in (cg.edges or [])[:3]:
    print(f"  leg={rec.get('edge_leg')}")
    print(f"  content={str(rec.get('content',''))[:80]}")
    conns = rec.get("connections", [])
    print(f"  connections ({len(conns)}):")
    for c in conns[:4]:
        print(f"    class_id={c.get('class_id')} instance_id={c.get('instance_id')}")
    print()

# 4. Check how _build_instance_adjacency works vs edge's connections
from src.data.graph import normalize_edge_leg
from src.config_loader import get_query_neighbor_traversal_config

_, _, legs = get_query_neighbor_traversal_config()
adj = cg._build_instance_adjacency(legs)
adj_sizes = [len(v) for v in adj.values()]
print(f"\n=== ADJACENCY ===")
print(f"Nodes in adj: {len(adj)}")
print(f"Avg degree: {sum(adj_sizes)/len(adj_sizes):.1f}" if adj_sizes else "Empty")
print(f"Max degree: {max(adj_sizes)}" if adj_sizes else "")

# Check if the adjacency keys use different format
sample_adj_keys = list(adj.keys())[:5]
print(f"Sample adj keys: {sample_adj_keys}")

# 5. Check if answer entities are in adjacency at all
answer_eid_samples = ["class_3:instance_1", "class_144:instance_1", "class_81:instance_1", "class_7:instance_1"]
print(f"\n=== ANSWER ENTITY ADJACENCY ===")
for eid in answer_eid_samples:
    # Convert to key format
    parts = eid.split(":")
    key = f"{parts[0]}_{parts[1]}"
    in_adj = key in adj
    deg = len(adj.get(key, set()))
    in_edge_count = inst_edge_count.get(eid, 0)
    print(f"  {eid} (key={key}): in_adj={in_adj}, deg={deg}, edge_records={in_edge_count}")
    
    # Also check if the class_id format in edges matches
    # The edge connections have class_id as just a number (e.g., "3") not "class_3"
    cid_num = parts[0].replace("class_", "")
    iid_num = parts[1].replace("instance_", "")
    edge_count2 = 0
    for rec in cg.edges or []:
        for c in rec.get("connections") or []:
            if str(c.get("class_id")) == cid_num and str(c.get("instance_id")) == iid_num:
                edge_count2 += 1
                break
    print(f"    Matching with numeric class_id={cid_num}, iid={iid_num}: {edge_count2} edge records")
