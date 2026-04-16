"""Trace round-robin selection for Q42 to see why targets aren't picked."""
import os, sys, json, pickle
from collections import defaultdict
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "2"
os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = "16"

from src.assist import build_instance_fragments
from src.data.graph import ClassGraph, normalize_edge_leg
from src.config_loader import get_query_neighbor_traversal_config

BASE = "/Users/zachary/Workspace/LongtermMemory"
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg = pickle.load(f)
cg.process_kw(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json")

query = "What are the breeds of Audrey's dogs?"
targets = {"class_144_instance_1", "class_3_instance_1", "class_81_instance_1", "class_7_instance_1"}

max_hops, max_extra, legs = get_query_neighbor_traversal_config()
adj = cg._build_instance_adjacency(legs)

# Get seeds
cg._adj_cache = None
cg.selected_instance_keys.clear()
ctx, trace = cg._search_by_sub_hash(query)
seed_eids = trace.get("retrieved_entity_ids", [])
seeds = set(eid.replace(":", "_") for eid in seed_eids)

print(f"Query: {query}")
print(f"Seeds: {len(seeds)}")
print(f"Targets: {targets}")

# Reproduce the BFS
key_to_inst = {}
for class_node in cg.graph.nodes:
    cid = getattr(class_node, "class_id", None) or ""
    for inst in getattr(class_node, "_instances", []) or []:
        iid = inst.get("instance_id")
        if iid is not None:
            key_to_inst[cg._instance_key(cid, iid)] = inst

def _inst_text(key):
    inst = key_to_inst.get(key)
    if inst is None: return ""
    parts = [tx for _ty, tx in build_instance_fragments(inst) if (tx or "").strip()]
    return "\n".join(parts) if parts else ""

q_lower = query.lower()
q_words = set()
for w in q_lower.split():
    w = w.strip(".,!?()[]\"':;")
    if len(w) > 2: q_words.add(w)
_stop = {"the", "and", "for", "are", "was", "were", "has", "have", "had",
         "that", "this", "with", "from", "what", "which", "who", "how",
         "does", "did", "not", "been", "but", "they", "them", "than",
         "can", "her", "his", "she", "you", "all", "any", "some", "will"}
q_words -= _stop

edge_texts_by_key = defaultdict(list)
for rec in cg.edges or []:
    leg = normalize_edge_leg(rec.get("edge_leg"))
    if leg not in legs: continue
    content = (rec.get("content") or "").lower()
    if not content: continue
    for c in rec.get("connections") or []:
        ocid = c.get("class_id")
        oid = c.get("instance_id")
        if ocid and oid is not None:
            edge_texts_by_key[cg._instance_key(str(ocid), oid)].append(content)

def _kw_hits(key):
    txt = _inst_text(key).lower()
    hits = sum(1 for kw in q_words if kw in txt)
    for etxt in edge_texts_by_key.get(key, []):
        hits += sum(1 for kw in q_words if kw in etxt)
    return hits

# BFS
from collections import deque
visited = {s: 0 for s in seeds}
frontier = deque((s, 0) for s in seeds)
candidates = []
candidate_hop = {}

while frontier:
    u, d = frontier.popleft()
    if d >= max_hops: continue
    nd = d + 1
    for v in adj.get(u, ()):
        if v in visited: continue
        visited[v] = nd
        if v not in seeds:
            candidates.append(v)
            candidate_hop[v] = nd
        if nd < max_hops:
            if nd == 1 or _kw_hits(v) >= 1:
                frontier.append((v, nd))

print(f"\nBFS candidates: {len(candidates)}")
for tgt in targets:
    if tgt in candidate_hop:
        print(f"  {tgt}: hop={candidate_hop[tgt]}")
    else:
        print(f"  {tgt}: NOT IN CANDIDATES")

# Now simulate round-robin
def _score(ck):
    edge_hits = 0
    for etxt in edge_texts_by_key.get(ck, []):
        edge_hits += sum(1 for kw in q_words if kw in etxt)
    edge_sc = min(edge_hits / max(len(q_words), 1), 1.0)
    inst_txt = _inst_text(ck).lower()
    inst_hits = sum(1 for kw in q_words if kw in inst_txt)
    inst_sc = min(inst_hits / max(len(q_words), 1), 1.0)
    return 0.5 * inst_sc + 0.5 * edge_sc

hop1_cands = [ck for ck in candidates if candidate_hop.get(ck, 1) == 1]

# Map: seed → sorted neighbors
seed_nbrs = defaultdict(list)
for ck in hop1_cands:
    sc = _score(ck)
    for sk in seeds:
        if ck in adj.get(sk, set()):
            seed_nbrs[sk].append((ck, sc))
for sk in seed_nbrs:
    seed_nbrs[sk].sort(key=lambda x: -x[1])

# Show which seeds have targets as their top neighbor
print(f"\nSeeds with target entities as neighbors:")
for sk in sorted(seed_nbrs):
    nbrs = seed_nbrs[sk]
    target_nbrs = [(ck, sc) for ck, sc in nbrs if ck in targets]
    if target_nbrs:
        # What's the rank of the target?
        for tgt_ck, tgt_sc in target_nbrs:
            rank = next(i for i, (ck, sc) in enumerate(nbrs) if ck == tgt_ck)
            print(f"  Seed {sk}: target {tgt_ck} rank={rank+1}/{len(nbrs)}, score={tgt_sc:.3f}, top_score={nbrs[0][1]:.3f}")

# Simulate round-robin
print(f"\nRound-robin selection:")
result = []
selected_set = set()
cursor = {sk: 0 for sk in seed_nbrs}
active = [sk for sk in seeds if sk in seed_nbrs]

rr_round = 0
while len(result) < max_extra and active:
    rr_round += 1
    next_active = []
    for sk in active:
        if len(result) >= max_extra: break
        nbrs = seed_nbrs[sk]
        while cursor[sk] < len(nbrs) and nbrs[cursor[sk]][0] in selected_set:
            cursor[sk] += 1
        if cursor[sk] < len(nbrs):
            ck, sc = nbrs[cursor[sk]]
            is_target = "***TARGET***" if ck in targets else ""
            print(f"  Round {rr_round}: seed={sk}, picked={ck} score={sc:.3f} {is_target}")
            result.append(ck)
            selected_set.add(ck)
            cursor[sk] += 1
            next_active.append(sk)
    if not next_active: break
    active = next_active

print(f"\nTotal selected: {len(result)}")
print(f"Target in selected: {targets & selected_set}")
print(f"Target missed: {targets - selected_set}")
