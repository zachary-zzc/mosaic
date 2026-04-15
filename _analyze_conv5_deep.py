"""Deep analysis of conv5 graph connectivity for multi-hop failures."""
import sys
sys.path.insert(0, 'mosaic')
from src.assist import load_mosaic_memory_pickle
from src.data.graph import build_instance_fragments
from src.data.dual_graph import ALL_EDGE_LEGS
from collections import defaultdict

GRAPH_PATH = 'experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl'
TAGS_PATH = 'experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json'

memory = load_mosaic_memory_pickle(GRAPH_PATH)
memory.process_kw(TAGS_PATH)

adj = memory._build_instance_adjacency(frozenset(ALL_EDGE_LEGS))

ikey_to_eid = {}
eid_to_ikey = {}
eid_to_text = {}
for node in memory.graph.nodes:
    cid = getattr(node, 'class_id', '')
    for inst in getattr(node, '_instances', []) or []:
        iid = inst.get('instance_id', '')
        ikey = memory._instance_key(cid, iid)
        eid = f'{cid}:{iid}'
        ikey_to_eid[ikey] = eid
        eid_to_ikey[eid] = ikey
        frags = build_instance_fragments(inst)
        eid_to_text[eid] = ' '.join(t for _, t in frags if t)

# 1. Check class_52:instance_1 (Roasted Chicken - DISCONNECTED)
eid = 'class_52:instance_1'
print(f'=== {eid} ===')
text = eid_to_text.get(eid, 'NOT FOUND')
print(f'Text: {text[:300]}')
ikey = eid_to_ikey.get(eid)
if ikey:
    neighbors = adj.get(ikey, set())
    print(f'Neighbors: {len(neighbors)}')
    for n in list(neighbors)[:5]:
        neid = ikey_to_eid.get(n, n)
        print(f'  -> {neid}: {eid_to_text.get(neid, "")[:100]}')
else:
    print('No instance key found')

# 2. Disconnected entity stats
total = len(eid_to_ikey)
disconnected = sum(1 for eid, ikey in eid_to_ikey.items() if not adj.get(ikey, set()))
print(f'\nTotal entities: {total}')
print(f'Disconnected (0 neighbors): {disconnected}')
print(f'Disconnected rate: {disconnected/total*100:.1f}%')

# List disconnected entities
print('\nDisconnected entities:')
for eid, ikey in eid_to_ikey.items():
    if not adj.get(ikey, set()):
        print(f'  {eid}: {eid_to_text.get(eid, "")[:100]}')

# 3. Cooking ↔ Dog connectivity
import re
cooking_eids = [e for e, t in eid_to_text.items() if re.search(r'cook|recipe|kitchen', t, re.I)]
dog_eids = [e for e, t in eid_to_text.items() if re.search(r'\bdog\b|\btoby\b|\bpuppy\b|\bpup\b', t, re.I)]

print(f'\nCooking entities: {len(cooking_eids)}')
for ce in cooking_eids:
    print(f'  {ce}: {eid_to_text[ce][:80]}')

print(f'\nDog entities: {len(dog_eids)}')

# Check cooking→dog 1-hop connectivity
print('\nCooking → Dog connectivity:')
for ce in cooking_eids:
    c_ikey = eid_to_ikey.get(ce)
    if not c_ikey:
        continue
    c_neighbors = adj.get(c_ikey, set())
    c_neighbor_eids = {ikey_to_eid.get(n, n) for n in c_neighbors}
    dog_1hop = set(dog_eids) & c_neighbor_eids
    print(f'  {ce} ({len(c_neighbors)} neighbors): {len(dog_1hop)} dog entities at 1-hop')
    if dog_1hop:
        for d in list(dog_1hop)[:3]:
            print(f'    -> {d}: {eid_to_text[d][:80]}')

# 4. For the retrieval of Q1, check what was actually retrieved vs what should have been
print('\n=== Q1 Retrieval Analysis ===')
q1 = "What is an indoor activity that Andrew would enjoy doing while make his dog happy?"
ctx, trace = memory._search_by_sub_hash(q1)
retrieved = set(trace.get('retrieved_entity_ids', []))
neighbor_ids = set(trace.get('neighbor_expansion', {}).get('entity_ids', []))

print(f'Retrieved TF-IDF: {len(retrieved)}')
print(f'Neighbor expansion: {len(neighbor_ids)}')
all_retrieved = retrieved | neighbor_ids
print(f'Total unique: {len(all_retrieved)}')

# Check: are cooking entities in retrieval?
for ce in cooking_eids:
    r = 'YES' if ce in all_retrieved else 'NO'
    via = ''
    if ce in retrieved:
        via = '(TF-IDF)'
    elif ce in neighbor_ids:
        via = '(neighbor)'
    print(f'  {ce} retrieved: {r} {via}')

# 5. Edge analysis stats
print('\n=== Edge Statistics ===')
print(f'Total edges records: {len(memory.edges or [])}')
edge_legs = defaultdict(int)
for rec in memory.edges or []:
    leg = rec.get('edge_leg', 'unknown')
    edge_legs[leg] += 1
for leg, count in sorted(edge_legs.items()):
    print(f'  {leg}: {count}')

# Degree distribution
degrees = []
for eid, ikey in eid_to_ikey.items():
    d = len(adj.get(ikey, set()))
    degrees.append(d)

degrees.sort()
import statistics
print(f'\nDegree stats: min={min(degrees)} max={max(degrees)} mean={statistics.mean(degrees):.1f} median={statistics.median(degrees):.1f}')
print(f'Entities with degree 0: {degrees.count(0)}')
print(f'Entities with degree 1-5: {sum(1 for d in degrees if 1 <= d <= 5)}')
print(f'Entities with degree 6-15: {sum(1 for d in degrees if 6 <= d <= 15)}')
print(f'Entities with degree 16+: {sum(1 for d in degrees if d >= 16)}')
