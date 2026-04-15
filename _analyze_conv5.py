"""Analyze conv5 graph for multi-hop retrieval failures."""
import sys, re, json
sys.path.insert(0, 'mosaic')

from src.assist import load_mosaic_memory_pickle
from src.data.graph import build_instance_fragments

GRAPH_PATH = 'experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl'
TAGS_PATH = 'experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json'

memory = load_mosaic_memory_pickle(GRAPH_PATH)
memory.process_kw(TAGS_PATH)

print(f'Graph nodes (classes): {len(memory.graph.nodes)}')
total_instances = sum(len(getattr(n, '_instances', []) or []) for n in memory.graph.nodes)
print(f'Total instances: {total_instances}')
print(f'Edges: {len(memory.edges or [])}')

# Build full text index for each entity
text_index = {}
for node in memory.graph.nodes:
    cid = getattr(node, 'class_id', '')
    for inst in getattr(node, '_instances', []) or []:
        iid = inst.get('instance_id', '')
        frags = build_instance_fragments(inst)
        full_text = ' '.join(t for _, t in frags if t)
        eid = f'{cid}:{iid}'
        text_index[eid] = full_text

# Multi-hop questions and their answers for conv5
questions = [
    ("What is an indoor activity that Andrew would enjoy doing while make his dog happy?", "cook dog treats"),
    ("Which meat does Audrey prefer eating more than others?", "chicken"),
    ("What can Andrew potentially do to improve his stress and accomodate his living situation with his dogs?", "Change to a hybrid or remote job"),
    ("Which US state do Audrey and Andrew potentially live in?", "Minnesota"),
    ("Which national park could Audrey and Andrew be referring to?", "Voyageurs National Park"),
    ("What is something that Andrew could do to make birdwatching hobby to fit in his city schedule?", "bird feeder"),
]

for q, ans in questions:
    print(f'\n{"="*70}')
    print(f'Q: {q}')
    print(f'Gold: {ans}')
    
    # Search for entities containing answer keywords
    keywords = [w for w in re.findall(r'\b\w{4,}\b', ans)]
    matching = []
    for eid, text in text_index.items():
        for kw in keywords:
            if re.search(re.escape(kw), text, re.I):
                matching.append((eid, kw, text[:150]))
                break
    
    print(f'  Entities containing answer keywords ({keywords}): {len(matching)}')
    for eid, kw, txt in matching[:5]:
        print(f'    {eid} [matched: {kw}] -> {txt}')
    
    # Now check what was actually retrieved
    # Run retrieval
    ctx, trace = memory._search_by_sub_hash(q)
    retrieved = set(trace.get('retrieved_entity_ids', []))
    neighbor_ids = set(trace.get('neighbor_expansion', {}).get('entity_ids', []))
    all_retrieved = retrieved | neighbor_ids
    
    # Check if answer entities were retrieved
    matched_eids = set(eid for eid, _, _ in matching)
    found_in_retrieval = matched_eids & all_retrieved
    missed = matched_eids - all_retrieved
    
    print(f'  Retrieved entities: {len(retrieved)}, Neighbors: {len(neighbor_ids)}')
    print(f'  Answer entities found in retrieval: {len(found_in_retrieval)}/{len(matched_eids)}')
    if missed:
        print(f'  MISSED answer entities: {missed}')
        # Check if missed entities have edges to any retrieved entity
        from src.data.dual_graph import ALL_EDGE_LEGS
        adj = memory._build_instance_adjacency(frozenset(ALL_EDGE_LEGS))
        for m_eid in list(missed)[:3]:
            m_ikey = None
            for node in memory.graph.nodes:
                cid = getattr(node, 'class_id', '')
                for inst in getattr(node, '_instances', []) or []:
                    iid = inst.get('instance_id', '')
                    if f'{cid}:{iid}' == m_eid:
                        m_ikey = memory._instance_key(cid, iid)
                        break
                if m_ikey:
                    break
            if m_ikey:
                neighbors_of_missed = adj.get(m_ikey, set())
                connected_to_retrieved = False
                for r_eid in all_retrieved:
                    r_ikey = None
                    for node in memory.graph.nodes:
                        cid2 = getattr(node, 'class_id', '')
                        for inst2 in getattr(node, '_instances', []) or []:
                            iid2 = inst2.get('instance_id', '')
                            if f'{cid2}:{iid2}' == r_eid:
                                r_ikey = memory._instance_key(cid2, iid2)
                                break
                        if r_ikey:
                            break
                    if r_ikey and r_ikey in neighbors_of_missed:
                        connected_to_retrieved = True
                        break
                print(f'    {m_eid}: {len(neighbors_of_missed)} graph neighbors, connected to retrieved: {connected_to_retrieved}')
            else:
                print(f'    {m_eid}: entity key not found')
    
    if found_in_retrieval:
        print(f'  FOUND answer entities: {found_in_retrieval}')
