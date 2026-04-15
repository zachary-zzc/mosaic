"""Survey conv4 and conv3 multi-hop failures - deep graph + retrieval analysis."""
import sys, re, json, statistics
from collections import defaultdict
sys.path.insert(0, 'mosaic')
from src.assist import load_mosaic_memory_pickle
from src.data.graph import build_instance_fragments
from src.data.dual_graph import ALL_EDGE_LEGS

def analyze_conv(conv_id):
    base = f'experiments/locomo/benchmark/runs/runs/conv{conv_id}'
    graph_path = f'{base}/artifacts/hybrid/graph_network_conv{conv_id}.pkl'
    tags_path = f'{base}/artifacts/hybrid/conv{conv_id}_tags.json'
    eval_path = f'{base}/results/qa_{conv_id}_eval_full_hybrid.json'

    print(f'\n{"#"*80}')
    print(f'# CONV{conv_id} ANALYSIS')
    print(f'{"#"*80}')

    memory = load_mosaic_memory_pickle(graph_path)
    memory.process_kw(tags_path)

    with open(eval_path) as f:
        eval_data = json.load(f)

    adj = memory._build_instance_adjacency(frozenset(ALL_EDGE_LEGS))

    # Build indexes
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

    total_entities = len(eid_to_ikey)
    disconnected = [eid for eid, ikey in eid_to_ikey.items() if not adj.get(ikey, set())]

    # Degree stats
    degrees = [len(adj.get(ikey, set())) for ikey in eid_to_ikey.values()]
    degrees.sort()

    # Edge leg distribution
    edge_legs = defaultdict(int)
    for rec in memory.edges or []:
        edge_legs[rec.get('edge_leg', 'unknown')] += 1

    print(f'\n--- Graph Structure ---')
    print(f'Classes: {len(memory.graph.nodes)}, Entities: {total_entities}')
    print(f'Edge records: {len(memory.edges or [])} (A={edge_legs.get("A",0)}, P={edge_legs.get("P",0)})')
    print(f'Disconnected: {len(disconnected)}/{total_entities} ({len(disconnected)/total_entities*100:.1f}%)')
    print(f'Degree: min={min(degrees)} max={max(degrees)} mean={statistics.mean(degrees):.1f} median={statistics.median(degrees):.1f}')
    print(f'Degree 0: {degrees.count(0)}, 1-5: {sum(1 for d in degrees if 1<=d<=5)}, 6-15: {sum(1 for d in degrees if 6<=d<=15)}, 16+: {sum(1 for d in degrees if d>=16)}')

    if disconnected:
        print(f'\nDisconnected entities:')
        for eid in disconnected:
            print(f'  {eid}: {eid_to_text.get(eid,"")[:90]}')

    # Multi-hop question analysis
    multihop_qs = [r for r in eval_data['results'] if r.get('category') == 3]
    wrong_qs = [r for r in multihop_qs if r.get('judgment') == 'WRONG']

    print(f'\n--- Multi-hop QA ---')
    print(f'Total: {len(multihop_qs)}, Correct: {len(multihop_qs)-len(wrong_qs)}, Wrong: {len(wrong_qs)}')

    for r in wrong_qs:
        question = r['question']
        gold = r['answer']
        ctx = r.get('retrieved_context', {})
        retrieved = set(ctx.get('retrieved_entity_ids', []))
        neighbor_ids = set(ctx.get('neighbor_expansion', {}).get('entity_ids', []))
        all_retrieved = retrieved | neighbor_ids

        print(f'\n  Q: {question}')
        print(f'  Gold: {gold}')
        print(f'  Gen: {r["generated_answer"][:150]}...')

        # Find entities containing answer keywords
        keywords = [w for w in re.findall(r'\b\w{4,}\b', gold)]
        if not keywords:
            keywords = [w for w in re.findall(r'\b\w{3,}\b', gold)]

        matching_eids = set()
        for eid, text in eid_to_text.items():
            for kw in keywords:
                if re.search(re.escape(kw), text, re.I):
                    matching_eids.add(eid)
                    break

        found_in_retrieval = matching_eids & all_retrieved
        missed = matching_eids - all_retrieved

        print(f'  Answer entities in graph: {len(matching_eids)}, Retrieved: {len(found_in_retrieval)}, Missed: {len(missed)}')

        # Classify the failure
        if len(matching_eids) == 0:
            print(f'  FAILURE TYPE: Answer content NOT in graph at all')
        elif len(found_in_retrieval) > 0 and len(found_in_retrieval) == len(matching_eids):
            print(f'  FAILURE TYPE: LLM judgment/generation error (all entities retrieved)')
        elif len(found_in_retrieval) > 0:
            print(f'  FAILURE TYPE: Partial retrieval ({len(found_in_retrieval)}/{len(matching_eids)} entities)')
        else:
            print(f'  FAILURE TYPE: Complete retrieval miss')

        # For missed entities, check connectivity
        disconnected_misses = 0
        connected_to_retrieved_misses = 0
        for m_eid in list(missed)[:5]:
            m_ikey = eid_to_ikey.get(m_eid)
            if not m_ikey:
                continue
            m_neighbors = adj.get(m_ikey, set())
            if not m_neighbors:
                disconnected_misses += 1
                print(f'    MISSED (disconnected): {m_eid}: {eid_to_text.get(m_eid,"")[:80]}')
            else:
                # Check if connected to any retrieved entity
                connected = False
                for re_eid in all_retrieved:
                    re_ikey = eid_to_ikey.get(re_eid)
                    if re_ikey and re_ikey in m_neighbors:
                        connected = True
                        break
                if connected:
                    connected_to_retrieved_misses += 1
                    print(f'    MISSED (connected to retrieved, but not expanded): {m_eid}: {eid_to_text.get(m_eid,"")[:80]}')
                else:
                    # Check 2-hop
                    hop1 = m_neighbors
                    hop2 = set()
                    for n in hop1:
                        hop2.update(adj.get(n, set()))
                    connected_2hop = False
                    for re_eid in all_retrieved:
                        re_ikey = eid_to_ikey.get(re_eid)
                        if re_ikey and re_ikey in hop2:
                            connected_2hop = True
                            break
                    if connected_2hop:
                        print(f'    MISSED (2-hop from retrieved): {m_eid}: {eid_to_text.get(m_eid,"")[:80]}')
                    else:
                        print(f'    MISSED (>2 hops from retrieved): {m_eid}: {eid_to_text.get(m_eid,"")[:80]}')

    # Check cross-topic edge connectivity
    print(f'\n--- Cross-topic Edge Analysis ---')
    # Sample: do entities from different dialogue sessions connect?
    # Check edge source info
    same_session_edges = 0
    cross_session_edges = 0
    for rec in memory.edges or []:
        conns = rec.get('connections', [])
        if len(conns) < 2:
            continue
        # Check if connections span multiple class IDs (proxy for different topics)
        cids = set(str(c.get('class_id', '')) for c in conns)
        if len(cids) > 1:
            cross_session_edges += 1
        else:
            same_session_edges += 1

    print(f'Edges connecting different classes: {cross_session_edges}')
    print(f'Edges within same class: {same_session_edges}')

    return {
        'conv_id': conv_id,
        'total_entities': total_entities,
        'disconnected': len(disconnected),
        'total_edges': len(memory.edges or []),
        'edge_A': edge_legs.get('A', 0),
        'edge_P': edge_legs.get('P', 0),
        'multihop_total': len(multihop_qs),
        'multihop_wrong': len(wrong_qs),
    }


if __name__ == '__main__':
    r4 = analyze_conv(4)
    r3 = analyze_conv(3)

    print(f'\n{"#"*80}')
    print('# COMPARISON SUMMARY')
    print(f'{"#"*80}')
    for r in [r4, r3]:
        print(f'Conv{r["conv_id"]}: entities={r["total_entities"]}, disconnected={r["disconnected"]}, '
              f'edges={r["total_edges"]} (A={r["edge_A"]}/P={r["edge_P"]}), '
              f'multihop={r["multihop_total"]-r["multihop_wrong"]}/{r["multihop_total"]}')
