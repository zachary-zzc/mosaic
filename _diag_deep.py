"""
Deep diagnostic: for each failing multi-hop question, show exactly WHY it fails.
- Are answer entities in the graph at all?
- Are they reachable from seeds (and at what hop distance)?
- What's the TF-IDF / BGE similarity of answer entities to query?
- Would increasing hops or max_extra help?
"""
import sys, os, json, pickle, importlib
from collections import defaultdict
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

BASE = "/Users/zachary/Workspace/LongtermMemory"
os.environ["MOSAIC_QUERY_NEIGHBOR_MMR_LAMBDA"] = "0.5"

from src.config_loader import get_query_neighbor_traversal_config
from src.assist import build_instance_fragments, calculate_tfidf_similarity
from src.data.dual_graph import ALL_EDGE_LEGS
from collections import deque

def load_conv(conv_id):
    for mode in ["hash_only", "hybrid"]:
        pkl_path = os.path.join(BASE, f"experiments/locomo/benchmark/runs/runs/conv{conv_id}/artifacts/{mode}/graph_network_conv{conv_id}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                cg = pickle.load(f)
            break
    with open(os.path.join(BASE, f"dataset/locomo/qa_{conv_id}.json")) as f:
        qa = json.load(f)
    return cg, qa

def find_answer_entities(cg, answer):
    """Find entities whose text contains answer keywords."""
    answer_lower = answer.lower()
    words = [w.strip(".,!?()[]\"'") for w in answer_lower.split()]
    sig_words = [w for w in words if len(w) > 3]
    
    matches = []
    for class_node in cg.graph.nodes:
        cid = getattr(class_node, "class_id", "") or ""
        for inst in getattr(class_node, "_instances", []) or []:
            iid = inst.get("instance_id")
            if not iid:
                continue
            eid = f"{cid}:{iid}"
            parts = [tx for _ty, tx in build_instance_fragments(inst) if (tx or "").strip()]
            text = "\n".join(parts).lower()
            found_words = [w for w in sig_words if w in text]
            if len(found_words) >= max(1, len(sig_words) * 0.3):
                matches.append((eid, cg._instance_key(cid, iid), len(found_words), len(sig_words), text[:100]))
    return matches

def bfs_distance(adj, seeds, target_keys, max_d=5):
    """BFS from seeds, return distance to each target key (or -1 if unreachable)."""
    visited = {s: 0 for s in seeds}
    queue = deque((s, 0) for s in seeds)
    while queue:
        u, d = queue.popleft()
        if d >= max_d:
            continue
        for v in adj.get(u, ()):
            if v not in visited:
                visited[v] = d + 1
                queue.append((v, d + 1))
    return {tk: visited.get(tk, -1) for tk in target_keys}

def count_neighbors_at_hop(adj, seeds, max_hops):
    """Count how many unique neighbors at each hop distance."""
    visited = {s: 0 for s in seeds}
    queue = deque((s, 0) for s in seeds)
    hop_counts = defaultdict(int)
    while queue:
        u, d = queue.popleft()
        if d >= max_hops:
            continue
        for v in adj.get(u, ()):
            if v not in visited:
                visited[v] = d + 1
                queue.append((v, d + 1))
                if v not in seeds:
                    hop_counts[d + 1] += 1
    return dict(hop_counts)

for conv_id in [3, 4, 5]:
    cg, qa = load_conv(conv_id)
    multihop = [q for q in qa if q.get("category") == 3]
    
    print(f"\n{'='*80}")
    print(f"  CONV{conv_id}: {len(multihop)} multi-hop questions")
    print(f"{'='*80}")
    
    # Build adjacency once
    adj = cg._build_instance_adjacency(frozenset(ALL_EDGE_LEGS))
    
    for qi, q_item in enumerate(multihop):
        question = q_item["question"]
        answer = q_item["answer"]
        
        # Get seeds
        cg.selected_instance_keys.clear()
        cg._adj_cache = None
        sensed = cg._sense_classes_by_tfidf(question, 10, threshold=0.6, allow_below_threshold=True)
        cg._fetch_instances_by_tfidf(question, 15, threshold=0.5, classes=sensed)
        cg._fetch_instances_by_tfidf(question, 15, threshold=0.1)
        seeds = set(cg.selected_instance_keys)
        
        # Find answer entities
        ans_ents = find_answer_entities(cg, answer)
        
        # Check reachability from seeds
        target_keys = [ae[1] for ae in ans_ents]
        distances = bfs_distance(adj, seeds, target_keys) if target_keys else {}
        
        # Check neighbor expansion results with current settings
        expanded_keys = cg._neighbor_expansion_key_list(seeds, query=question)
        expanded_set = set(expanded_keys)
        
        # Count neighbors at each hop
        hop_counts = count_neighbors_at_hop(adj, seeds, 3)
        
        # Check if retrieved (seeds + expanded)
        all_retrieved = seeds | expanded_set
        
        status_parts = []
        for eid, ikey, nfound, ntotal, text_preview in ans_ents:
            dist = distances.get(ikey, -1)
            in_seeds = ikey in seeds
            in_expanded = ikey in expanded_set
            status = "SEED" if in_seeds else ("EXPANDED" if in_expanded else f"MISSED(d={dist})")
            status_parts.append(f"    {eid}: {status} | dist={dist} | words={nfound}/{ntotal}")
        
        # Determine overall status
        if not ans_ents:
            overall = "NOT_IN_GRAPH"
        elif any(k in all_retrieved for _, k, _, _, _ in ans_ents):
            overall = "RETRIEVED"
        elif any(distances.get(k, -1) == 1 for _, k, _, _, _ in ans_ents):
            overall = "FAIL:1-hop_reachable_but_not_selected"
        elif any(distances.get(k, -1) == 2 for _, k, _, _, _ in ans_ents):
            overall = "FAIL:2-hop_away"
        elif any(distances.get(k, -1) > 2 for _, k, _, _, _ in ans_ents):
            overall = "FAIL:3+_hops_away"
        elif any(distances.get(k, -1) == -1 for _, k, _, _, _ in ans_ents):
            overall = "FAIL:unreachable(disconnected)"
        else:
            overall = "FAIL:unknown"
        
        print(f"\n  Q{qi+1}: {question[:85]}")
        print(f"  A: {answer[:70]}")
        print(f"  Status: {overall}")
        print(f"  Seeds={len(seeds)} | Hop-1={hop_counts.get(1,0)} | Hop-2={hop_counts.get(2,0)} | Hop-3={hop_counts.get(3,0)}")
        print(f"  Expanded(MMR,max16)={len(expanded_keys)}")
        if ans_ents:
            print(f"  Answer entities ({len(ans_ents)}):")
            for line in status_parts:
                print(line)
        else:
            print(f"  Answer entities: NONE FOUND IN GRAPH")

print("\n\n" + "="*80)
print("SUMMARY OF FAILURE PATTERNS:")
print("="*80)
