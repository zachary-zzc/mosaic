"""Debug: trace exact scores for Q42 and Q30 candidates."""
import os, sys, json, pickle
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "2"
os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = "16"

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim
from collections import defaultdict, deque

from src.assist import build_instance_fragments, calculate_tfidf_similarity
from src.data.graph import ClassGraph
from src.config_loader import get_query_neighbor_traversal_config
from src.data.dual_graph import normalize_edge_leg, ALL_EDGE_LEGS

BASE = "/Users/zachary/Workspace/LongtermMemory"
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg = pickle.load(f)
tag_path = f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json"
cg.process_kw(tag_path)

def eid_to_key(eid):
    parts = eid.split(":")
    return f"{parts[0]}_{parts[1]}" if len(parts) == 2 else eid

def key_to_eid(key):
    # class_X_instance_Y -> class_X:instance_Y
    parts = key.split("_")
    if len(parts) >= 3 and parts[0] == "class":
        cid = f"{parts[0]}_{parts[1]}"
        iid = "_".join(parts[2:])
        return f"{cid}:{iid}"
    return key

# For Q42 manually reproduce the scoring
for q_label, q_text, answer_targets in [
    ("Q42", "What are the breeds of Audrey's dogs?", 
     ["class_144:instance_1", "class_3:instance_1", "class_69:instance_1", "class_81:instance_1"]),
    ("Q30", "What are the names of Audrey's dogs?",
     ["class_3:instance_1", "class_14:instance_1", "class_81:instance_1", "class_144:instance_1"]),
]:
    print(f"\n{'='*70}")
    print(f"{q_label}: {q_text}")
    
    # Get seeds via TF-IDF retrieval
    ctx, trace = cg._search_by_sub_hash(q_text)
    retrieved_eids = trace.get("retrieved_entity_ids", [])
    seeds = set(eid_to_key(e) for e in retrieved_eids)
    
    # Build adjacency
    allowed_legs = frozenset(ALL_EDGE_LEGS)
    adj = cg._build_instance_adjacency(allowed_legs)
    
    # Build edge content index
    q_lower = q_text.lower()
    q_words = set()
    for w in q_lower.split():
        w = w.strip(".,!?()[]\"':;")
        if len(w) > 2:
            q_words.add(w)
    _stop = {"the", "and", "for", "are", "was", "were", "has", "have", "had",
             "that", "this", "with", "from", "what", "which", "who", "how",
             "does", "did", "not", "been", "but", "they", "them", "than",
             "can", "her", "his", "she", "you", "all", "any", "some", "will"}
    q_words -= _stop
    
    edge_texts_by_key = defaultdict(list)
    for rec in cg.edges or []:
        leg = normalize_edge_leg(rec.get("edge_leg"))
        if leg not in allowed_legs:
            continue
        content = (rec.get("content") or "").lower()
        if not content:
            continue
        for c in rec.get("connections") or []:
            ocid = c.get("class_id")
            oid = c.get("instance_id")
            if ocid and oid is not None:
                edge_texts_by_key[cg._instance_key(str(ocid), oid)].append(content)
    
    # BFS
    visited = {s: 0 for s in seeds}
    frontier = deque((s, 0) for s in seeds)
    candidates = []
    candidate_hop = {}
    max_hops = 2
    
    def _kw_hits(key):
        inst = None
        for node in cg.graph.nodes:
            cid = getattr(node, "class_id", "")
            for i in getattr(node, "_instances", []) or []:
                if cg._instance_key(cid, i.get("instance_id", "")) == key:
                    inst = i
                    break
        txt = ""
        if inst:
            parts = [tx for _ty, tx in build_instance_fragments(inst) if (tx or "").strip()]
            txt = "\n".join(parts).lower()
        hits = sum(1 for kw in q_words if kw in txt)
        for etxt in edge_texts_by_key.get(key, []):
            hits += sum(1 for kw in q_words if kw in etxt)
        return hits
    
    while frontier:
        u, d = frontier.popleft()
        if d >= max_hops:
            continue
        nd = d + 1
        for v in adj.get(u, ()):
            if v in visited:
                continue
            visited[v] = nd
            if v not in seeds:
                candidates.append(v)
                candidate_hop[v] = nd
            if nd < max_hops:
                if nd == 1 or _kw_hits(v) >= 1:
                    frontier.append((v, nd))
    
    hop1_count = sum(1 for v in candidates if candidate_hop.get(v) == 1)
    hop2_count = sum(1 for v in candidates if candidate_hop.get(v) == 2)
    print(f"  Candidates: {len(candidates)} total (hop1={hop1_count}, hop2={hop2_count})")
    
    # Check if targets are in candidates
    target_keys = [eid_to_key(t) for t in answer_targets]
    for tk, te in zip(target_keys, answer_targets):
        in_cand = tk in candidates
        hop = candidate_hop.get(tk, -1)
        print(f"  {te}: in_candidates={in_cand}, hop={hop}")
    
    # Show the selected 16 neighbors
    selected = trace.get("neighbor_expansion", {}).get("entity_ids", [])
    target_eid_set = set(answer_targets)
    for se in selected:
        marker = " <-- ANSWER" if se in target_eid_set else ""
        sk = eid_to_key(se)
        hop = candidate_hop.get(sk, "?")
        print(f"  Selected: {se} (hop={hop}){marker}")
