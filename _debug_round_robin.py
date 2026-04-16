"""Debug: trace per-seed neighbor rankings for Q42 and Q30."""
import os, sys, json, pickle
from collections import defaultdict
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "2"
os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = "16"

from src.assist import build_instance_fragments, calculate_tfidf_similarity
from src.data.graph import ClassGraph, normalize_edge_leg
from src.config_loader import get_query_neighbor_traversal_config

BASE = "/Users/zachary/Workspace/LongtermMemory"
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg = pickle.load(f)
cg.process_kw(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json")

# Target entities from previous analysis
targets = {
    "Q42": ["class_144:instance_1", "class_3:instance_1", "class_69:instance_1", "class_81:instance_1"],
    "Q30": ["class_3:instance_1", "class_14:instance_1", "class_81:instance_1", "class_144:instance_1"],
}
queries = {
    "Q42": "What are the breeds of Audrey's dogs?",
    "Q30": "What are the names of Audrey's dogs?",
}

max_hops, max_extra, legs = get_query_neighbor_traversal_config()

for qid, query in queries.items():
    print(f"\n{'='*70}")
    print(f"{qid}: {query}")

    # Run TF-IDF retrieval to get seeds
    ctx, trace = cg._search_by_sub_hash(query)
    seeds = set(trace.get("retrieved_entity_ids", []))
    print(f"Seeds ({len(seeds)}): {sorted(seeds)[:10]}...")

    # Build adjacency
    adj = cg._build_instance_adjacency(legs)

    # Check which seeds each target is adjacent to
    for tgt in targets[qid]:
        adj_seeds = [s for s in seeds if tgt in adj.get(s, set())]
        print(f"\n  Target {tgt}: adjacent to {len(adj_seeds)} seeds: {adj_seeds}")

        if not adj_seeds:
            print(f"    NOT REACHABLE from any seed!")
            continue

        # For each seed, show the target's rank
        for sk in adj_seeds:
            nbrs = list(adj.get(sk, set()) - seeds)
            # Score function (same as in _neighbor_bfs_ranked)
            q_lower = query.lower()
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

            # Build edge text index
            edge_texts_by_key = defaultdict(list)
            for rec in cg.edges or []:
                leg = normalize_edge_leg(rec.get("edge_leg"))
                if leg not in legs:
                    continue
                content = (rec.get("content") or "").lower()
                if not content:
                    continue
                for c in rec.get("connections") or []:
                    ocid = c.get("class_id")
                    oid = c.get("instance_id")
                    if ocid and oid is not None:
                        edge_texts_by_key[cg._instance_key(str(ocid), oid)].append(content)

            # Build inst text
            key_to_inst = {}
            for node in cg.graph.nodes:
                cid = getattr(node, "class_id", None) or ""
                for inst in getattr(node, "_instances", []) or []:
                    iid = inst.get("instance_id")
                    if iid is not None:
                        key_to_inst[cg._instance_key(cid, iid)] = inst

            def _inst_text(key):
                inst = key_to_inst.get(key)
                if inst is None:
                    return ""
                parts = [tx for _ty, tx in build_instance_fragments(inst) if (tx or "").strip()]
                return "\n".join(parts) if parts else ""

            def _score(ck):
                edge_hits = 0
                for etxt in edge_texts_by_key.get(ck, []):
                    edge_hits += sum(1 for kw in q_words if kw in etxt)
                edge_sc = min(edge_hits / max(len(q_words), 1), 1.0)
                inst_txt = _inst_text(ck).lower()
                inst_hits = sum(1 for kw in q_words if kw in inst_txt)
                inst_sc = min(inst_hits / max(len(q_words), 1), 1.0)
                return 0.5 * inst_sc + 0.5 * edge_sc

            scored = [(n, _score(n)) for n in nbrs]
            scored.sort(key=lambda x: -x[1])

            # Find target's rank
            tgt_score = _score(tgt)
            rank = next((i for i, (n, s) in enumerate(scored) if n == tgt), -1)
            print(f"    Seed {sk}: {len(scored)} neighbors, target rank={rank+1}/{len(scored)}, score={tgt_score:.3f}")
            if rank >= 0 and rank < 5:
                print(f"      Top-5: {[(n, f'{s:.3f}') for n, s in scored[:5]]}")
            elif rank >= 0:
                print(f"      Top-3: {[(n, f'{s:.3f}') for n, s in scored[:3]]}")
                print(f"      Target context: {[(n, f'{s:.3f}') for n, s in scored[max(0,rank-1):rank+2]]}")
