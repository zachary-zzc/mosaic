"""Audit conv5 multi-hop failures: diagnose seed selection + graph construction gaps.

For each failing query, check:
1. GRAPH: Are answer entities in the graph? What class/instance?
2. SEEDS: Which classes/instances does TF-IDF select? Why are answer entities missed?
3. EDGES: Are answer entities connected to anything? What edges exist?
"""
import os, sys, json, pickle
from collections import defaultdict
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")
os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = "2"
os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = "16"

from src.assist import build_instance_fragments, calculate_tfidf_similarity
from src.data.graph import ClassGraph, normalize_edge_leg

BASE = "/Users/zachary/Workspace/LongtermMemory"
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg = pickle.load(f)
cg.process_kw(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json")

with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/results/qa_5_eval_full_hybrid.json") as f:
    eval_data = json.load(f)

# Build full instance text index with class info
inst_index = {}  # eid -> {text, class_id, instance_id, class_name, tags}
for node in cg.graph.nodes:
    cid = getattr(node, "class_id", "")
    cname = getattr(node, "theme", "") or getattr(node, "name", "") or ""
    tags = getattr(node, "tags", []) or []
    for inst in getattr(node, "_instances", []) or []:
        iid = inst.get("instance_id", "")
        eid = f"{cid}:{iid}"
        frags = build_instance_fragments(inst)
        text = "\n".join(t for _ty, t in frags if t and t.strip())
        inst_index[eid] = {
            "text": text, "class_id": cid, "instance_id": iid,
            "class_name": cname, "tags": tags,
            "text_lower": text.lower(),
        }

# Build edge index
edge_index = defaultdict(list)  # eid -> [(other_eid, edge_leg, content_snippet)]
for rec in cg.edges or []:
    leg = rec.get("edge_leg", "")
    content = (rec.get("content") or "")[:100]
    conns = rec.get("connections") or []
    conn_eids = []
    for c in conns:
        ocid = c.get("class_id")
        oid = c.get("instance_id")
        if ocid and oid is not None:
            conn_eids.append(f"class_{ocid}:instance_{oid}")
    for i, e1 in enumerate(conn_eids):
        for e2 in conn_eids[i+1:]:
            edge_index[e1].append((e2, leg, content))
            edge_index[e2].append((e1, leg, content))

cat1_wrong = [r for r in eval_data["results"] if r["category"] == 1 and r["judgment"] == "WRONG"]

print(f"{'='*80}")
print(f"AUDIT: {len(cat1_wrong)} failing multi-hop queries on conv5")
print(f"Graph: {len(inst_index)} instances, {len(cg.edges or [])} edges")
print(f"{'='*80}\n")

for r in cat1_wrong:
    q = r["question"]
    gold = r["answer"]
    qi = r["qa_source_index"] + 1
    
    print(f"\n{'─'*80}")
    print(f"Q{qi}: {q}")
    print(f"Gold: {gold[:120]}")
    
    # 1. Find answer entities in graph
    gold_lower = gold.lower()
    gold_words = [w.strip(".,!?()[]\"'") for w in gold_lower.split() if len(w.strip(".,!?()[]\"'")) > 3]
    answer_eids = set()
    for eid, info in inst_index.items():
        found = [w for w in gold_words if w in info["text_lower"]]
        if len(found) >= max(1, len(gold_words) * 0.3):
            answer_eids.add(eid)
    
    if not answer_eids:
        print(f"  [GRAPH GAP] No answer entities in graph! Gold words: {gold_words[:8]}")
        # Search for partial matches
        partial = []
        for eid, info in inst_index.items():
            found = [w for w in gold_words if w in info["text_lower"]]
            if found:
                partial.append((eid, len(found), info["class_name"][:40]))
        partial.sort(key=lambda x: -x[1])
        if partial:
            print(f"  Closest partial matches:")
            for eid, cnt, cn in partial[:5]:
                print(f"    {eid} ({cn}): {cnt}/{len(gold_words)} gold words")
        continue
    
    print(f"  Answer entities: {len(answer_eids)}")
    for eid in sorted(answer_eids)[:6]:
        info = inst_index[eid]
        edges = edge_index.get(eid, [])
        print(f"    {eid} [{info['class_name'][:40]}] edges={len(edges)} tags={info['tags'][:3]}")
        print(f"      text: {info['text'][:120]}...")
    
    # 2. Run TF-IDF retrieval to see what gets selected
    cg._adj_cache = None
    cg.selected_instance_keys.clear()
    
    # Stage 1: class sensing
    sensed = cg._sense_classes_by_tfidf(q, 10, threshold=0.6, allow_below_threshold=True)
    sensed_cids = [str(c.get("class_id")) for c in sensed.get("selected_classes", []) if c.get("class_id")]
    
    # Check if answer entities' classes are sensed
    answer_cids = set(inst_index[e]["class_id"] for e in answer_eids)
    sensed_set = set(sensed_cids)
    missed_classes = answer_cids - sensed_set
    hit_classes = answer_cids & sensed_set
    
    print(f"\n  [SEED SELECTION]")
    print(f"    TF-IDF sensed classes: {sensed_cids[:10]}")
    print(f"    Answer classes: {sorted(answer_cids)}")
    print(f"    Hit: {sorted(hit_classes)} | Missed: {sorted(missed_classes)}")
    
    # Show TF-IDF scores for answer classes
    all_class_ids = []
    all_class_texts = []
    for node in cg.graph.nodes:
        cid = str(getattr(node, "class_id", ""))
        if cid:
            theme = getattr(node, "theme", "") or ""
            tags = getattr(node, "tags", []) or []
            txt = f"{theme} {' '.join(tags)}".strip()
            all_class_ids.append(cid)
            all_class_texts.append(txt if txt else f"class_{cid}")
    
    if all_class_texts:
        try:
            sims, _, _ = calculate_tfidf_similarity(q, all_class_texts)
            class_scores = sorted(zip(all_class_ids, sims), key=lambda x: -x[1])
            
            for acls in sorted(answer_cids):
                rank = next((i for i, (c, s) in enumerate(class_scores) if c == acls), -1)
                score = next((s for c, s in class_scores if c == acls), 0.0)
                cname = inst_index.get(f"{acls}:instance_1", {}).get("class_name", "?")
                print(f"    Answer class {acls} ({cname[:30]}): rank={rank+1}/{len(class_scores)}, score={score:.4f}")
        except Exception as e:
            print(f"    (TF-IDF scoring failed: {e})")

    # 3. Check edge connectivity
    print(f"\n  [CONNECTIVITY]")
    for eid in sorted(answer_eids)[:4]:
        edges = edge_index.get(eid, [])
        if edges:
            connected_classes = set()
            for other, leg, _ in edges:
                oc = other.split(":")[0]
                connected_classes.add(oc)
            print(f"    {eid}: connected to {len(connected_classes)} classes via {len(edges)} edges")
            print(f"      Connected to sensed classes? {connected_classes & sensed_set}")
        else:
            print(f"    {eid}: NO EDGES (isolated!)")
