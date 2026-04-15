"""Quick test: compare old (pure TF-IDF) vs new (MMR) neighbor expansion on conv5 multi-hop questions."""
import sys, os, json, pickle, importlib
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

# Load graph
BASE = "/Users/zachary/Workspace/LongtermMemory"
graph_path = os.path.join(BASE, "experiments/locomo/benchmark/runs/runs/conv5/artifacts/hash_only/graph_network_conv5.pkl")
with open(graph_path, "rb") as f:
    cg = pickle.load(f)

# Load QA
with open(os.path.join(BASE, "dataset/locomo/qa_5.json")) as f:
    qa = json.load(f)

multihop = [q for q in qa if q.get("category") == 3]
print(f"Conv5 multi-hop questions: {len(multihop)}")

from src.config_loader import get_query_neighbor_traversal_config
max_hops, max_extra, legs = get_query_neighbor_traversal_config()
print(f"Config: hops={max_hops}, max_extra={max_extra}, legs={legs}")

for lam_val, label in [(1.0, "OLD (pure TF-IDF)"), (0.5, "NEW (MMR lambda=0.5)")]:
    os.environ["MOSAIC_QUERY_NEIGHBOR_MMR_LAMBDA"] = str(lam_val)
    # Reload config to pick up new env var
    import src.config_loader as cl
    importlib.reload(cl)

    print(f"\n{'='*60}")
    print(f"  {label} (lambda={cl.get_query_neighbor_mmr_lambda()})")
    print(f"{'='*60}")

    for q_item in multihop[:4]:
        question = q_item["question"]
        answer = q_item["answer"]

        cg.selected_instance_keys.clear()
        cg._adj_cache = None
        sensed = cg._sense_classes_by_tfidf(question, 10, threshold=0.6, allow_below_threshold=True)
        cg._fetch_instances_by_tfidf(question, 15, threshold=0.5, classes=sensed)
        cg._fetch_instances_by_tfidf(question, 15, threshold=0.1)
        seeds = set(cg.selected_instance_keys)

        expanded_keys = cg._neighbor_expansion_key_list(seeds, query=question)
        expanded_eids = [cg._instance_key_to_entity_id(k) for k in expanded_keys]

        class_dist = {}
        for eid in expanded_eids:
            cls = eid.split(":")[0] if ":" in eid else "?"
            class_dist[cls] = class_dist.get(cls, 0) + 1

        print(f"\n  Q: {question[:90]}")
        print(f"  A: {answer[:70]}")
        print(f"  Seeds={len(seeds)}, Expanded={len(expanded_keys)}")
        print(f"  Expanded class distribution: {class_dist}")

print("\nDone!")
