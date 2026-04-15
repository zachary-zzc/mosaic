"""
Full evaluation: compare answer retrieval rate between pure TF-IDF (λ=1.0)
and MMR-diversified (λ=0.5) neighbor expansion across conv3, conv4, conv5.
"""
import sys, os, json, pickle, importlib
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

BASE = "/Users/zachary/Workspace/LongtermMemory"

def load_conv(conv_id):
    pkl_path = os.path.join(BASE, f"experiments/locomo/benchmark/runs/runs/conv{conv_id}/artifacts/hash_only/graph_network_conv{conv_id}.pkl")
    if not os.path.exists(pkl_path):
        # Try hybrid
        pkl_path = os.path.join(BASE, f"experiments/locomo/benchmark/runs/runs/conv{conv_id}/artifacts/hybrid/graph_network_conv{conv_id}.pkl")
    with open(pkl_path, "rb") as f:
        cg = pickle.load(f)
    with open(os.path.join(BASE, f"dataset/locomo/qa_{conv_id}.json")) as f:
        qa = json.load(f)
    return cg, qa


def get_all_retrieved_text(cg, question):
    """Run full retrieval and return combined text (seeds + neighbors)."""
    cg.selected_instance_keys.clear()
    cg._adj_cache = None
    sensed = cg._sense_classes_by_tfidf(question, 10, threshold=0.6, allow_below_threshold=True)
    _, inst_str1, _ = cg._fetch_instances_by_tfidf(question, 15, threshold=0.5, classes=sensed)
    _, inst_str2, _ = cg._fetch_instances_by_tfidf(question, 15, threshold=0.1)
    seeds = set(cg.selected_instance_keys)
    expanded_keys = cg._neighbor_expansion_key_list(seeds, query=question)
    neighbor_str = cg._query_neighbor_context_string(seeds, _precomputed_keys=expanded_keys)
    combined = "\n".join(s for s in (inst_str1, inst_str2, neighbor_str) if (s or "").strip())
    return combined.lower(), len(seeds), len(expanded_keys)


def check_answer_in_context(answer_text, context):
    """Simple keyword check: are key answer words in the context?"""
    answer_lower = answer_text.lower()
    # Extract significant words (>3 chars)
    words = [w.strip(".,!?()[]\"'") for w in answer_lower.split()]
    sig_words = [w for w in words if len(w) > 3 and w not in {"that", "this", "with", "from", "they", "their", "them", "have", "been", "were", "would", "could", "should", "about", "also", "does", "more", "than"}]
    if not sig_words:
        return True  # trivial answer
    found = sum(1 for w in sig_words if w in context)
    return found >= len(sig_words) * 0.5  # at least 50% of significant words found


import src.config_loader as cl

results = {}
for lam_val, label in [(1.0, "OLD"), (0.5, "MMR")]:
    os.environ["MOSAIC_QUERY_NEIGHBOR_MMR_LAMBDA"] = str(lam_val)
    importlib.reload(cl)
    print(f"\n{'='*60}")
    print(f"  Strategy: {label} (lambda={cl.get_query_neighbor_mmr_lambda()})")
    print(f"{'='*60}")

    total_correct = 0
    total_mh = 0

    for conv_id in [3, 4, 5]:
        cg, qa = load_conv(conv_id)
        multihop = [q for q in qa if q.get("category") == 3]
        correct = 0
        for q_item in multihop:
            question = q_item["question"]
            answer = q_item["answer"]
            context, n_seeds, n_expand = get_all_retrieved_text(cg, question)
            if check_answer_in_context(answer, context):
                correct += 1
        print(f"  conv{conv_id}: {correct}/{len(multihop)} multi-hop retrieved ({100*correct/len(multihop):.1f}%)")
        total_correct += correct
        total_mh += len(multihop)

    results[label] = (total_correct, total_mh)
    print(f"  TOTAL: {total_correct}/{total_mh} ({100*total_correct/total_mh:.1f}%)")

print(f"\n{'='*60}")
print("COMPARISON:")
for label, (c, t) in results.items():
    print(f"  {label}: {c}/{t} ({100*c/t:.1f}%)")
print(f"{'='*60}")
