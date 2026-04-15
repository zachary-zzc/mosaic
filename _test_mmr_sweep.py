"""Sweep MMR lambda values to find optimal setting."""
import sys, os, json, pickle, importlib
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

BASE = "/Users/zachary/Workspace/LongtermMemory"

def load_conv(conv_id):
    pkl_path = os.path.join(BASE, f"experiments/locomo/benchmark/runs/runs/conv{conv_id}/artifacts/hash_only/graph_network_conv{conv_id}.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(BASE, f"experiments/locomo/benchmark/runs/runs/conv{conv_id}/artifacts/hybrid/graph_network_conv{conv_id}.pkl")
    with open(pkl_path, "rb") as f:
        cg = pickle.load(f)
    with open(os.path.join(BASE, f"dataset/locomo/qa_{conv_id}.json")) as f:
        qa = json.load(f)
    return cg, qa

def get_all_retrieved_text(cg, question):
    cg.selected_instance_keys.clear()
    cg._adj_cache = None
    sensed = cg._sense_classes_by_tfidf(question, 10, threshold=0.6, allow_below_threshold=True)
    _, inst_str1, _ = cg._fetch_instances_by_tfidf(question, 15, threshold=0.5, classes=sensed)
    _, inst_str2, _ = cg._fetch_instances_by_tfidf(question, 15, threshold=0.1)
    seeds = set(cg.selected_instance_keys)
    expanded_keys = cg._neighbor_expansion_key_list(seeds, query=question)
    neighbor_str = cg._query_neighbor_context_string(seeds, _precomputed_keys=expanded_keys)
    combined = "\n".join(s for s in (inst_str1, inst_str2, neighbor_str) if (s or "").strip())
    return combined.lower()

def check_answer_in_context(answer_text, context):
    answer_lower = answer_text.lower()
    words = [w.strip(".,!?()[]\"'") for w in answer_lower.split()]
    sig_words = [w for w in words if len(w) > 3 and w not in {"that", "this", "with", "from", "they", "their", "them", "have", "been", "were", "would", "could", "should", "about", "also", "does", "more", "than"}]
    if not sig_words:
        return True
    found = sum(1 for w in sig_words if w in context)
    return found >= len(sig_words) * 0.5

import src.config_loader as cl

# Pre-load graphs
graphs = {}
for conv_id in [3, 4, 5]:
    graphs[conv_id] = load_conv(conv_id)

print(f"{'lambda':>8} | {'conv3':>8} | {'conv4':>8} | {'conv5':>8} | {'total':>8}")
print("-" * 55)

for lam_val in [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
    os.environ["MOSAIC_QUERY_NEIGHBOR_MMR_LAMBDA"] = str(lam_val)
    importlib.reload(cl)

    total_correct = 0
    total_mh = 0
    conv_results = {}

    for conv_id in [3, 4, 5]:
        cg, qa = graphs[conv_id]
        multihop = [q for q in qa if q.get("category") == 3]
        correct = sum(1 for q in multihop if check_answer_in_context(q["answer"], get_all_retrieved_text(cg, q["question"])))
        conv_results[conv_id] = (correct, len(multihop))
        total_correct += correct
        total_mh += len(multihop)

    c3, t3 = conv_results[3]
    c4, t4 = conv_results[4]
    c5, t5 = conv_results[5]
    print(f"{lam_val:>8.1f} | {c3}/{t3} ({100*c3/t3:4.1f}%) | {c4}/{t4} ({100*c4/t4:4.1f}%) | {c5}/{t5} ({100*c5/t5:4.1f}%) | {total_correct}/{total_mh} ({100*total_correct/total_mh:4.1f}%)")
