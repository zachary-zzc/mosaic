"""
Test multiple parameter configurations on the 2 fixable failures:
- conv3 Q7: "How many hikes has Joanna been on?" → class_26:instance_1 (d=1, 76 candidates)
- conv4 Q8: "Which US states might Tim be in..." → class_3:instance_1 (d=1, 88 candidates)

Also check: does increasing max_extra and hops help OVERALL?
"""
import sys, os, json, pickle, importlib
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

BASE = "/Users/zachary/Workspace/LongtermMemory"

from src.assist import build_instance_fragments, calculate_tfidf_similarity
from src.data.dual_graph import ALL_EDGE_LEGS
from collections import deque

def load_conv(conv_id):
    for mode in ["hash_only", "hybrid"]:
        pkl_path = os.path.join(BASE, f"experiments/locomo/benchmark/runs/runs/conv{conv_id}/artifacts/{mode}/graph_network_conv{conv_id}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                cg = pickle.load(f)
            return cg
    raise FileNotFoundError(f"No graph for conv{conv_id}")

def load_qa(conv_id):
    with open(os.path.join(BASE, f"dataset/locomo/qa_{conv_id}.json")) as f:
        return json.load(f)

def check_answer_in_context(answer_text, context):
    answer_lower = answer_text.lower()
    words = [w.strip(".,!?()[]\"'") for w in answer_lower.split()]
    sig_words = [w for w in words if len(w) > 3 and w not in {"that", "this", "with", "from", "they", "their", "them", "have", "been", "were", "would", "could", "should", "about", "also", "does", "more", "than"}]
    if not sig_words:
        return True
    found = sum(1 for w in sig_words if w in context)
    return found >= len(sig_words) * 0.5

def eval_config(convs, hops, max_extra, mmr_lambda):
    os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = str(hops)
    os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = str(max_extra)
    os.environ["MOSAIC_QUERY_NEIGHBOR_MMR_LAMBDA"] = str(mmr_lambda)
    import src.config_loader as cl
    importlib.reload(cl)
    
    total_correct = 0
    total_mh = 0
    conv_results = {}
    
    for conv_id in convs:
        cg = graphs[conv_id]
        qa = qas[conv_id]
        multihop = [q for q in qa if q.get("category") == 3]
        correct = 0
        for q in multihop:
            cg.selected_instance_keys.clear()
            cg._adj_cache = None
            sensed = cg._sense_classes_by_tfidf(q["question"], 10, threshold=0.6, allow_below_threshold=True)
            _, inst1, _ = cg._fetch_instances_by_tfidf(q["question"], 15, threshold=0.5, classes=sensed)
            _, inst2, _ = cg._fetch_instances_by_tfidf(q["question"], 15, threshold=0.1)
            seeds = set(cg.selected_instance_keys)
            expanded_keys = cg._neighbor_expansion_key_list(seeds, query=q["question"])
            neighbor_str = cg._query_neighbor_context_string(seeds, _precomputed_keys=expanded_keys)
            combined = "\n".join(s for s in (inst1, inst2, neighbor_str) if (s or "").strip())
            if check_answer_in_context(q["answer"], combined.lower()):
                correct += 1
        conv_results[conv_id] = (correct, len(multihop))
        total_correct += correct
        total_mh += len(multihop)
    
    return conv_results, total_correct, total_mh

# Pre-load
print("Loading graphs...")
graphs = {}
qas = {}
for cid in [3, 4, 5]:
    graphs[cid] = load_conv(cid)
    qas[cid] = load_qa(cid)
print("Done.\n")

test_convs = [3, 4, 5]

print(f"{'config':>35} | {'conv3':>10} | {'conv4':>10} | {'conv5':>10} | {'total':>10}")
print("-" * 90)

configs = [
    # (hops, max_extra, mmr_lambda, label)
    (1, 16, 1.0, "baseline(h1,n16,lam1.0)"),
    (1, 16, 0.5, "mmr-only(h1,n16,lam0.5)"),
    (1, 32, 0.5, "h1,n32,lam0.5"),
    (1, 48, 0.5, "h1,n48,lam0.5"),
    (1, 64, 0.5, "h1,n64,lam0.5"),
    (2, 16, 0.5, "h2,n16,lam0.5"),
    (2, 32, 0.5, "h2,n32,lam0.5"),
    (2, 48, 0.5, "h2,n48,lam0.5"),
    (1, 32, 1.0, "h1,n32,lam1.0"),
    (1, 48, 1.0, "h1,n48,lam1.0"),
    (1, 64, 1.0, "h1,n64,lam1.0"),
    (2, 32, 1.0, "h2,n32,lam1.0"),
]

for hops, max_extra, mmr_lambda, label in configs:
    conv_results, tc, tm = eval_config(test_convs, hops, max_extra, mmr_lambda)
    c3, t3 = conv_results[3]
    c4, t4 = conv_results[4]
    c5, t5 = conv_results[5]
    print(f"{label:>35} | {c3}/{t3} ({100*c3/t3:4.1f}%) | {c4}/{t4} ({100*c4/t4:4.1f}%) | {c5}/{t5} ({100*c5/t5:4.1f}%) | {tc}/{tm} ({100*tc/tm:4.1f}%)")
