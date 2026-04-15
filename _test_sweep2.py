"""Focused param sweep - fewer configs, faster."""
import sys, os, json, pickle, importlib
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

BASE = "/Users/zachary/Workspace/LongtermMemory"

from src.assist import build_instance_fragments, calculate_tfidf_similarity

def load_conv(conv_id):
    for mode in ["hash_only", "hybrid"]:
        pkl_path = os.path.join(BASE, f"experiments/locomo/benchmark/runs/runs/conv{conv_id}/artifacts/{mode}/graph_network_conv{conv_id}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                return pickle.load(f)

def check_answer_in_context(answer_text, context):
    answer_lower = answer_text.lower()
    words = [w.strip(".,!?()[]\"'") for w in answer_lower.split()]
    sig_words = [w for w in words if len(w) > 3 and w not in {"that", "this", "with", "from", "they", "their", "them", "have", "been", "were", "would", "could", "should", "about", "also", "does", "more", "than"}]
    if not sig_words:
        return True
    found = sum(1 for w in sig_words if w in context)
    return found >= len(sig_words) * 0.5

print("Loading graphs...", flush=True)
graphs = {cid: load_conv(cid) for cid in [3, 4, 5]}
qas = {}
for cid in [3, 4, 5]:
    with open(os.path.join(BASE, f"dataset/locomo/qa_{cid}.json")) as f:
        qas[cid] = [q for q in json.load(f) if q.get("category") == 3]
print("Done.\n", flush=True)

print(f"{'config':>30} | {'conv3':>10} | {'conv4':>10} | {'conv5':>10} | {'total':>10}", flush=True)
print("-" * 85, flush=True)

configs = [
    (1, 16, 1.0, "baseline(h1,n16,lam1.0)"),
    (1, 16, 0.5, "mmr(h1,n16,lam0.5)"),
    (1, 32, 0.5, "mmr(h1,n32,lam0.5)"),
    (1, 48, 0.5, "mmr(h1,n48,lam0.5)"),
    (2, 32, 0.5, "mmr(h2,n32,lam0.5)"),
]

for hops, max_extra, mmr_lambda, label in configs:
    os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = str(hops)
    os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = str(max_extra)
    os.environ["MOSAIC_QUERY_NEIGHBOR_MMR_LAMBDA"] = str(mmr_lambda)
    import src.config_loader as cl
    importlib.reload(cl)
    
    tc, tm = 0, 0
    cr = {}
    for cid in [3, 4, 5]:
        cg = graphs[cid]
        mh = qas[cid]
        c = 0
        for q in mh:
            cg.selected_instance_keys.clear()
            cg._adj_cache = None
            sensed = cg._sense_classes_by_tfidf(q["question"], 10, threshold=0.6, allow_below_threshold=True)
            _, s1, _ = cg._fetch_instances_by_tfidf(q["question"], 15, threshold=0.5, classes=sensed)
            _, s2, _ = cg._fetch_instances_by_tfidf(q["question"], 15, threshold=0.1)
            seeds = set(cg.selected_instance_keys)
            ek = cg._neighbor_expansion_key_list(seeds, query=q["question"])
            ns = cg._query_neighbor_context_string(seeds, _precomputed_keys=ek)
            ctx = "\n".join(s for s in (s1, s2, ns) if (s or "").strip())
            if check_answer_in_context(q["answer"], ctx.lower()):
                c += 1
        cr[cid] = (c, len(mh))
        tc += c
        tm += len(mh)
    
    c3, t3 = cr[3]; c4, t4 = cr[4]; c5, t5 = cr[5]
    print(f"{label:>30} | {c3}/{t3} ({100*c3/t3:4.1f}%) | {c4}/{t4} ({100*c4/t4:4.1f}%) | {c5}/{t5} ({100*c5/t5:4.1f}%) | {tc}/{tm} ({100*tc/tm:4.1f}%)", flush=True)

print("\nDone!", flush=True)
