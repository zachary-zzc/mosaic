"""Single-config test. Usage: python _test_single.py <hops> <max_extra> <mmr_lambda> [--suffix _v2]"""
import sys, os, json, pickle, importlib
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

BASE = "/Users/zachary/Workspace/LongtermMemory"
hops = int(sys.argv[1])
max_extra = int(sys.argv[2])
mmr_lambda = float(sys.argv[3])
suffix = ""
if "--suffix" in sys.argv:
    idx = sys.argv.index("--suffix")
    suffix = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""

os.environ["MOSAIC_QUERY_NEIGHBOR_HOPS"] = str(hops)
os.environ["MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA"] = str(max_extra)
os.environ["MOSAIC_QUERY_NEIGHBOR_MMR_LAMBDA"] = str(mmr_lambda)

from src.assist import calculate_tfidf_similarity

def check_answer(answer_text, context):
    words = [w.strip(".,!?()[]\"'") for w in answer_text.lower().split()]
    sig = [w for w in words if len(w) > 3 and w not in {"that","this","with","from","they","their","them","have","been","were","would","could","should","about","also","does","more","than"}]
    if not sig: return True
    return sum(1 for w in sig if w in context) >= len(sig) * 0.5

results = []
for cid in [3, 4, 5]:
    for mode in ["hash_only", "hybrid"]:
        p = os.path.join(BASE, f"experiments/locomo/benchmark/runs/runs/conv{cid}/artifacts/{mode}/graph_network_conv{cid}{suffix}.pkl")
        if not os.path.exists(p):
            p = os.path.join(BASE, f"experiments/locomo/benchmark/runs/runs/conv{cid}/artifacts/{mode}/graph_network_conv{cid}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f: cg = pickle.load(f)
            break
    with open(os.path.join(BASE, f"dataset/locomo/qa_{cid}.json")) as f:
        mh = [q for q in json.load(f) if q.get("category") == 3]
    c = 0
    for q in mh:
        cg.selected_instance_keys.clear(); cg._adj_cache = None
        sensed = cg._sense_classes_by_tfidf(q["question"], 10, threshold=0.6, allow_below_threshold=True)
        _, s1, _ = cg._fetch_instances_by_tfidf(q["question"], 15, threshold=0.5, classes=sensed)
        _, s2, _ = cg._fetch_instances_by_tfidf(q["question"], 15, threshold=0.1)
        seeds = set(cg.selected_instance_keys)
        ek = cg._neighbor_expansion_key_list(seeds, query=q["question"])
        ns = cg._query_neighbor_context_string(seeds, _precomputed_keys=ek)
        ctx = "\n".join(s for s in (s1, s2, ns) if (s or "").strip())
        if check_answer(q["answer"], ctx.lower()): c += 1
    results.append((cid, c, len(mh)))

tc = sum(c for _,c,_ in results)
tm = sum(t for _,_,t in results)
parts = " | ".join(f"c{cid}={c}/{t}({100*c/t:.0f}%)" for cid,c,t in results)
tag = f"{suffix}" if suffix else ""
print(f"h={hops} n={max_extra} lam={mmr_lambda}{tag} | {parts} | total={tc}/{tm}({100*tc/tm:.1f}%)")
