"""Check: for each 'unfixable' query, what conv fragments contain the answer,
and are those fragments captured in graph entities?"""
import os, sys, json, pickle
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

from src.assist import build_instance_fragments

BASE = "/Users/zachary/Workspace/LongtermMemory"

# Load conversation
with open(f"{BASE}/dataset/locomo/locomo_conv5.json") as f:
    conv = json.load(f)

# Load graph
with open(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/graph_network_conv5.pkl", "rb") as f:
    cg = pickle.load(f)
cg.process_kw(f"{BASE}/experiments/locomo/benchmark/runs/runs/conv5/artifacts/hybrid/conv5_tags.json")

# Build instance text index
inst_text = {}
for node in cg.graph.nodes:
    cid = getattr(node, "class_id", "")
    for inst in getattr(node, "_instances", []) or []:
        iid = inst.get("instance_id", "")
        key = f"{cid}_instance_{iid}" if not iid.startswith("instance_") else f"{cid}_{iid}"
        frags = build_instance_fragments(inst)
        text = "\n".join(t for _ty, t in frags if t and t.strip())
        inst_text[f"{cid}:{iid}"] = text

queries = {
    "Q61": {
        "question": "How many dogs does Andrew have?",
        "gold": "3",
        "search_terms": ["toby", "adopted", "pup", "andrew", "dog"],
        "notes": "Andrew adopted Toby (session 5), then Cooper (session 24), and Max (session 26?). Need all adoption mentions."
    },
    "Q18": {
        "question": "How many times did Audrey and Andrew plan to hike together?",
        "gold": "three times",
        "search_terms": ["hike", "hiking", "trail", "plan"],
        "notes": "Need each distinct hiking plan mention"
    },
    "Q25": {
        "question": "Did Audrey and Andrew grow up with a pet dog?",
        "gold": "Yes",
        "search_terms": ["grew up", "grow up", "childhood", "kid", "young", "family dog", "family pet"],
        "notes": "Need childhood pet references"
    },
    "Q13": {
        "question": "What outdoor activities has Andrew done other than hiking in nature?",
        "gold": "rock climbing, fishing, camping",
        "search_terms": ["rock climb", "fishing", "camping", "outdoor"],
        "notes": "Need each activity mention"
    },
    "Q3": {
        "question": "What kind of indoor activities has Andrew pursued with his girlfriend?",
        "gold": "boardgames, volunteering at pet shelter, wine tasting, growing flowers",
        "search_terms": ["boardgame", "board game", "wine tast", "volunteer", "flower", "indoor", "girlfriend"],
        "notes": "Need each indoor activity"
    },
    "Q19": {
        "question": "Where did Audrey get Pixie from?",
        "gold": "breeder",
        "search_terms": ["pixie", "breeder", "got", "adopted", "puppy"],
        "notes": "Need Pixie acquisition mention"
    },
}

for qid, info in queries.items():
    print(f"\n{'='*80}")
    print(f"{qid}: {info['question']}")
    print(f"Gold: {info['gold']}")
    
    # 1. Find conversation fragments
    print(f"\n  --- Conversation fragments ---")
    conv_hits = []
    for i in range(1, 29):
        session = conv.get(f"session_{i}", [])
        for j, msg in enumerate(session):
            text = msg.get("text", "")
            text_lower = text.lower()
            matched = [t for t in info["search_terms"] if t in text_lower]
            if matched:
                conv_hits.append((i, j, text[:200], matched))
    
    for sess, idx, text, matched in conv_hits:
        print(f"    S{sess:02d}.{idx:02d} [{','.join(matched)}]: {text}")
    print(f"  Total conv fragments: {len(conv_hits)}")
    
    # 2. Find graph entities containing these terms
    print(f"\n  --- Graph entities with answer content ---")
    graph_hits = []
    for eid, text in inst_text.items():
        text_lower = text.lower()
        matched = [t for t in info["search_terms"] if t in text_lower]
        if len(matched) >= 2:  # At least 2 search terms
            graph_hits.append((eid, matched, text[:150]))
    
    for eid, matched, text in graph_hits[:8]:
        print(f"    {eid} [{','.join(matched)}]: {text[:120]}")
    print(f"  Total graph entities (>=2 terms): {len(graph_hits)}")
    
    # 3. Check if the specific answer info is in ANY entity
    print(f"\n  --- Answer-specific check ---")
    gold_lower = info["gold"].lower()
    gold_words = [w for w in gold_lower.replace(",", " ").split() if len(w) > 2]
    exact_hits = []
    for eid, text in inst_text.items():
        tl = text.lower()
        found = [w for w in gold_words if w in tl]
        if len(found) >= max(1, len(gold_words) * 0.5):
            exact_hits.append((eid, found))
    
    if exact_hits:
        for eid, found in exact_hits[:5]:
            print(f"    {eid}: contains {found}")
    else:
        print(f"    NO entity contains answer words: {gold_words}")
