"""
Test script: verify Fix A (keyword stemming) and Fix B (uncovered message sweep).

Fix A: possessive stripping in _neighbor_bfs_ranked keyword scoring
Fix B: sweep_uncovered_messages creates hash instances for dropped messages
"""
import os, sys, json, pickle, copy

os.chdir('/Users/zachary/Workspace/LongtermMemory/mosaic')
sys.path.insert(0, '.')

# ─── Fix A: keyword stemming ───────────────────────────────────────
print("=" * 60)
print("FIX A: Keyword stemming / possessive stripping")
print("=" * 60)

import re
_POSS_RE = re.compile(r"['\u2019]s$")
_POSS_TEXT_RE = re.compile(r"['\u2019]s\b")

# Test: query keyword extraction with possessives
test_queries = [
    "What does Audrey's dog like?",
    "What is Andrew's favorite outdoor activity?",
    "Tell me about Audrey's childhood pet",
]

for q in test_queries:
    q_lower = q.lower()
    q_words = set()
    for w in q_lower.split():
        w = w.strip(".,!?()[]\"':;")
        w_clean = _POSS_RE.sub("", w)
        if len(w_clean) > 2:
            q_words.add(w_clean)
    _stop = {"the", "and", "for", "are", "was", "were", "has", "have", "had",
             "that", "this", "with", "from", "what", "which", "who", "how",
             "does", "did", "not", "been", "but", "they", "them", "than",
             "can", "her", "his", "she", "you", "all", "any", "some", "will"}
    q_words -= _stop
    print(f"  Query: {q}")
    print(f"  Keywords: {sorted(q_words)}")

    # Test matching with normalized text
    test_texts = ["audrey loves her dog pepper", "andrew enjoys rock climbing"]
    for txt in test_texts:
        txt_norm = _POSS_TEXT_RE.sub("", txt)
        hits = sum(1 for kw in q_words if kw in txt_norm)
        if hits > 0:
            print(f"    '{txt}' -> {hits} hits")
    print()

# Verify "audrey's" now becomes "audrey"
test_kw = "audrey's"
cleaned = _POSS_RE.sub("", test_kw.strip(".,!?()[]\"':;"))
assert cleaned == "audrey", f"Expected 'audrey' but got '{cleaned}'"
print("  [PASS] 'audrey\\'s' -> 'audrey' after possessive stripping")
print()

# ─── Fix B: uncovered message sweep ────────────────────────────────
print("=" * 60)
print("FIX B: Uncovered message sweep")
print("=" * 60)

from src.assist import conv_message_splitter
from src.data.graph import ClassGraph

# Load the existing conv5 graph
with open('src/locomo results/graph/pkl_con5/graph_network_conv_5_20260119_170445.pkl', 'rb') as f:
    g_nx = pickle.load(f)

# Create a ClassGraph and set its graph
memory = ClassGraph.__new__(ClassGraph)
memory.graph = g_nx
memory.edges = []
memory.tags = []
memory.selected_instance_keys = set()
memory._graph_save_dir = '/tmp/test_graph'
memory.G_p = __import__('networkx').DiGraph()
memory.G_a = __import__('networkx').Graph()
memory._bge_embedding_cache = None

# Load the edge records
with open('src/locomo results/graph/pkl_con5/graph_edge_conv_5_20260119_170445.json') as f:
    memory.edges = json.load(f)

# Load conv5 data and split into batches
with open('../dataset/locomo/locomo_conv5.json') as f:
    conv_data = json.load(f)

result = conv_message_splitter(conv_data)
print(f"  Total batches: {len(result)}")
total_msgs = sum(len(batch) for batch, _ in result)
print(f"  Total messages: {total_msgs}")

# Count covered labels before sweep
covered_labels = set()
for cn in memory.graph.nodes:
    for inst in getattr(cn, '_instances', []) or []:
        for lab in inst.get('message_labels', []):
            covered_labels.add(str(lab).strip())

print(f"  Labels covered by instances: {len(covered_labels)}")

# Count uncovered
all_labels = set()
uncovered_labels = set()
uncovered_with_fish = []
for batch, _ in result:
    for msg in batch:
        label = str(msg.get('label', '')).strip()
        all_labels.add(label)
        if label not in covered_labels:
            text = msg.get('message', '')
            if len(text.strip()) >= 30:
                uncovered_labels.add(label)
                if 'fish' in text.lower():
                    uncovered_with_fish.append((label, text[:100]))

print(f"  Total data labels: {len(all_labels)}")
print(f"  Uncovered labels (>= 30 chars): {len(uncovered_labels)}")
print(f"  Coverage rate: {len(covered_labels & all_labels)}/{len(all_labels)} "
      f"({100 * len(covered_labels & all_labels) / len(all_labels):.1f}%)")
print()

if uncovered_with_fish:
    print("  Fishing messages that would be caught by sweep:")
    for label, text in uncovered_with_fish:
        print(f"    L{label}: {text}")
    print()

# Run the actual sweep
print("  Running sweep_uncovered_messages...")
stats = memory.sweep_uncovered_messages(result)
print(f"  Sweep results: {stats}")

# Verify fishing is now in the graph
fish_found = False
for cn in memory.graph.nodes:
    cid = getattr(cn, 'class_id', '')
    for inst in getattr(cn, '_instances', []) or []:
        txt = json.dumps(inst).lower()
        if 'fish' in txt and 'fishing' in txt:
            iid = inst.get('instance_id', '')
            iname = inst.get('instance_name', '')
            cname = getattr(cn, 'class_name', '')
            print(f"  [FOUND] fishing in {cid}:{iid} ({iname}) class={cname}")
            fish_found = True

if fish_found:
    print("  [PASS] Fishing content is now in the graph after sweep")
else:
    print("  [FAIL] Fishing content still not found after sweep")

# Count new coverage
new_covered = set()
for cn in memory.graph.nodes:
    for inst in getattr(cn, '_instances', []) or []:
        for lab in inst.get('message_labels', []):
            new_covered.add(str(lab).strip())
print(f"\n  Coverage after sweep: {len(new_covered & all_labels)}/{len(all_labels)} "
      f"({100 * len(new_covered & all_labels) / len(all_labels):.1f}%)")

print("\nDone!")
