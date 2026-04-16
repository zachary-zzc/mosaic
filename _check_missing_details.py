"""Examine specific instances and look at conv5 conversation for missing terms."""
import pickle, sys, json
sys.path.insert(0, 'mosaic')

with open('mosaic/src/locomo results/graph/pkl_con5/graph_network_conv_5_20260119_170445.pkl', 'rb') as f:
    g = pickle.load(f)

# Show full content of board game instance
for n in g.nodes:
    cid = getattr(n, 'class_id', '')
    for inst in getattr(n, '_instances', []) or []:
        iid = inst.get('instance_id', '')
        if cid == 'class_1' and iid == 'instance_55':
            print("=== class_1:instance_55 (Board Game Night) ===")
            print(json.dumps(inst, indent=2, default=str)[:2000])
            print()

# Also check instance_23 (camping)
for n in g.nodes:
    cid = getattr(n, 'class_id', '')
    for inst in getattr(n, '_instances', []) or []:
        iid = inst.get('instance_id', '')
        if cid == 'class_1' and iid == 'instance_23':
            print("=== class_1:instance_23 (Camping Plan) ===")
            print(json.dumps(inst, indent=2, default=str)[:2000])
            print()

# Now check the conv5 conversation for the missing terms
print("\n=== Searching conv5 conversation for missing terms ===")
with open('dataset/locomo/locomo_conv5.json', 'r') as f:
    conv = json.load(f)

for term in ['settlers of catan', 'ticket to ride', 'codenames', 'fishing', 'volleyball', 'meditation']:
    for session in conv:
        for msg in session.get('conversation', []):
            text = msg.get('text', '').lower()
            if term in text:
                sid = session.get('session_id', '?')
                speaker = msg.get('speaker', '?')
                print(f"[{term}] S{sid}: {speaker}: {msg['text'][:200]}")
