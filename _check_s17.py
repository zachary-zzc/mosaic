"""Check what happened to S17 messages in the graph."""
import pickle, sys, json
sys.path.insert(0, 'mosaic')

with open('mosaic/src/locomo results/graph/pkl_con5/graph_network_conv_5_20260119_170445.pkl', 'rb') as f:
    g = pickle.load(f)

# Let's find what message_labels correspond to S17
# First check what labels exist for the fishing conversation
# The conversation at S17 mentions fishing, let's grep all instances for label ranges

# Find message labels that contain 'fish' or 'lake' content from S17
# First, let's find what labels S17 would have
with open('dataset/locomo/locomo_conv5.json') as f:
    conv = json.load(f)

# Collect S17 dia_ids
s17 = conv.get('session_17', [])
print(f"Session 17 has {len(s17)} messages")
for msg in s17:
    print(f"  dia_id={msg.get('dia_id')}: {msg['text'][:100]}")

# Now find which instances have message_labels from S17
print("\n=== Instances with S17 message labels ===")
s17_labels = {msg.get('dia_id') for msg in s17}
print(f"S17 labels: {sorted(s17_labels)}")

for n in g.nodes:
    cid = getattr(n, 'class_id', '')
    cname = getattr(n, 'class_name', '')
    for inst in getattr(n, '_instances', []) or []:
        iid = inst.get('instance_id', '')
        labels = set(inst.get('message_labels', []))
        common = labels & s17_labels
        if common:
            iname = inst.get('instance_name', '')
            txt = json.dumps(inst).lower()
            has_fish = 'fish' in txt
            print(f"  {cid}:{iid} ({iname}) labels={sorted(common)} fish_mention={has_fish}")
            # Show the relevant text
            ufield = inst.get('uninstance_field', '')
            if ufield:
                print(f"    uninstance: {ufield[:200]}")
