"""Check message label format in graph and conversation."""
import pickle, sys, json, os
sys.path.insert(0, '.')
os.chdir('/Users/zachary/Workspace/LongtermMemory/mosaic')

with open('src/locomo results/graph/pkl_con5/graph_network_conv_5_20260119_170445.pkl', 'rb') as f:
    g = pickle.load(f)

# Collect all message_labels used in instances
all_labels = set()
for n in g.nodes:
    for inst in getattr(n, '_instances', []) or []:
        for lab in inst.get('message_labels', []):
            all_labels.add(lab)

label_types = set(type(l).__name__ for l in all_labels)
print(f"Total unique labels in graph: {len(all_labels)}")
print(f"Label types: {label_types}")
print(f"Sample labels: {sorted(all_labels)[:20]}")
print(f"Max label: {max(all_labels) if all_labels else 'N/A'}")

# Now check what the conv_message_splitter produces
# The splitter assigns integer labels
# Let's look at how the conversation is processed
from src.assist import conv_message_splitter
with open('../dataset/locomo/locomo_conv5.json') as f:
    conv = json.load(f)

result = conv_message_splitter(conv)
print(f"\nBatches from splitter: {len(result)}")

# Find the batch containing the fishing message
for i, (batch, context) in enumerate(result):
    for msg in batch:
        text = (msg.get('message', '') or '').lower()
        if 'fish' in text:
            print(f"Batch {i}: label={msg.get('label')}: {msg['message'][:200]}")
