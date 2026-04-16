"""Search graph entities and edges for missing terms."""
import pickle, sys, json
sys.path.insert(0, 'mosaic')

with open('mosaic/src/locomo results/graph/pkl_con5/graph_network_conv_5_20260119_170445.pkl', 'rb') as f:
    g = pickle.load(f)

terms_to_search = ['wine tast', 'fishing', 'lake', 'fish']
for term in terms_to_search:
    hits = []
    for n in g.nodes:
        cid = getattr(n, 'class_id', '')
        cname = getattr(n, 'class_name', '')
        for inst in getattr(n, '_instances', []) or []:
            iid = inst.get('instance_id', '')
            txt = json.dumps(inst).lower()
            if term in txt:
                hits.append(f'{cid}:{iid} in {cname}: {inst.get("instance_name","?")}')
    print(f'[{term}] {len(hits)} hits')
    for h in hits[:5]:
        print(f'  {h}')
    print()

# Check edges for fishing content
print("=== Checking edges for fishing/lake content ===")
with open('mosaic/src/locomo results/graph/pkl_con5/graph_edge_conv_5_20260119_170445.json', 'r') as f:
    edges = json.load(f)

for edge in edges:
    content = (edge.get('content', '') or '').lower()
    if 'fish' in content or 'lake' in content:
        print(f"Edge label={edge.get('label')}: {content[:200]}")
        conns = edge.get('connections', [])
        print(f"  Connections: {conns[:3]}")
        print()
