"""Search for specific terms in the conv5 graph to identify missing entities."""
import pickle, sys, json
sys.path.insert(0, 'mosaic')

with open('mosaic/src/locomo results/graph/pkl_con5/graph_network_conv_5_20260119_170445.pkl', 'rb') as f:
    g = pickle.load(f)

search_terms = [
    'board game', 'settlers', 'catan', 'ticket to ride', 'codenames', 'game night',
    'camping', 'fishing', 'volleyball', 'flower', 'garden', 'max', 'childhood',
    'scout', 'rock climbing', 'sushi', 'meditation',
]

for term in search_terms:
    hits = []
    for n in g.nodes:
        cid = getattr(n, 'class_id', '')
        cname = getattr(n, 'class_name', '')
        for inst in getattr(n, '_instances', []) or []:
            iid = inst.get('instance_id', '')
            iname = inst.get('instance_name', '')
            txt = json.dumps(inst).lower()
            if term.lower() in txt:
                hits.append(f'{cid}:{iid} ({iname}) in class {cname}')
    if hits:
        print(f'[{term}] {len(hits)} hits:')
        for h in hits[:3]:
            print(f'  {h}')
    else:
        print(f'[{term}] NO HITS')
