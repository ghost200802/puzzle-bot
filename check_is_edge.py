import os, json

d = 'output/puzzle_run/3_vector'
files = sorted([f for f in os.listdir(d) if f.startswith('side_') and f.endswith('.json')])

corners = {}
for f in files:
    parts = f.replace('.json','').split('_')
    pid = int(parts[1])
    si = int(parts[2])
    with open(os.path.join(d, f)) as fh:
        data = json.load(fh)
    is_edge = data.get('is_edge', False)
    if pid not in corners:
        corners[pid] = [False, False, False, False]
    corners[pid][si] = is_edge

c2 = sorted([p for p,flags in corners.items() if sum(flags)==2])
c1 = sorted([p for p,flags in corners.items() if sum(flags)==1])
c0 = sorted([p for p,flags in corners.items() if sum(flags)==0])

print(f"2 edges: {c2} ({len(c2)})")
print(f"1 edge: {len(c1)}")
print(f"0 edges: {len(c0)}")
