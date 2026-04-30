import json, os

vector_dir = 'output/puzzle_run/3_vector'
corners = 0
edges = 0
edge_pieces = []
for f in os.listdir(vector_dir):
    if f.startswith('side_') and f.endswith('.json'):
        with open(os.path.join(vector_dir, f)) as fh:
            data = json.load(fh)
        if data.get('is_edge', False):
            edges += 1

pieces = {}
for f in os.listdir(vector_dir):
    if f.startswith('side_') and f.endswith('.json'):
        parts = f.replace('side_', '').replace('.json', '').split('_')
        pid = int(parts[0])
        if pid not in pieces:
            pieces[pid] = [False, False, False, False]
        si = int(parts[1])
        with open(os.path.join(vector_dir, f)) as fh:
            data = json.load(fh)
        pieces[pid][si] = data.get('is_edge', False)

for pid, flags in sorted(pieces.items()):
    ec = sum(flags)
    if ec >= 2:
        corners += 1
        edge_pieces.append(f"P{pid}(C,edges={flags})")
    elif ec >= 1:
        edge_pieces.append(f"P{pid}(E,edges={flags})")

print(f"Total pieces: {len(pieces)}")
print(f"Total side-edges: {edges}")
print(f"Corners (>=2 edge sides): {corners}")
print(f"Edge pieces (1 edge side): {sum(1 for p in pieces.values() if sum(p) == 1)}")
print(f"Inner pieces (0 edge sides): {sum(1 for p in pieces.values() if sum(p) == 0)}")
print(f"\nCorner pieces: {[p for p in edge_pieces if '(C,' in p]}")
print(f"\nAll edge pieces:")
for ep in edge_pieces:
    print(f"  {ep}")
