import sys, os, json, math
sys.path.insert(0, 'src')

conn_path = 'output/puzzle_run/5_connectivity/connectivity.json'
with open(conn_path) as f:
    connectivity = json.load(f)

vector_dir = 'output/puzzle_run/3_vector'

# Load side data to determine is_edge for each piece
pieces_info = {}
for f in sorted(os.listdir(vector_dir)):
    if f.startswith('side_') and f.endswith('.json'):
        parts = f.replace('.json', '').split('_')
        pid = int(parts[1])
        side_idx = int(parts[2])
        with open(os.path.join(vector_dir, f)) as fh:
            data = json.load(fh)
        if pid not in pieces_info:
            pieces_info[pid] = {}
        pieces_info[pid][side_idx] = data['is_edge']

# For each piece, count edges and non-matching sides
corners = []
edges = []
interiors = []

for pid_str in connectivity:
    pid = int(pid_str)
    sides = connectivity[pid_str]
    side_edges = {}
    for si in range(4):
        if si in pieces_info.get(pid, {}):
            side_edges[si] = pieces_info[pid][si]
    
    edge_count = sum(1 for v in side_edges.values() if v)
    
    # Count sides with NO matches
    no_match_count = 0
    for si in range(len(sides)):
        matches = sides[si]
        if len(matches) == 0:
            no_match_count += 1
    
    if edge_count == 2:
        corners.append((pid, edge_count, no_match_count, side_edges))
    elif edge_count == 1:
        edges.append((pid, edge_count, no_match_count, side_edges))
    else:
        interiors.append((pid, edge_count, no_match_count, side_edges))

print(f"Total pieces: {len(connectivity)}")
print(f"Corners (2 edges): {len(corners)}")
for pid, ec, nm, se in corners:
    print(f"  Piece {pid}: is_edge={se}, no_match_sides={nm}")
print(f"\nEdges (1 edge): {len(edges)}")
print(f"Interiors (0 edges): {len(interiors)}")

# The issue: 6 corners instead of 4
# 2 of these might have edges incorrectly detected
# Let's check: a true corner should have is_edge=True on 2 adjacent sides
print("\n--- Checking if corner pieces have adjacent edge sides ---")
for pid, ec, nm, se in corners:
    edge_sides = [si for si, is_e in se.items() if is_e]
    adjacent = False
    for i in range(len(edge_sides)):
        for j in range(i+1, len(edge_sides)):
            if abs(edge_sides[i] - edge_sides[j]) == 1 or abs(edge_sides[i] - edge_sides[j]) == 3:
                adjacent = True
    print(f"  Piece {pid}: edge_sides={edge_sides}, adjacent={adjacent}, no_match={nm}")
