import sys, os, json
sys.path.insert(0, 'src')

conn_path = 'output/puzzle_run/5_connectivity/connectivity.json'
with open(conn_path) as f:
    connectivity = json.load(f)

# Count no-match sides for each piece (this is how connect.py determines corners/edges)
no_match_counts = {}
for pid_str, sides in connectivity.items():
    pid = int(pid_str)
    nm = sum(1 for s in sides if len(s) == 0)
    no_match_counts[pid] = nm

# Corners = 2 no-match sides, edges = 1 no-match side
corners_2nm = sorted([pid for pid, nm in no_match_counts.items() if nm == 2])
edges_1nm = sorted([pid for pid, nm in no_match_counts.items() if nm == 1])
interiors_0nm = sorted([pid for pid, nm in no_match_counts.items() if nm == 0])
high_nm = sorted([pid for pid, nm in no_match_counts.items() if nm > 2])

print(f"Total pieces: {len(connectivity)}")
print(f"2 no-match sides (corners): {corners_2nm} ({len(corners_2nm)})")
print(f"1 no-match side (edges): {edges_1nm} ({len(edges_1nm)})")
print(f"0 no-match sides (interiors): {interiors_0nm} ({len(interiors_0nm)})")
print(f">2 no-match sides: {high_nm} ({len(high_nm)})")

# Also check is_edge from side JSON
vector_dir = 'output/puzzle_run/3_vector'
print("\n--- is_edge from side JSON ---")
for pid in corners_2nm + edges_1nm[:5]:
    edge_info = {}
    for si in range(4):
        fp = os.path.join(vector_dir, f'side_{pid}_{si}.json')
        if os.path.exists(fp):
            with open(fp) as f:
                d = json.load(f)
            edge_info[si] = d['is_edge']
    print(f"  Piece {pid}: is_edge={edge_info}")

# Show which sides have no matches for corners
print("\n--- No-match sides for corners ---")
for pid in corners_2nm:
    sides = connectivity[str(pid)]
    nm_sides = [i for i, s in enumerate(sides) if len(s) == 0]
    print(f"  Piece {pid}: no-match sides = {nm_sides}")
