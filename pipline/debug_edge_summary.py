import json

with open('../output/puzzle_new/5_connectivity/connectivity_summary.json') as f:
    summary = json.load(f)

edges_with_incomplete = []
corners_with_incomplete = []

for pid_str, info in summary.items():
    pid = int(pid_str)
    if info['is_edge']:
        expected = 3
        if info['sides_with_matches'] < expected:
            edges_with_incomplete.append((pid, info['sides_with_matches'], info['total_matches']))
    elif info['is_corner']:
        expected = 2
        if info['sides_with_matches'] < expected:
            corners_with_incomplete.append((pid, info['sides_with_matches'], info['total_matches']))

print(f"Edge pieces with incomplete matches ({len(edges_with_incomplete)}):")
for pid, sides, total in edges_with_incomplete:
    print(f"  Piece {pid}: {sides}/3 sides with matches, total={total}")

print(f"\nCorner pieces with incomplete matches ({len(corners_with_incomplete)}):")
for pid, sides, total in corners_with_incomplete:
    print(f"  Piece {pid}: {sides}/2 sides with matches, total={total}")

edge_count = sum(1 for v in summary.values() if v['is_edge'])
corner_count = sum(1 for v in summary.values() if v['is_corner'])
inner_count = sum(1 for v in summary.values() if v['is_inner'])
print(f"\nTotal: {edge_count} edges, {corner_count} corners, {inner_count} inner")
print(f"Edges with all 3 sides: {edge_count - len(edges_with_incomplete)}/{edge_count}")
print(f"Corners with all 2 sides: {corner_count - len(corners_with_incomplete)}/{corner_count}")
