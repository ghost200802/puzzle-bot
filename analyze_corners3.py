import sys, os, json
sys.path.insert(0, 'src')
from common.vector import Vector
from common import util

conn_path = 'output/puzzle_run/5_connectivity/connectivity.json'
with open(conn_path) as f:
    connectivity = json.load(f)

no_match_counts = {}
for pid_str, sides in connectivity.items():
    pid = int(pid_str)
    nm = sum(1 for s in sides if len(s) == 0)
    no_match_counts[pid] = nm

corners_2nm = sorted([pid for pid, nm in no_match_counts.items() if nm == 2])
edges_1nm = sorted([pid for pid, nm in no_match_counts.items() if nm == 1])

print(f"Corners (2 no-match): {corners_2nm} ({len(corners_2nm)})")
print(f"Edges (1 no-match): {edges_1nm} ({len(edges_1nm)})")

# Now re-vectorize these "corners" and check their area values
vector_dir = 'output/puzzle_run/3_vector'
print("\n--- Area values for detected corners ---")
for pid in corners_2nm:
    bmp_path = os.path.join(vector_dir, f'piece_{pid}.bmp')
    if not os.path.exists(bmp_path):
        continue
    v = Vector.from_file(bmp_path, pid)
    v.find_border_raster()
    v.vectorize()
    v.find_four_corners()
    v.extract_four_sides()
    
    print(f"\nPiece {pid}: scalar={v.scalar:.2f}, threshold={2.5*v.scalar:.2f}")
    nm_sides = [i for i, s in enumerate(connectivity[str(pid)]) if len(s) == 0]
    for i, side in enumerate(v.sides):
        area = util.normalized_area_between_corners(side.vertices)
        is_edge = bool(area < 2.5 * v.scalar)
        nm = "NO_MATCH" if i in nm_sides else ""
        print(f"  Side {i}: area={area:.2f}, is_edge={is_edge} {nm}")
