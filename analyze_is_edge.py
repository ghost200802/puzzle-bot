import sys, os, json
sys.path.insert(0, 'src')
from common.vector import Vector
from common import util

vector_dir = 'output/puzzle_run/3_vector'

# Test pieces that should be corners/edges
test_pieces = [13, 43, 46, 84, 86, 2, 7, 12, 15, 1, 10]

for pid in test_pieces:
    bmp_path = os.path.join(vector_dir, f'piece_{pid}.bmp')
    if not os.path.exists(bmp_path):
        continue
    v = Vector.from_file(bmp_path, pid)
    v.find_border_raster()
    v.vectorize()
    v.find_four_corners()
    v.extract_four_sides()
    
    print(f"\nPiece {pid}: scalar={v.scalar:.2f}, threshold={0.75*v.scalar:.2f}")
    for i, side in enumerate(v.sides):
        area = util.normalized_area_between_corners(side.vertices)
        is_edge = bool(area < 0.75 * v.scalar)
        print(f"  Side {i}: area={area:.2f}, threshold={0.75*v.scalar:.2f}, is_edge={is_edge}")
