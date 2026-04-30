import sys, os
sys.path.insert(0, 'src')
from common.vector import Vector
from common import util

d = 'output/puzzle_run/3_vector'

for pid in [13, 43, 46, 84, 86]:
    bmp_path = os.path.join(d, f'piece_{pid}.bmp')
    v = Vector.from_file(bmp_path, pid)
    v.find_border_raster()
    v.vectorize()
    v.find_four_corners()
    v.extract_four_sides()
    
    print(f"\nPiece {pid}: scalar={v.scalar:.2f}, threshold={1.8*v.scalar:.2f}")
    for i, side in enumerate(v.sides):
        area = util.normalized_area_between_corners(side.vertices)
        print(f"  Side {i}: area={area:.2f}, is_edge={area < 1.8*v.scalar}")
