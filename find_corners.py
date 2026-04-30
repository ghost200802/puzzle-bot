import sys, os
sys.path.insert(0, 'src')
from common.vector import Vector
from common import util

d = 'output/puzzle_run/3_vector'
import glob

results = []
for bmp_path in sorted(glob.glob(os.path.join(d, 'piece_*.bmp'))):
    pid = int(os.path.basename(bmp_path).replace('piece_','').replace('.bmp',''))
    v = Vector.from_file(bmp_path, pid)
    v.find_border_raster()
    v.vectorize()
    v.find_four_corners()
    v.extract_four_sides()
    
    areas = []
    for i, side in enumerate(v.sides):
        area = util.normalized_area_between_corners(side.vertices)
        areas.append(area)
    
    areas_sorted = sorted(areas)
    results.append((pid, areas, areas_sorted[:2]))

# Sort by sum of 2 smallest areas
results.sort(key=lambda r: r[2][0] + r[2][1])

print("Top 20 pieces with smallest 2 area values (potential corners):")
for pid, areas, top2 in results[:20]:
    print(f"  Piece {pid}: areas={[f'{a:.1f}' for a in areas]}, smallest 2: {top2[0]:.1f} + {top2[1]:.1f} = {top2[0]+top2[1]:.1f}")
