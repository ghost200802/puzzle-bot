import sys, os
sys.path.insert(0, 'src')
from common.vector import Vector
from common import util
from PIL import Image
import numpy as np

example_dir = os.path.join('..', 'example_data', '2_segmented')
bmps = sorted([f for f in os.listdir(example_dir) if f.endswith('.bmp')])[:20]

print("=== Example data pieces (original robot mode) ===")
areas = []
for b in bmps:
    bmp_path = os.path.join(example_dir, b)
    v = Vector.from_file(bmp_path, 0)
    scalar = v.scalar
    v.find_border_raster()
    v.vectorize()
    v.find_four_corners()
    v.extract_four_sides()
    
    edge_count = sum(1 for s in v.sides if s.is_edge)
    for i, side in enumerate(v.sides):
        area = util.normalized_area_between_corners(side.vertices)
        areas.append((area, side.is_edge, scalar))
        if area < 30:
            print(f"  {b}: side {i}: area={area:.2f}, is_edge={side.is_edge}, scalar={scalar:.2f}, threshold={0.75*scalar:.2f}")

print(f"\nTotal sides analyzed: {len(areas)}")
edge_areas = [a for a, ie, s in areas if ie]
nonedge_areas = [a for a, ie, s in areas if not ie]
print(f"Edge sides: {len(edge_areas)}, area range: {min(edge_areas):.2f} - {max(edge_areas):.2f}, mean: {sum(edge_areas)/len(edge_areas):.2f}")
print(f"Non-edge sides: {len(nonedge_areas)}, area range: {min(nonedge_areas):.2f} - {max(nonedge_areas):.2f}, mean: {sum(nonedge_areas)/len(nonedge_areas):.2f}")
