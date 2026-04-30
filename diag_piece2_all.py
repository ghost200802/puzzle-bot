import sys
sys.path.insert(0, 'src')
from common.vector import Vector
from common import util
import math

pid = 2
img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
p = Vector.from_file(img_path, id=pid)
p.find_border_raster()
p.vectorize()
candidates = p.find_corner_candidates()
candidates = p.merge_nearby_candidates(candidates)
bbox = p._get_bbox()
candidates_sorted = sorted(candidates, key=lambda c: c.score())

print(f"Piece_2: {len(candidates_sorted)} candidates\n")

for idx in range(len(candidates_sorted)):
    c = candidates_sorted[idx]
    angle_error = max(0, c.angle - math.pi/2)
    ae_raw = angle_error
    if angle_error > math.pi/6:
        angle_error = angle_error + 3.0 * (angle_error - math.pi/6)
    base = c.score()
    print(f"#{idx} ({c.v[0]:4d},{c.v[1]:4d}) angle={c.angle*180/math.pi:.1f}° ae_raw={ae_raw:.2f} ae_penalized={angle_error:.2f} stdev={c.stdev:.3f} offset={c.offset_from_center*180/math.pi:.1f}° curve={c.curve_score:.2f} score={base:.2f}")
