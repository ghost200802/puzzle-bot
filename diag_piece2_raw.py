import sys
sys.path.insert(0, 'src')
from common.vector import Vector
import math

pid = 2
img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
p = Vector.from_file(img_path, id=pid)
p.find_border_raster()
p.vectorize()

candidates_raw = p.find_corner_candidates()

target = (186, 200)
print("Raw candidates near (186,200):")
for c in candidates_raw:
    d = math.sqrt((c.v[0]-target[0])**2 + (c.v[1]-target[1])**2)
    if d < 80:
        print(f"  ({c.v[0]:4d},{c.v[1]:4d}) i={c.i} angle={c.angle*180/math.pi:.1f}° score={c.score():.2f} dist={d:.0f}")

print(f"\nTotal raw candidates: {len(candidates_raw)}")
print(f"merge_dist_px threshold: {5 * p.scalar:.0f}")
