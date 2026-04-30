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
print(f"Before merge: {len(candidates_raw)} candidates")
for idx, c in enumerate(sorted(candidates_raw, key=lambda x: x.i)):
    print(f"  raw #{idx}: ({c.v[0]:4d},{c.v[1]:4d}) i={c.i} angle={c.angle*180/math.pi:.1f}° score={c.score():.2f}")

candidates_merged = p.merge_nearby_candidates(candidates_raw)
print(f"\nAfter merge: {len(candidates_merged)} candidates")
for idx, c in enumerate(sorted(candidates_merged, key=lambda x: x.score())):
    print(f"  merged #{idx}: ({c.v[0]:4d},{c.v[1]:4d}) i={c.i} angle={c.angle*180/math.pi:.1f}° score={c.score():.2f}")

target = (186, 200)
for c in candidates_raw:
    if abs(c.v[0] - target[0]) < 5 and abs(c.v[1] - target[1]) < 5:
        print(f"\n  Found target in RAW: ({c.v[0]},{c.v[1]}) i={c.i}")
found_merged = False
for c in candidates_merged:
    if abs(c.v[0] - target[0]) < 50 and abs(c.v[1] - target[1]) < 50:
        print(f"  Found nearby in MERGED: ({c.v[0]},{c.v[1]}) i={c.i} dist={math.sqrt((c.v[0]-target[0])**2+(c.v[1]-target[1])**2):.0f}")
        found_merged = True
if not found_merged:
    print("  Target NOT found in merged candidates!")
