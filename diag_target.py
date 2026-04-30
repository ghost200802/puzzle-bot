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

target_i = None
for i, v in enumerate(p.vertices):
    if abs(int(v[0]) - 186) < 3 and abs(int(v[1]) - 200) < 3:
        target_i = i
        break

if target_i is None:
    print("Target point (186,200) not found in vertices!")
    for i, v in enumerate(p.vertices):
        if abs(int(v[0]) - 186) < 10 and abs(int(v[1]) - 200) < 10:
            print(f"  Nearby: ({v[0]},{v[1]}) i={i}")
else:
    print(f"Found target at i={target_i}: ({p.vertices[target_i][0]},{p.vertices[target_i][1]})")
    c = Candidate.from_vertex(p.vertices, target_i, p.centroid, debug=True, scalar=p.scalar)
    if c:
        print(f"  angle={c.angle*180/math.pi:.1f}° score={c.score():.2f}")
    else:
        print("  Candidate was REJECTED")
