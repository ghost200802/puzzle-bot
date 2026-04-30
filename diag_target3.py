import sys
sys.path.insert(0, 'src')
from common.vector import Vector, Candidate
import math

pid = 2
img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
p = Vector.from_file(img_path, id=pid)
p.find_border_raster()
p.vectorize()

c = Candidate.from_vertex(p.vertices, 2367, p.centroid, debug=False, scalar=p.scalar)
if c:
    print(f"angle={c.angle*180/math.pi:.1f}° stdev={c.stdev:.3f} offset={c.offset_from_center*180/math.pi:.1f}° curve={c.curve_score:.2f}")
    print(f"score={c.score():.2f}")
else:
    print("REJECTED")
