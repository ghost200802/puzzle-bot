import sys
sys.path.insert(0, 'src')
from common.vector import Vector
import json
import math
import os

for pid in range(1, 11):
    img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
    if not os.path.exists(img_path):
        continue
    p = Vector.from_file(img_path, id=pid)
    p.find_border_raster()
    p.vectorize()
    candidates = p.find_corner_candidates()
    candidates = p.merge_nearby_candidates(candidates)
    bbox = p._get_bbox()
    candidates_sorted = sorted(candidates, key=lambda c: c.score_with_bbox(bbox[0], bbox[1], bbox[2], bbox[3]))

    print(f"\nPiece_{pid}: {len(candidates_sorted)} candidates, scalar={p.scalar:.1f}, merge_threshold={2*p.scalar:.0f} indices")
    
    for idx, c in enumerate(candidates_sorted):
        bbox_sc = c.score_with_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
        neighbors = 0
        for c2 in candidates_sorted:
            if c2 is c:
                continue
            d = math.sqrt((c.v[0]-c2.v[0])**2 + (c.v[1]-c2.v[1])**2)
            if d < 50:
                neighbors += 1
        cluster = f"[{neighbors} nearby]" if neighbors > 0 else ""
        print(f"  {idx:2d}. ({c.v[0]:4d},{c.v[1]:4d}) bbox={bbox_sc:.2f} angle={c.angle*180/math.pi:.1f}° stdev={c.stdev:.3f} {cluster}")
