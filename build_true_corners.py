import sys
sys.path.insert(0, 'src')
from common.vector import Vector
import json
import os

user_map = {
    1: [0, "2/4", "1/3", "5/6/7"],
    2: ["11/13", "1/5", "2/4", "3/7"],
    3: ["1/0", 6, 2, 3],
    4: [2, 3, 1, 7],
    5: [4, "2/3", 5, 8],
    6: ["10/12", "1/5", "2/4", "3/7"],
    7: [2, 1, "3/4", 0],
    8: [7, "0/1", "5/6", "2/4"],
    9: ["4/6", 0, 5, 2],
    10: ["7/9", "10/8", "2/6", "3/5"],
}

result = {}

for pid in range(1, 11):
    img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
    p = Vector.from_file(img_path, id=pid)
    p.find_border_raster()
    p.vectorize()
    candidates = p.find_corner_candidates()
    candidates = p.merge_nearby_candidates(candidates)
    bbox = p._get_bbox()
    candidates_sorted = sorted(candidates, key=lambda c: c.score_with_bbox(bbox[0], bbox[1], bbox[2], bbox[3]))

    corners = []
    for corner_spec in user_map[pid]:
        if isinstance(corner_spec, int):
            indices = [corner_spec]
        else:
            indices = [int(x) for x in corner_spec.split('/')]
        
        points = []
        for idx in indices:
            c = candidates_sorted[idx]
            points.append([int(c.v[0]), int(c.v[1])])
        corners.append(points)
    
    result[str(pid)] = corners
    print(f"Piece_{pid}: {corners}")

os.makedirs('input', exist_ok=True)
with open('input/true_corners.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nSaved to input/true_corners.json")
