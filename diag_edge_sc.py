import sys
sys.path.insert(0, 'src')
from common.vector import Vector, Candidate
import json
import math

def get_true_corners(pid):
    sides = []
    for si in range(4):
        path = f'output/puzzle_run/3_vector/side_{pid}_{si}.json'
        try:
            with open(path) as f:
                data = json.load(f)
            sides.append(data['vertices'])
        except:
            return None
    return [tuple(sides[3][-1]), tuple(sides[0][-1]), tuple(sides[1][-1]), tuple(sides[2][-1])]

for pid in range(1, 11):
    true_c = get_true_corners(pid)
    if not true_c:
        continue
    try:
        img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
        p = Vector.from_file(img_path, id=pid)
        p.find_border_raster()
        p.vectorize()
        p.find_four_corners()
        det_c = [(int(c[0]), int(c[1])) for c in p.corners]
    except Exception as e:
        print(f"Piece_{pid}: ERROR: {e}")
        continue

    det_ok = True
    for i, dt in enumerate(det_c):
        best_d = 999
        best_j = -1
        for j, tc in enumerate(true_c):
            d = math.sqrt((dt[0]-tc[0])**2 + (dt[1]-tc[1])**2)
            if d < best_d:
                best_d = d
                best_j = j
        if best_d > 25:
            det_ok = False

    print(f"\n{'='*60}")
    print(f"Piece_{pid} {'OK' if det_ok else 'FAIL'}")
    print(f"  Detected: {det_c}")
    print(f"  True:     {true_c}")

    candidates = p.find_corner_candidates()
    candidates = p.merge_nearby_candidates(candidates)
    bbox = p._get_bbox()

    print(f"\n  Candidates with edge_sc (top 20 by bbox score):")
    scored_list = []
    for c in candidates:
        bbox_sc = c.score_with_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
        edge_sc = c.score_with_straight_edges(p.vertices, p.scalar, p.dim)
        is_true = any(abs(c.v[0]-tc[0]) < 30 and abs(c.v[1]-tc[1]) < 30 for tc in true_c)
        scored_list.append((bbox_sc, edge_sc, c.v, is_true, c.i))

    scored_list.sort(key=lambda x: x[0])
    for j, (bsc, esc, v, is_true, idx) in enumerate(scored_list[:20]):
        mark = "TRUE" if is_true else "    "
        print(f"    {j:2d}. bbox={bsc:6.2f} edge_sc={esc:6.3f} {mark} ({v[0]:4d},{v[1]:4d}) i={idx}")
