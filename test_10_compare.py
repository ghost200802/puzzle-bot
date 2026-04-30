import sys
sys.path.insert(0, 'src')
from common.vector import Vector
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

print(f"{'Piece':>8} | {'Detected':^50} | {'True':^50} | Match")
print("-" * 140)

all_pass = True
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
        print(f"  Piece_{pid:2d} | ERROR: {e}")
        all_pass = False
        continue

    results = []
    used = set()
    for dt in det_c:
        best_d = 999
        best_j = -1
        for j, tc in enumerate(true_c):
            if j in used:
                continue
            d = math.sqrt((dt[0]-tc[0])**2 + (dt[1]-tc[1])**2)
            if d < best_d:
                best_d = d
                best_j = j
        if best_d < 50:
            results.append(f"{best_d:.0f}px")
            used.add(best_j)
        else:
            results.append(f"MISS({best_d:.0f})")

    ok = all("MISS" not in r for r in results) and all(int(r.replace('px','')) < 25 for r in results)
    if not ok:
        all_pass = False

    det_str = "  ".join([f"({c[0]},{c[1]})" for c in det_c])
    true_str = "  ".join([f"({c[0]},{c[1]})" for c in true_c])
    match_str = "  ".join(results)
    status = "OK" if ok else "FAIL"
    print(f"  Piece_{pid:2d} | {det_str:50s} | {true_str:50s} | {match_str:30s} [{status}]")

print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
