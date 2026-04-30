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

def detailed_edge_analysis(c, vertices, scalar, dim):
    n = len(vertices)
    i = c.i % n
    step = max(1, round(scalar))
    dim_val = float(dim)

    def scan_direction(start, step_sign):
        base_angle = None
        cumulative_length = 0.0
        j = start
        straight_length = 0.0
        for _ in range(200):
            j = (j + step_sign * step) % n
            j2 = (j + step_sign * step) % n
            dx = float(vertices[j2][0]) - float(vertices[j][0])
            dy = float(vertices[j2][1]) - float(vertices[j][1])
            seg_angle = math.atan2(dy, dx)
            seg_len = math.sqrt(dx * dx + dy * dy)
            cumulative_length += seg_len
            if base_angle is None:
                base_angle = seg_angle
                straight_length = cumulative_length
                continue
            change = abs(seg_angle - base_angle)
            if change > math.pi:
                change = 2 * math.pi - change
            if change > 0.25:
                break
            straight_length = cumulative_length
        return straight_length, base_angle

    fwd_len, fwd_angle = scan_direction(i, 1)
    back_len, back_angle = scan_direction(i, -1)
    angle_diff = abs(fwd_angle - back_angle)
    if angle_diff > math.pi:
        angle_diff = 2 * math.pi - angle_diff
    interior_angle_deg = angle_diff * 180.0 / math.pi
    angle_dev = abs(interior_angle_deg - 90.0) / 90.0
    fwd_ratio = fwd_len / dim_val if dim_val > 0 else 0
    back_ratio = back_len / dim_val if dim_val > 0 else 0
    min_ratio = min(fwd_ratio, back_ratio)
    sum_ratio = fwd_ratio + back_ratio
    return fwd_len, back_len, interior_angle_deg, angle_dev, fwd_ratio, back_ratio, min_ratio, sum_ratio

print("=" * 120)
print(f"{'Piece':>6} | {'Pos':^12} | {'True?':^5} | {'bbox_sc':^7} | {'fwd_len':^7} | {'back_len':^7} | {'int_ang':^7} | {'ang_dev':^7} | {'fwd_r':^6} | {'bk_r':^6} | {'min_r':^6} | {'sum_r':^6}")
print("=" * 120)

for pid in [2, 4, 5, 6, 9, 10]:
    true_c = get_true_corners(pid)
    if not true_c:
        print(f"Piece_{pid}: no true corners")
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

    candidates = p.find_corner_candidates()
    candidates = p.merge_nearby_candidates(candidates)
    bbox = p._get_bbox()

    true_matched = set()
    false_matched = set()
    for dt in det_c:
        best_d = 999
        best_j = -1
        for j, tc in enumerate(true_c):
            d = math.sqrt((dt[0]-tc[0])**2 + (dt[1]-tc[1])**2)
            if d < best_d:
                best_d = d
                best_j = j
        if best_d > 25:
            false_matched.add((dt[0], dt[1]))

    for tc in true_c:
        found = False
        for dt in det_c:
            d = math.sqrt((dt[0]-tc[0])**2 + (dt[1]-tc[1])**2)
            if d < 25:
                found = True
                break
        if not found:
            true_matched.add((tc[0], tc[1]))

    key_positions = set()
    for pos in true_matched:
        key_positions.add(pos)
    for pos in false_matched:
        key_positions.add(pos)

    print(f"\n--- Piece_{pid} (dim={p.dim:.0f}, scalar={p.scalar:.1f}) ---")

    scored_list = []
    for c in candidates:
        is_true = any(abs(c.v[0]-tc[0]) < 30 and abs(c.v[1]-tc[1]) < 30 for tc in true_c)
        is_key = any(abs(c.v[0]-kp[0]) < 30 and abs(c.v[1]-kp[1]) < 30 for kp in key_positions)
        if is_key:
            bbox_sc = c.score_with_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
            fwd_len, back_len, int_ang, ang_dev, fwd_r, bk_r, min_r, sum_r = detailed_edge_analysis(c, p.vertices, p.scalar, p.dim)
            is_true_str = "TRUE" if is_true else "FALSE"
            scored_list.append((c, bbox_sc, fwd_len, back_len, int_ang, ang_dev, fwd_r, bk_r, min_r, sum_r, is_true_str))

    for c, bsc, fl, bl, ia, ad, fr, br, mr, sr, t in sorted(scored_list, key=lambda x: x[1]):
        print(f"  {pid:>4} | ({c.v[0]:4d},{c.v[1]:4d}) | {t:^5} | {bsc:7.2f} | {fl:7.1f} | {bl:7.1f} | {ia:6.1f}° | {ad:7.3f} | {fr:6.3f} | {br:6.3f} | {mr:6.3f} | {sr:6.3f}")
