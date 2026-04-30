import sys
sys.path.insert(0, 'src')
from common.vector import Vector
import math

pid = 1
img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
p = Vector.from_file(img_path, id=pid)
p.find_border_raster()
p.vectorize()
candidates = p.find_corner_candidates()
candidates = p.merge_nearby_candidates(candidates)
bbox = p._get_bbox()
candidates_sorted = sorted(candidates, key=lambda c: c.score_with_bbox(bbox[0], bbox[1], bbox[2], bbox[3]))

print(f"Piece_1: scalar={p.scalar:.1f}, merge_threshold={2*p.scalar:.0f} indices, total_vertices={len(p.vertices)}")
print(f"\n{'idx':>3} | {'pos':^14} | {'vertex_i':>8} | {'bbox_sc':>7} | {'stdev':>6} | {'angle':>6}")
print("-" * 70)
for idx, c in enumerate(candidates_sorted):
    print(f"{idx:3d} | ({c.v[0]:4d},{c.v[1]:4d}) | {c.i:8d} | {c.score_with_bbox(bbox[0], bbox[1], bbox[2], bbox[3]):7.2f} | {c.stdev:6.3f} | {c.angle*180/math.pi:5.1f}°")

print(f"\n--- Pairs the user says should be merged ---")
pairs = [(6,7), (1,3), (2,4), (5,6), (5,7)]
for a, b in pairs:
    ca, cb = candidates_sorted[a], candidates_sorted[b]
    px_dist = math.sqrt((ca.v[0]-cb.v[0])**2 + (ca.v[1]-cb.v[1])**2)
    idx_diff = abs(ca.i - cb.i)
    idx_diff_wrap = len(p.vertices) - idx_diff
    min_idx_diff = min(idx_diff, idx_diff_wrap)
    print(f"  #{a}({ca.v[0]},{ca.v[1]}) i={ca.i}  <->  #{b}({cb.v[0]},{cb.v[1]}) i={cb.i}")
    print(f"    pixel_dist={px_dist:.1f}px, index_diff={idx_diff} (wrap={idx_diff_wrap}, min={min_idx_diff}), threshold={2*p.scalar:.0f}")
    print(f"    merge? {'YES' if min_idx_diff <= 2*p.scalar else 'NO (index too far)'}")
