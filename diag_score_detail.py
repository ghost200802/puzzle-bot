import sys
sys.path.insert(0, 'src')
from common.vector import Vector, Candidate
from common import util
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

print(f"bbox: {bbox}")
print(f"centroid: ({p.centroid[0]:.0f}, {p.centroid[1]:.0f})")
print(f"scalar: {p.scalar:.1f}")
print()

for idx in [0, 1, 2, 3]:
    c = candidates_sorted[idx]
    print(f"=== Candidate #{idx}: ({c.v[0]}, {c.v[1]}) ===")
    
    angle_error = max(0, c.angle - math.pi/2)
    stdev_sq = c.stdev ** 2
    base = c.score()
    print(f"  angle: {c.angle*180/math.pi:.1f}° => angle_error={angle_error:.3f} => 0.7*ae={0.7*angle_error:.3f}")
    print(f"  offset_from_center: {c.offset_from_center*180/math.pi:.1f}° => 0.4*offset={0.4*c.offset_from_center:.3f}")
    print(f"  stdev: {c.stdev:.4f} => stdev²={stdev_sq:.4f} => 11.0*stdev²={11.0*stdev_sq:.3f}")
    print(f"  curve_score: {c.curve_score:.3f} => 0.8*curve={0.8*c.curve_score:.3f}")
    print(f"  BASE SCORE = {base:.3f}")
    
    x, y = c.v
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    dl = (x - bbox[0]) / bw
    dr = (bbox[2] - x) / bw
    dt = (y - bbox[1]) / bh
    db = (bbox[3] - y) / bh
    min_dist = min(dl, dr, dt, db)
    edge_penalty = 0.0
    if min_dist > 0.05:
        edge_penalty = 5.0 * (min_dist - 0.05)
    print(f"  bbox dists: left={dl:.3f} right={dr:.3f} top={dt:.3f} bottom={db:.3f}")
    print(f"  min_dist_to_edge: {min_dist:.3f} => edge_penalty={edge_penalty:.3f}")
    print(f"  TOTAL bbox_score = {base + edge_penalty:.3f}")
    print()

    vec_offset = 1 if p.scalar < 2 else 2
    vec_len_for_stdev = round(8 * p.scalar)
    n = len(p.vertices)
    i = c.i % n

    print(f"  --- Spoke analysis (vec_len_for_stdev={vec_len_for_stdev}) ---")
    
    pts_h = util.slice(p.vertices, i-vec_len_for_stdev-vec_offset, i-vec_offset-1)
    pts_j = util.slice(p.vertices, i+vec_offset+1, i+vec_len_for_stdev+vec_offset)
    
    _, stdev_h = util.colinearity(from_point=p.vertices[i], to_points=pts_h)
    _, stdev_j = util.colinearity(from_point=p.vertices[i], to_points=pts_j)
    print(f"  stdev_h={stdev_h:.4f}, stdev_j={stdev_j:.4f} (sum={stdev_h+stdev_j:.4f})")
    print(f"  spoke_h has {len(pts_h)} points, spoke_j has {len(pts_j)} points")
    
    if len(pts_h) > 0:
        print(f"  spoke_h start: {pts_h[0]}, end: {pts_h[-1]}")
    if len(pts_j) > 0:
        print(f"  spoke_j start: {pts_j[0]}, end: {pts_j[-1]}")
    print()
