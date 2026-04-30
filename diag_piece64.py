import sys
sys.path.insert(0, 'src')
from common.vector import Vector, Candidate
from common import util
import math

pid = 64
img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
p = Vector.from_file(img_path, id=pid)
p.find_border_raster()
p.vectorize()

candidates_raw = p.find_corner_candidates()
candidates_merged = p.merge_nearby_candidates(candidates_raw)
candidates_sorted = sorted(candidates_merged, key=lambda c: c.score())

print(f"Piece_{pid}: dim={p.dim:.0f}, scalar={p.scalar:.1f}, vertices={len(p.vertices)}")
print(f"Raw candidates: {len(candidates_raw)}, Merged: {len(candidates_merged)}")
print(f"bbox: {p._get_bbox()}")
print(f"centroid: ({p.centroid[0]:.0f}, {p.centroid[1]:.0f})")
print()

print("=== ALL candidates (sorted by score) ===")
for idx, c in enumerate(candidates_sorted):
    angle_deg = c.angle * 180 / math.pi
    print(f"  #{idx}: ({c.v[0]:4d},{c.v[1]:4d}) i={c.i:4d} angle={angle_deg:5.1f}° stdev={c.stdev:.3f} offset={c.offset_from_center*180/math.pi:.1f}° curve={c.curve_score:.2f} sym={c.centroid_symmetry:.2f} score={c.score():.2f}")

print("\n=== Sides info ===")
for si, side in enumerate(p.sides):
    print(f"  Side {si}: is_edge={side.is_edge}, vertices={len(side.vertices)}, area={getattr(side, '_area', 'N/A')}")

p.find_four_corners()
det_c = [(int(c[0]), int(c[1])) for c in p.corners]
print(f"\n=== Detected corners: {det_c} ===")

print("\n=== Looking at bottom-left region ===")
bbox = p._get_bbox()
bl_x, bl_y = bbox[0], bbox[3]
print(f"  bbox bottom-left: ({bl_x:.0f}, {bl_y:.0f})")

nearby = []
for c in candidates_sorted:
    d = math.sqrt((c.v[0] - bl_x)**2 + (c.v[1] - bl_y)**2)
    if d < 200:
        nearby.append((c, d))
        print(f"  Candidate ({c.v[0]:4d},{c.v[1]:4d}) i={c.i} dist_to_bl={d:.0f} score={c.score():.2f} angle={c.angle*180/math.pi:.1f}°")

if not nearby:
    print("  No candidates near bottom-left!")
    print("\n  Checking raw candidates near bottom-left:")
    for c in candidates_raw:
        d = math.sqrt((c.v[0] - bl_x)**2 + (c.v[1] - bl_y)**2)
        if d < 200:
            print(f"    Raw ({c.v[0]:4d},{c.v[1]:4d}) i={c.i} dist={d:.0f} score={c.score():.2f} angle={c.angle*180/math.pi:.1f}°")

print("\n=== Bottom-left corner vertex analysis ===")
n = len(p.vertices)
min_dist = 999
min_i = -1
for i in range(n):
    d = math.sqrt((p.vertices[i][0] - bl_x)**2 + (p.vertices[i][1] - bl_y)**2)
    if d < min_dist:
        min_dist = d
        min_i = i
print(f"  Closest vertex to bbox bottom-left: i={min_i} ({p.vertices[min_i][0]:.0f},{p.vertices[min_i][1]:.0f}) dist={min_dist:.0f}")

if min_i >= 0:
    c_test = Candidate.from_vertex(p.vertices, min_i, p.centroid, debug=False, scalar=p.scalar)
    if c_test:
        print(f"    Candidate at closest: angle={c_test.angle*180/math.pi:.1f}° stdev={c_test.stdev:.3f} score={c_test.score():.2f}")
    else:
        print(f"    NOT a valid candidate (rejected by from_vertex)")

    vec_offset = 1 if p.scalar < 2 else 2
    vec_len_for_angle = round(3 * p.scalar)
    a_ih, _ = util.colinearity(from_point=p.vertices[min_i], to_points=util.slice(p.vertices, min_i-vec_len_for_angle-vec_offset, min_i-vec_offset-1))
    a_ij, _ = util.colinearity(from_point=p.vertices[min_i], to_points=util.slice(p.vertices, min_i+vec_offset+1, min_i+vec_len_for_angle+vec_offset))
    a_ih_flipped = a_ih + math.pi
    p_h = (p.vertices[min_i][0] + 10 * math.cos(a_ih_flipped), p.vertices[min_i][1] + 10 * math.sin(a_ih_flipped))
    p_j = (p.vertices[min_i][0] + 10 * math.cos(a_ij), p.vertices[min_i][1] + 10 * math.sin(a_ij))
    angle_hij = util.counterclockwise_angle_between_vectors(p_h, p.vertices[min_i], p_j)
    print(f"    a_ih(flipped)={a_ih_flipped*180/math.pi:.1f}° a_ij={a_ij*180/math.pi:.1f}° angle_hij={angle_hij*180/math.pi:.1f}°")

    for delta in [-20, -10, -5, 0, 5, 10, 20]:
        ti = (min_i + delta) % n
        c_t = Candidate.from_vertex(p.vertices, ti, p.centroid, debug=False, scalar=p.scalar)
        if c_t:
            print(f"    i={ti} (delta={delta:+3d}): ({p.vertices[ti][0]:.0f},{p.vertices[ti][1]:.0f}) angle={c_t.angle*180/math.pi:.1f}° score={c_t.score():.2f}")
