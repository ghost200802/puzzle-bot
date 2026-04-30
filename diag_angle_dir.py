import sys
sys.path.insert(0, 'src')
from common.vector import Vector
from common import util
import math
import numpy as np

pid = 1
img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
p = Vector.from_file(img_path, id=pid)
p.find_border_raster()
p.vectorize()
candidates = p.find_corner_candidates()
candidates = p.merge_nearby_candidates(candidates)
bbox = p._get_bbox()
candidates_sorted = sorted(candidates, key=lambda c: c.score_with_bbox(bbox[0], bbox[1], bbox[2], bbox[3]))

for cidx in [0, 1, 2, 3]:
    c = candidates_sorted[cidx]
    vec_offset = 1 if p.scalar < 2 else 2
    vec_len_for_stdev = round(8 * p.scalar)
    stdev_step = max(1, round(0.75 * p.scalar))
    n = len(p.vertices)
    i = c.i % n

    pts_h = util.slice(p.vertices, i-vec_len_for_stdev-vec_offset, i-vec_offset-1, step=stdev_step)
    pts_j = util.slice(p.vertices, i+vec_offset+1, i+vec_len_for_stdev+vec_offset, step=stdev_step)

    avg_h, stdev_h = util.colinearity(from_point=p.vertices[i], to_points=pts_h)
    avg_j, stdev_j = util.colinearity(from_point=p.vertices[i], to_points=pts_j)

    seg_angles_h = []
    seg_angles_j = []
    seg_lens_h = []
    seg_lens_j = []

    for k in range(1, len(pts_h)):
        dx = float(pts_h[k][0]) - float(pts_h[k-1][0])
        dy = float(pts_h[k][1]) - float(pts_h[k-1][1])
        seg_angles_h.append(math.atan2(dy, dx))
        seg_lens_h.append(math.sqrt(dx*dx + dy*dy))

    for k in range(1, len(pts_j)):
        dx = float(pts_j[k][0]) - float(pts_j[k-1][0])
        dy = float(pts_j[k][1]) - float(pts_j[k-1][1])
        seg_angles_j.append(math.atan2(dy, dx))
        seg_lens_j.append(math.sqrt(dx*dx + dy*dy))

    if seg_angles_h:
        avg_h_seg = np.arctan2(np.mean(np.sin(seg_angles_h)), np.mean(np.cos(seg_angles_h)))
    else:
        avg_h_seg = avg_h
    if seg_angles_j:
        avg_j_seg = np.arctan2(np.mean(np.sin(seg_angles_j)), np.mean(np.cos(seg_angles_j)))
    else:
        avg_j_seg = avg_j

    print(f"#{cidx} ({c.v[0]},{c.v[1]}):")
    print(f"  h: colinearity_avg={avg_h*180/math.pi:.1f}°  seg_direction_avg={avg_h_seg*180/math.pi:.1f}°  diff={abs(avg_h-avg_h_seg)*180/math.pi:.1f}°")
    print(f"  j: colinearity_avg={avg_j*180/math.pi:.1f}°  seg_direction_avg={avg_j_seg*180/math.pi:.1f}°  diff={abs(avg_j-avg_j_seg)*180/math.pi:.1f}°")
    print(f"  seg_angles_h: {[f'{a*180/math.pi:.1f}' for a in seg_angles_h]}")
    print(f"  seg_angles_j: {[f'{a*180/math.pi:.1f}' for a in seg_angles_j]}")
    print()
