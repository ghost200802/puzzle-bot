import sys
sys.path.insert(0, 'src')
from common.vector import Vector, Candidate
from common import util
import math
import os

pid = 64
img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
p = Vector.from_file(img_path, id=pid)
p.find_border_raster()
p.vectorize()
p.find_four_corners()
det_c = [(int(c[0]), int(c[1])) for c in p.corners]

d = p.scalar / 2.0
out_dir = 'output/puzzle_run/3_vector/candidates'
os.makedirs(out_dir, exist_ok=True)

lines = []
lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
lines.append(f'<svg width="{3 * p.width / d}" height="{3 * p.height / d}" viewBox="-10 -10 {20 + p.width / d} {20 + p.height /d}" xmlns="http://www.w3.org/2000/svg">')

border_sampled = p.vertices[::3]
border_pts = ' '.join([f'{float(v[0])/d:.2f},{float(v[1])/d:.2f}' for v in border_sampled])
lines.append(f'<polyline points="{border_pts}" style="fill:none; stroke:#cccccc; stroke-width:1.5" />')

vec_offset = 1 if p.scalar < 2 else 2
vec_len_for_angle = round(15 * p.scalar)
n = len(p.vertices)

for ci, dc in enumerate(det_c):
    dx, dy = float(dc[0])/d, float(dc[1])/d
    lines.append(f'<circle cx="{dx:.2f}" cy="{dy:.2f}" r="5" style="fill:#ff4444; stroke:#000; stroke-width:1" />')
    lines.append(f'<text x="{dx+6:.2f}" y="{dy-4:.2f}" font-size="10" font-family="monospace" fill="#ff4444">C{ci}</text>')

    best_i = -1
    best_dist = 999
    for vi, v in enumerate(p.vertices):
        dist = math.sqrt((v[0]-dc[0])**2 + (v[1]-dc[1])**2)
        if dist < best_dist:
            best_dist = dist
            best_i = vi

    if best_i < 0:
        continue

    pts_h = util.slice(p.vertices, best_i-vec_len_for_angle-vec_offset, best_i-vec_offset-1)
    pts_j = util.slice(p.vertices, best_i+vec_offset+1, best_i+vec_len_for_angle+vec_offset)

    h_pts = ' '.join([f'{float(v[0])/d:.2f},{float(v[1])/d:.2f}' for v in pts_h])
    lines.append(f'<polyline points="{h_pts}" style="fill:none; stroke:#ff4444; stroke-width:2.5" />')

    j_pts = ' '.join([f'{float(v[0])/d:.2f},{float(v[1])/d:.2f}' for v in pts_j])
    lines.append(f'<polyline points="{j_pts}" style="fill:none; stroke:#4444ff; stroke-width:2.5" />')

    avg_h, _ = util.colinearity(from_point=p.vertices[best_i], to_points=pts_h)
    avg_j, _ = util.colinearity(from_point=p.vertices[best_i], to_points=pts_j)
    avg_h_flipped = avg_h + math.pi

    line_h = util.line_from_angle_and_point(angle=avg_h_flipped, point=p.vertices[best_i], length=800)
    line_j = util.line_from_angle_and_point(angle=avg_j, point=p.vertices[best_i], length=800)

    lines.append(f'<line x1="{float(line_h[0][0])/d:.2f}" y1="{float(line_h[0][1])/d:.2f}" x2="{float(line_h[1][0])/d:.2f}" y2="{float(line_h[1][1])/d:.2f}" style="stroke:#ff4444; stroke-width:2; stroke-dasharray=" />')
    lines.append(f'<line x1="{float(line_j[0][0])/d:.2f}" y1="{float(line_j[0][1])/d:.2f}" x2="{float(line_j[1][0])/d:.2f}" y2="{float(line_j[1][1])/d:.2f}" style="stroke:#4444ff; stroke-width:2; stroke-dasharray=" />')

    c_obj = Candidate.from_vertex(p.vertices, best_i, p.centroid, debug=False, scalar=p.scalar)
    angle_deg = c_obj.angle * 180 / math.pi if c_obj else 0
    lines.append(f'<text x="{dx+6:.2f}" y="{dy+12:.2f}" font-size="8" font-family="monospace" fill="#000">{angle_deg:.1f}deg</text>')

lines.append(f'<circle cx="{float(p.centroid[0])/d:.2f}" cy="{float(p.centroid[1])/d:.2f}" r="3" style="fill:#4444ff; stroke-width:0" />')

lines.append('</svg>')

svg_path = os.path.join(out_dir, f'result_piece_{pid}.svg')
with open(svg_path, 'w') as f:
    f.write('\n'.join(lines))

print(f"Piece_{pid}: {det_c}")
print(f"Saved: {svg_path}")
