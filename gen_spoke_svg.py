import sys
sys.path.insert(0, 'src')
from common.vector import Vector
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

d = p.scalar / 2.0

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
    avg_h = avg_h + math.pi
    weights_h = util._straightness_weights(pts_h)
    weights_j = util._straightness_weights(pts_j)

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append(f'<svg width="{3 * p.width / d}" height="{3 * p.height / d}" viewBox="-10 -10 {20 + p.width / d} {20 + p.height /d}" xmlns="http://www.w3.org/2000/svg">')

    border_sampled = p.vertices[::3]
    border_pts = ' '.join([f'{float(v[0])/d:.2f},{float(v[1])/d:.2f}' for v in border_sampled])
    lines.append(f'<polyline points="{border_pts}" style="fill:none; stroke:#cccccc; stroke-width:1.5" />')

    def draw_spoke(pts, weights, avg_angle, color, label):
        step = max(1, len(pts) // 10)
        sampled_indices = list(range(0, len(pts), step))[:10]
        if len(sampled_indices) < 10 and len(pts) >= 10:
            sampled_indices = list(range(0, len(pts), len(pts)//10))[:10]

        pts_str = ' '.join([f'{float(v[0])/d:.2f},{float(v[1])/d:.2f}' for v in pts])
        lines.append(f'<polyline points="{pts_str}" style="fill:none; stroke:{color}; stroke-width:2.5" />')

        line = util.line_from_angle_and_point(angle=avg_angle, point=p.vertices[i], length=800)
        lines.append(f'<line x1="{float(line[0][0])/d:.2f}" y1="{float(line[0][1])/d:.2f}" x2="{float(line[1][0])/d:.2f}" y2="{float(line[1][1])/d:.2f}" style="stroke:{color}; stroke-width:2; stroke-dasharray=" />')

        for k, si in enumerate(sampled_indices):
            px, py = float(pts[si][0])/d, float(pts[si][1])/d
            w = weights[si] if si < len(weights) else 0
            w_norm = min(w / 10.0, 1.0)
            r = 2 + 4 * w_norm
            fill = '#00cc00' if w_norm > 0.5 else '#ffaa00' if w_norm > 0.1 else '#ff4444'
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{r:.1f}" style="fill:{fill}; stroke:#000; stroke-width:0.5" />')
            lines.append(f'<text x="{px+5:.2f}" y="{py-5:.2f}" font-size="6" font-family="monospace" fill="#333">{k} w={w:.1f}</text>')

    draw_spoke(pts_h, weights_h, avg_h, '#ff4444', 'h')
    draw_spoke(pts_j, weights_j, avg_j, '#4444ff', 'j')

    cx, cy = float(c.v[0])/d, float(c.v[1])/d
    lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5" style="fill:#00cc00; stroke:#000; stroke-width:1" />')

    bbox_sc = c.score_with_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
    ix, iy = 10, 20
    lines.append(f'<text x="{ix}" y="{iy}" font-size="10" font-family="monospace" fill="#000">#{cidx} ({c.v[0]},{c.v[1]}) bbox={bbox_sc:.2f}</text>')
    lines.append(f'<text x="{ix}" y="{iy+14}" font-size="9" font-family="monospace" fill="#000">stdev: h={stdev_h:.3f} j={stdev_j:.3f} sum={stdev_h+stdev_j:.3f}</text>')
    lines.append(f'<text x="{ix}" y="{iy+28}" font-size="8" font-family="monospace" fill="#333">dot size = weight (green=high, orange=mid, red=low)</text>')

    lines.append('</svg>')

    out_dir = 'output/puzzle_run/3_vector/candidates'
    svg_path = f'{out_dir}/spoke_piece1_{cidx}.svg'
    with open(svg_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"#{cidx} ({c.v[0]},{c.v[1]}): h_pts={len(pts_h)} j_pts={len(pts_j)}")
