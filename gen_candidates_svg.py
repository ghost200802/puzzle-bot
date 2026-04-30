import sys
sys.path.insert(0, 'src')
from common.vector import Vector
import json
import math
import os

output_dir = 'output/puzzle_run/3_vector/candidates'

os.makedirs(output_dir, exist_ok=True)

for pid in range(1, 11):
    img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
    if not os.path.exists(img_path):
        print(f"Piece_{pid}: file not found")
        continue

    try:
        p = Vector.from_file(img_path, id=pid)
        p.find_border_raster()
        p.vectorize()
    except Exception as e:
        print(f"Piece_{pid}: ERROR: {e}")
        continue

    candidates = p.find_corner_candidates()
    candidates = p.merge_nearby_candidates(candidates)
    bbox = p._get_bbox()
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    d = p.scalar / 2.0

    svg = f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    svg += f'<svg width="{3 * p.width / d}" height="{3 * p.height / d}" viewBox="-30 -30 {60 + p.width / d} {60 + p.height /d}" xmlns="http://www.w3.org/2000/svg">'

    border_pts = ' '.join([f'{v[0]/d},{v[1]/d}' for v in p.vertices])
    svg += f'<polyline points="{border_pts}" style="fill:none; stroke:#888888; stroke-width:1.5" />'

    candidates_sorted = sorted(candidates, key=lambda c: c.score_with_bbox(bbox[0], bbox[1], bbox[2], bbox[3]))

    for idx, c in enumerate(candidates_sorted):
        bbox_sc = c.score_with_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
        edge_sc = c.score_with_straight_edges(p.vertices, p.scalar, p.dim)

        x, y = c.v[0] / d, c.v[1] / d
        r = 4.0
        fill_color = '#00cc00' if bbox_sc < 2.5 else '#ffaa00' if bbox_sc < 3.0 else '#ff4444'
        svg += f'<circle cx="{x}" cy="{y}" r="{r}" style="fill:{fill_color}; stroke:#000; stroke-width:0.5" />'
        label_x = x + 6
        label_y = y - 6
        svg += f'<text x="{label_x}" y="{label_y}" font-size="8" font-family="monospace" fill="#000">{idx}</text>'
        svg += f'<text x="{label_x}" y="{label_y + 10}" font-size="6" font-family="monospace" fill="#555">b:{bbox_sc:.1f} e:{edge_sc:.2f}</text>'

    svg += f'<circle cx="{p.centroid[0]/d}" cy="{p.centroid[1]/d}" r="2" style="fill:#4444ff; stroke-width:0" />'

    svg += '</svg>'

    svg_path = os.path.join(output_dir, f'candidates_piece_{pid}.svg')
    with open(svg_path, 'w') as f:
        f.write(svg)

    print(f"Piece_{pid}: {len(candidates_sorted)} candidates -> {svg_path}")
