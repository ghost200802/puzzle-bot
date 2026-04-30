import sys
sys.path.insert(0, 'src')
from common.vector import Vector
import json
import math
import os

with open('input/true_corners.json') as f:
    true_data = json.load(f)

output_dir = 'output/puzzle_run/3_vector/candidates'
os.makedirs(output_dir, exist_ok=True)

for pid in range(1, 11):
    img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
    if not os.path.exists(img_path):
        continue

    try:
        p = Vector.from_file(img_path, id=pid)
        p.find_border_raster()
        p.vectorize()
        p.find_four_corners()
        det_c = [(int(c[0]), int(c[1])) for c in p.corners]
    except Exception as e:
        print(f"Piece_{pid}: ERROR: {e}")
        continue

    true_corners = true_data.get(str(pid), [])
    d = p.scalar / 2.0

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append(f'<svg width="{3 * p.width / d}" height="{3 * p.height / d}" viewBox="-10 -10 {20 + p.width / d} {20 + p.height /d}" xmlns="http://www.w3.org/2000/svg">')

    border_sampled = p.vertices[::3]
    border_pts = ' '.join([f'{float(v[0])/d:.2f},{float(v[1])/d:.2f}' for v in border_sampled])
    lines.append(f'<polyline points="{border_pts}" style="fill:none; stroke:#cccccc; stroke-width:1.5" />')

    for ci, tc_list in enumerate(true_corners):
        for tc in tc_list:
            tx, ty = float(tc[0])/d, float(tc[1])/d
            lines.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="5" style="fill:none; stroke:#00cc00; stroke-width:2" />')

    for ci, dc in enumerate(det_c):
        dx, dy = float(dc[0])/d, float(dc[1])/d
        lines.append(f'<circle cx="{dx:.2f}" cy="{dy:.2f}" r="3" style="fill:#ff4444; stroke:#000; stroke-width:0.5" />')
        lines.append(f'<text x="{dx+6:.2f}" y="{dy-4:.2f}" font-size="9" font-family="monospace" fill="#ff4444">C{ci}</text>')

    lines.append(f'<text x="10" y="20" font-size="10" font-family="monospace" fill="#00cc00">O circle = true corners</text>')
    lines.append(f'<text x="10" y="34" font-size="10" font-family="monospace" fill="#ff4444">Red dot = detected corners</text>')

    results = []
    all_ok = True
    for dc in det_c:
        best_d = 999
        best_ci = -1
        for ci, tc_list in enumerate(true_corners):
            for tc in tc_list:
                dist = math.sqrt((dc[0]-tc[0])**2 + (dc[1]-tc[1])**2)
                if dist < best_d:
                    best_d = dist
                    best_ci = ci
        ok = best_d < 30
        if not ok:
            all_ok = False
        results.append(f"C{best_ci}(d={best_d:.0f})" if ok else f"MISS(d={best_d:.0f})")

    status = "OK" if all_ok else "FAIL"
    lines.append(f'<text x="10" y="48" font-size="10" font-family="monospace" fill="#000">Piece_{pid}: {status} {" ".join(results)}</text>')

    lines.append('</svg>')

    svg_path = os.path.join(output_dir, f'result_piece_{pid}.svg')
    with open(svg_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Piece_{pid}: {status} | det={det_c} | {' '.join(results)}")
