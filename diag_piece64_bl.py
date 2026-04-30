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

d = p.scalar / 2.0
n = len(p.vertices)

target_i = 1598
start = target_i - 80
end = target_i + 80

lines = []
lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
lines.append(f'<svg width="800" height="800" viewBox="-10 -10 820 820" xmlns="http://www.w3.org/2000/svg">')

border_sampled = p.vertices[::3]
border_pts = ' '.join([f'{float(v[0])/d:.2f},{float(v[1])/d:.2f}' for v in border_sampled])
lines.append(f'<polyline points="{border_pts}" style="fill:none; stroke:#cccccc; stroke-width:1.5" />')

zoom_pts = []
for i in range(start, end):
    idx = i % n
    zoom_pts.append(p.vertices[idx])
zoom_str = ' '.join([f'{float(v[0])/d:.2f},{float(v[1])/d:.2f}' for v in zoom_pts])
lines.append(f'<polyline points="{zoom_str}" style="fill:none; stroke:#ff4444; stroke-width:2.5" />')

for offset in range(-30, 31, 3):
    ti = (target_i + offset) % n
    x, y = float(p.vertices[ti][0])/d, float(p.vertices[ti][1])/d
    c = Candidate.from_vertex(p.vertices, ti, p.centroid, debug=False, scalar=p.scalar)
    if c:
        angle_deg = c.angle * 180 / math.pi
        fill = '#00cc00' if angle_deg < 150 else '#ff4444'
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" style="fill:{fill}; stroke:#000; stroke-width:0.5" />')
        if offset % 9 == 0:
            lines.append(f'<text x="{x+5:.2f}" y="{y-5:.2f}" font-size="7" font-family="monospace" fill="#000">d={offset:+d} a={angle_deg:.0f}°</text>')
    else:
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" style="fill:#888; stroke-width:0" />')
        if offset % 9 == 0:
            lines.append(f'<text x="{x+5:.2f}" y="{y-5:.2f}" font-size="7" font-family="monospace" fill="#888">d={offset:+d} REJ</text>')

lines.append(f'<text x="10" y="20" font-size="10" font-family="monospace" fill="#000">Piece_64 bottom-left region</text>')
lines.append(f'<text x="10" y="34" font-size="9" font-family="monospace" fill="#000">Green=passed angle filter, Red=>150 deg, Gray=rejected</text>')

lines.append('</svg>')

out_dir = 'output/puzzle_run/3_vector/candidates'
svg_path = os.path.join(out_dir, 'piece64_bottomleft.svg')
with open(svg_path, 'w') as f:
    f.write('\n'.join(lines))
print(f"Saved: {svg_path}")

print(f"\n=== Detailed angle scan around i={target_i} ===")
print(f"{'delta':>5} | {'pos':^14} | {'angle':^7} | {'stdev':^7} | {'score':^7} | {'status':^8}")
print("-" * 60)
for offset in range(-30, 31, 2):
    ti = (target_i + offset) % n
    c = Candidate.from_vertex(p.vertices, ti, p.centroid, debug=False, scalar=p.scalar)
    if c:
        status = "OK" if c.score() < 3.0 else "FILTERED"
        print(f"{offset:+5d} | ({c.v[0]:4d},{c.v[1]:4d}) | {c.angle*180/math.pi:5.1f}° | {c.stdev:5.3f} | {c.score():5.2f} | {status}")
    else:
        print(f"{offset:+5d} | ({p.vertices[ti][0]:4d},{p.vertices[ti][1]:4d}) | REJECTED")
