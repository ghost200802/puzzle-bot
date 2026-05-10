import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))

import importlib
import common.texture_verify as tv
importlib.reload(tv)
from common.texture_verify import (
    load_side_data, _resample_polyline,
    _find_corresponding_points_on_edge, _compute_transform,
    _apply_transform, _apply_inverse_transform, N_SAMPLES
)
from show_connectivity import _get_outline, _load_piece_data

from pipline.config import get_vector_path, get_check_path
from pipline.debug_utils import load_fonts, transform_point, CanvasViewport

DEDUPED_PATH = get_vector_path()
CHECK_PATH = get_check_path()

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

fonts = load_fonts(title_size=20, idx_size=10)

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)

verts_a = side_a['vertices']
verts_b = side_b['vertices']
verts_bf = verts_b[::-1].copy()

sample_a = _resample_polyline(verts_a, N_SAMPLES)

src_mid, tgt_mid, rot = _compute_transform(verts_b, verts_a)
verts_bf_aligned = _apply_transform(verts_bf, src_mid, tgt_mid, rot)
corr_aligned = _find_corresponding_points_on_edge(sample_a, verts_bf_aligned)
corr_orig = _apply_inverse_transform(corr_aligned, src_mid, tgt_mid, rot)
corr_aligned_back = _apply_transform(corr_orig, src_mid, tgt_mid, rot)

piece_data = _load_piece_data(DEDUPED_PATH)
outline_a = _get_outline(piece_data[PID_A])
outline_b = [transform_point(np.array(p), src_mid, tgt_mid, rot) for p in _get_outline(piece_data[PID_B])]

all_pts = outline_a + outline_b + [p.tolist() for p in sample_a] + [p.tolist() for p in corr_aligned_back]
vp = CanvasViewport(all_pts, canvas_w=1600, target_h=900, margin=80, top_offset=50)
tc = vp.tc
draw = vp.draw

draw.text((15, 8),
          "Debug: direction to NEXT point only  |  Blue=A  Red=B  |  Points 8-10,16-18 highlighted",
          fill=(255, 255, 255, 255), font=fonts['title'])

for outline, color in [(outline_a, (60,100,180,140)), (outline_b, (180,60,60,140))]:
    pts = [tc(x, y) for x, y in outline]
    if len(pts) >= 3:
        draw.polygon(pts, fill=None, outline=color, width=1)

a_edge = [tc(v[0], v[1]) for v in sample_a]
draw.line(a_edge, fill=(80,130,220,80), width=1)
b_edge = [tc(p[0], p[1]) for p in corr_aligned_back]
draw.line(b_edge, fill=(220,80,80,80), width=1)

ARROW_LEN = 20

def draw_arrow(draw, x0, y0, dx, dy, color, width=2):
    x1 = x0 + dx
    y1 = y0 + dy
    draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
    length = math.sqrt(dx*dx + dy*dy)
    if length < 1e-3:
        return
    ux, uy = dx/length, dy/length
    px, py = -uy, ux
    head = 5
    draw.polygon([
        (x1, y1),
        (x1 - ux*head + px*head*0.5, y1 - uy*head + py*head*0.5),
        (x1 - ux*head - px*head*0.5, y1 - uy*head - py*head*0.5),
    ], fill=color)

PROBLEM = set([8, 9, 10, 16, 17, 18])

for i in range(N_SAMPLES):
    is_p = i in PROBLEM
    r = 6 if is_p else 3

    ax, ay = tc(sample_a[i][0], sample_a[i][1])
    draw.ellipse([ax-r, ay-r, ax+r, ay+r], fill=(50,140,255,255), outline=(255,255,255,150) if is_p else (255,255,255,80))
    draw.text((ax+7, ay-8), str(i), fill=(0,255,100,255), font=fonts['idx'])

    a_tangents = []
    if i > 0:
        d = sample_a[i] - sample_a[i-1]
        dl = np.linalg.norm(d)
        if dl > 1e-6:
            a_tangents.append(d / dl)
    if i < N_SAMPLES - 1:
        d = sample_a[i+1] - sample_a[i]
        dl = np.linalg.norm(d)
        if dl > 1e-6:
            a_tangents.append(d / dl)
    if a_tangents:
        a_t = np.mean(a_tangents, axis=0)
        atl = np.linalg.norm(a_t)
        if atl > 1e-6:
            a_t = a_t / atl
            draw_arrow(draw, ax, ay, a_t[0]*ARROW_LEN, a_t[1]*ARROW_LEN, (100,180,255,220), 2)
            a_n = np.array([a_t[1], -a_t[0]])
            draw_arrow(draw, ax, ay, a_n[0]*ARROW_LEN, a_n[1]*ARROW_LEN, (0,100,255,200), 3 if is_p else 2)

    bx, by = tc(corr_aligned_back[i][0], corr_aligned_back[i][1])
    draw.ellipse([bx-r, by-r, bx+r, by+r], fill=(255,80,80,255), outline=(255,255,255,150) if is_p else (255,255,255,80))

    b_tangents = []
    if i > 0:
        d = corr_aligned_back[i] - corr_aligned_back[i-1]
        dl = np.linalg.norm(d)
        if dl > 1e-6:
            b_tangents.append(d / dl)
    if i < N_SAMPLES - 1:
        d = corr_aligned_back[i+1] - corr_aligned_back[i]
        dl = np.linalg.norm(d)
        if dl > 1e-6:
            b_tangents.append(d / dl)
    if b_tangents:
        b_t = np.mean(b_tangents, axis=0)
        btl = np.linalg.norm(b_t)
        if btl > 1e-6:
            b_t = b_t / btl
            draw_arrow(draw, bx, by, b_t[0]*ARROW_LEN, b_t[1]*ARROW_LEN, (255,130,130,220), 2)
            b_n = np.array([-b_t[1], b_t[0]])
            draw_arrow(draw, bx, by, b_n[0]*ARROW_LEN, b_n[1]*ARROW_LEN, (255,50,50,200), 3 if is_p else 2)

ly = vp.canvas.size[1] - 16
draw.text((15, ly), "Thin=tangent(next)  Bold=normal  Blue(A):CW  Red(B):CCW  |  No center dependency!",
          fill=(180,180,180,255), font=fonts['idx'])

out_path = os.path.join(CHECK_PATH, f'debug_next_dir_{PID_A}_{PID_B}.png')
vp.canvas.save(out_path)
print(f"Saved: {out_path}")

print("\nProblem points - direction to next:")
print(f"  {'idx':>3}  {'A_dx':>7} {'A_dy':>7} {'A_angle':>8}  |  {'B_dx':>7} {'B_dy':>7} {'B_angle':>8}")
for i in sorted(PROBLEM):
    if i < N_SAMPLES - 1:
        ad = sample_a[i+1] - sample_a[i]
        adl = np.linalg.norm(ad)
        aa = math.degrees(math.atan2(ad[1], ad[0])) if adl > 1e-6 else 0

        bd = corr_aligned_back[i+1] - corr_aligned_back[i]
        bdl = np.linalg.norm(bd)
        ba = math.degrees(math.atan2(bd[1], bd[0])) if bdl > 1e-6 else 0

        print(f"  [{i:2d}]  {ad[0]:+7.1f} {ad[1]:+7.1f} {aa:>+8.1f}\u00b0  |  {bd[0]:+7.1f} {bd[1]:+7.1f} {ba:>+8.1f}\u00b0")
