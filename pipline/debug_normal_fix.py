import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

import importlib
import common.texture_verify as tv
importlib.reload(tv)
from common.texture_verify import (
    load_side_data, _resample_polyline, _compute_tangent_at_positions,
    _find_corresponding_points_on_edge, _compute_transform,
    _apply_transform, _apply_inverse_transform,
    INNER_OFFSET, BAND_WIDTH, N_SAMPLES
)
from show_connectivity import _get_outline, _load_piece_data

from config import get_vector_path, get_check_path
from debug_utils import load_fonts, transform_point, rotate_vector

DEDUPED_PATH = get_vector_path()
CHECK_PATH = get_check_path()

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

fonts = load_fonts(title_size=22, idx_size=11, tiny_size=9)

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

tangent_old = np.zeros_like(corr_orig)
for i in range(N_SAMPLES):
    if i == 0:
        tangent_old[i] = corr_orig[1] - corr_orig[0]
    elif i == N_SAMPLES - 1:
        tangent_old[i] = corr_orig[-1] - corr_orig[-2]
    else:
        tangent_old[i] = corr_orig[i + 1] - corr_orig[i - 1]
    tl = np.linalg.norm(tangent_old[i])
    if tl > 1e-6:
        tangent_old[i] /= tl

tangent_new = tv._compute_tangent_at_positions(verts_bf, corr_orig)

def compute_normal(tangent, pos, piece_center):
    normal = np.array([-tangent[1], tangent[0]])
    to_c = piece_center - pos
    if np.dot(normal, to_c) < 0:
        normal = -normal
    return normal

print("Tangent comparison at problem points:")
print(f"  {'idx':>3}  {'OLD angle':>10}  {'NEW angle':>10}  {'diff':>8}")
for i in range(N_SAMPLES):
    old_angle = math.degrees(math.atan2(tangent_old[i][1], tangent_old[i][0]))
    new_angle = math.degrees(math.atan2(tangent_new[i][1], tangent_new[i][0]))
    diff = abs(old_angle - new_angle)
    if diff > 180:
        diff = 360 - diff
    marker = " <<<" if i in [8, 9, 10, 16, 17, 18] else ""
    print(f"  [{i:2d}]  {old_angle:>+9.1f}\u00b0  {new_angle:>+9.1f}\u00b0  {diff:>7.1f}\u00b0{marker}")

piece_data = _load_piece_data(DEDUPED_PATH)
outline_a = _get_outline(piece_data[PID_A])
outline_b = [transform_point(np.array(p), src_mid, tgt_mid, rot) for p in _get_outline(piece_data[PID_B])]

corr_aligned_back = _apply_transform(corr_orig, src_mid, tgt_mid, rot)
tangent_new_aligned = np.array([rotate_vector(t, rot) for t in tangent_new])
tangent_old_aligned = np.array([rotate_vector(t, rot) for t in tangent_old])

from debug_utils import CanvasViewport

all_pts = outline_a + outline_b + [p.tolist() for p in sample_a] + [p.tolist() for p in corr_aligned_back]
vp = CanvasViewport(all_pts, canvas_w=1600, target_h=900, margin=80, top_offset=50)
tc = vp.tc
draw = vp.draw

draw.text((15, 8),
          f"Normal Fix: B tangent from edge (green) vs from corr points (red)  |  Problem points: 8-10, 16-18",
          fill=(255, 255, 255, 255), font=fonts['title'])

for outline, color in [
    (outline_a, (80, 120, 200, 160)),
    (outline_b, (200, 80, 80, 160)),
]:
    pts = [tc(x, y) for x, y in outline]
    if len(pts) >= 3:
        draw.polygon(pts, fill=None, outline=color, width=2)

a_edge = [tc(v[0], v[1]) for v in sample_a]
draw.line(a_edge, fill=(100, 150, 255, 100), width=1)

b_edge = [tc(p[0], p[1]) for p in corr_aligned_back]
draw.line(b_edge, fill=(255, 100, 100, 100), width=1)

ARROW_LEN = 30

for i in range(N_SAMPLES):
    is_problem = i in [8, 9, 10, 16, 17, 18]

    ax, ay = tc(sample_a[i][0], sample_a[i][1])
    r = 5 if is_problem else 3
    draw.ellipse([ax-r, ay-r, ax+r, ay+r], fill=(50, 140, 255, 255), outline=(255,255,255,150))
    draw.text((ax+6, ay-9), str(i), fill=(0, 255, 100, 255), font=fonts['idx'])

    bx, by = tc(corr_aligned_back[i][0], corr_aligned_back[i][1])
    draw.ellipse([bx-r, by-r, bx+r, by+r], fill=(255, 80, 80, 255), outline=(255,255,255,150))

    if is_problem:
        old_tx = tangent_old_aligned[i][0]
        old_ty = tangent_old_aligned[i][1]
        draw.line([(bx, by), (bx + old_tx*ARROW_LEN, by + old_ty*ARROW_LEN)],
                  fill=(255, 50, 50, 200), width=2)

        old_n = rotate_vector(compute_normal(tangent_old[i], corr_orig[i], side_b['piece_center']), rot)
        draw.line([(bx, by), (bx + old_n[0]*ARROW_LEN, by + old_n[1]*ARROW_LEN)],
                  fill=(255, 150, 50, 200), width=2)

    if is_problem:
        new_tx = tangent_new_aligned[i][0]
        new_ty = tangent_new_aligned[i][1]
        draw.line([(bx, by), (bx + new_tx*ARROW_LEN, by + new_ty*ARROW_LEN)],
                  fill=(0, 255, 100, 200), width=2)

        new_n = rotate_vector(compute_normal(tangent_new[i], corr_orig[i], side_b['piece_center']), rot)
        draw.line([(bx, by), (bx + new_n[0]*ARROW_LEN, by + new_n[1]*ARROW_LEN)],
                  fill=(0, 255, 255, 200), width=2)

        band_new = corr_aligned_back[i] + new_n * (INNER_OFFSET + BAND_WIDTH // 2)
        bnx, bny = tc(band_new[0], band_new[1])
        draw.ellipse([bnx-3, bny-3, bnx+3, bny+3], fill=(0, 255, 255, 255))

ly = vp.canvas.size[1] - 18
draw.text((15, ly), "Red arrow=OLD tangent  Orange=OLD normal  Green=NEW tangent  Cyan=NEW normal  Yellow dot=NEW band_pos",
          fill=(200,200,200,255), font=fonts['tiny'])

out_path = os.path.join(CHECK_PATH, f'normal_fix_{PID_A}_{PID_B}.png')
vp.canvas.save(out_path)
print(f"\nSaved: {out_path}")
