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
    load_side_data, _resample_polyline, INNER_OFFSET, BAND_WIDTH, N_SAMPLES
)
from pipline.show_connectivity import _get_outline, _load_piece_data

from config import get_vector_path, get_check_path
from debug_utils import load_fonts, CanvasViewport, transform_point, rotate_vector, compute_tangent_normal

DEDUPED_PATH = get_vector_path()
CHECK_PATH = get_check_path()

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

fonts = load_fonts(title_size=22, idx_size=11, tiny_size=9)

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)

verts_a = side_a['vertices']
verts_bf = side_b['vertices'][::-1].copy()

sample_a = _resample_polyline(verts_a, N_SAMPLES)
sample_bf = _resample_polyline(verts_bf, N_SAMPLES)

info_a = compute_tangent_normal(sample_a, side_a['piece_center'])
info_bf = compute_tangent_normal(sample_bf, side_b['piece_center'])

for label, info_list in [("A", info_a), ("B_flipped", info_bf)]:
    print(f"\n{label} tangent/normal at key points:")
    for i in range(N_SAMPLES):
        t = info_list[i]['tangent']
        n = info_list[i]['normal']
        t_angle = math.degrees(math.atan2(t[1], t[0]))
        n_angle = math.degrees(math.atan2(n[1], n[0]))
        print(f"  [{i:2d}] tangent=({t[0]:+.3f},{t[1]:+.3f}) angle={t_angle:+7.1f}\u00b0  "
              f"normal=({n[0]:+.3f},{n[1]:+.3f}) angle={n_angle:+7.1f}\u00b0")

def get_transform(src_verts, tgt_verts):
    src_mid = (src_verts[0] + src_verts[-1]) / 2.0
    tgt_mid = (tgt_verts[0] + tgt_verts[-1]) / 2.0
    src_theta = math.atan2(src_verts[-1][1] - src_verts[0][1], src_verts[-1][0] - src_verts[0][0])
    tgt_theta = math.atan2(tgt_verts[-1][1] - tgt_verts[0][1], tgt_verts[-1][0] - tgt_verts[0][0])
    rot = tgt_theta + math.pi - src_theta
    return src_mid, tgt_mid, rot

src_mid, tgt_mid, rot = get_transform(side_b['vertices'], verts_a)

piece_data = _load_piece_data(DEDUPED_PATH)
outline_a = _get_outline(piece_data[PID_A])
outline_b_raw = _get_outline(piece_data[PID_B])
outline_b = [transform_point(np.array(p), src_mid, tgt_mid, rot) for p in outline_b_raw]

info_bf_aligned = []
for info in info_bf:
    info_bf_aligned.append({
        'pos': transform_point(info['pos'], src_mid, tgt_mid, rot),
        'tangent': rotate_vector(info['tangent'], rot),
        'normal': rotate_vector(info['normal'], rot),
        'band_pos': transform_point(info['band_pos'], src_mid, tgt_mid, rot),
    })

all_pts = (outline_a + outline_b +
           [info['pos'].tolist() for info in info_a] +
           [info['band_pos'].tolist() for info in info_a] +
           [info['pos'].tolist() for info in info_bf_aligned] +
           [info['band_pos'].tolist() for info in info_bf_aligned])

vp = CanvasViewport(all_pts, canvas_w=1600, target_h=900, margin=80, top_offset=40)
tc = vp.tc
draw = vp.draw

draw.text((15, 8),
          f"Direction Debug: Piece {PID_A}[{SI_A}] + Piece {PID_B}[{SI_B}] \u2014 Tangent(cyan) Normal(green) Band(yellow)",
          fill=(255, 255, 255, 255), font=fonts['title'])

for outline, color, w in [
    (outline_a, (80, 120, 200, 160), 2),
    (outline_b, (200, 80, 80, 160), 2),
]:
    pts = [tc(x, y) for x, y in outline]
    if len(pts) >= 3:
        draw.polygon(pts, fill=None, outline=color, width=w)

a_edge = [tc(v[0], v[1]) for v in sample_a]
if len(a_edge) >= 2:
    draw.line(a_edge, fill=(100, 150, 255, 120), width=1)

b_edge = [tc(info['pos'][0], info['pos'][1]) for info in info_bf_aligned]
if len(b_edge) >= 2:
    draw.line(b_edge, fill=(255, 100, 100, 120), width=1)

ARROW_LEN = 25

for label, info_list, dot_color, num_offset in [
    ("A", info_a, (50, 140, 255, 255), (7, -9)),
    ("B", info_bf_aligned, (255, 80, 80, 255), (-24, 5)),
]:
    for i in range(N_SAMPLES):
        info = info_list[i]
        px, py = tc(info['pos'][0], info['pos'][1])

        is_problem = i in [8, 9, 10, 16, 17, 18]
        r = 6 if is_problem else 4

        draw.ellipse([px-r, py-r, px+r, py+r], fill=dot_color, outline=(255,255,255,200) if is_problem else (255,255,255,100))

        num_color = (255, 255, 0, 255) if is_problem else (0, 255, 100, 255)
        draw.text((px + num_offset[0], py + num_offset[1]), str(i), fill=num_color, font=fonts['idx'])

        tx, ty = info['tangent'][0], info['tangent'][1]
        ex, ey = px + tx * ARROW_LEN, py + ty * ARROW_LEN
        draw.line([(px, py), (ex, ey)], fill=(0, 255, 255, 200), width=2)

        nx, ny = info['normal'][0], info['normal'][1]
        nex, ney = px + nx * ARROW_LEN, py + ny * ARROW_LEN
        draw.line([(px, py), (nex, ney)], fill=(0, 255, 0, 200), width=2)

        if is_problem:
            bx, by = tc(info['band_pos'][0], info['band_pos'][1])
            draw.ellipse([bx-3, by-3, bx+3, by+3], fill=(255, 255, 0, 255), outline=(255,255,255,200))
            draw.line([(px, py), (bx, by)], fill=(255, 255, 0, 150), width=1)

ly = vp.canvas.size[1] - 16
draw.text((15, ly), "Cyan=tangent  Green=normal  Yellow=band_pos (problem points only)", fill=(200,200,200,255), font=fonts['tiny'])
draw.text((500, ly), "Yellow numbers = problem points (8-10, 16-18)", fill=(255,255,0,255), font=fonts['tiny'])

out_path = os.path.join(CHECK_PATH, f'direction_debug_{PID_A}_{PID_B}.png')
vp.canvas.save(out_path)
print(f"\nSaved: {out_path}")
