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
    load_side_data, _resample_polyline, INNER_OFFSET, BAND_WIDTH, N_SAMPLES
)
from show_connectivity import _get_outline, _load_piece_data

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '3_vector')
CHECK_PATH = os.path.join(OUTPUT_DIR, 'check')

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

try:
    fonts = {
        'title': ImageFont.truetype("arialbd.ttf", 22),
        'idx': ImageFont.truetype("arialbd.ttf", 11),
        'tiny': ImageFont.truetype("arial.ttf", 9),
    }
except:
    default = ImageFont.load_default()
    fonts = {k: default for k in ['title', 'idx', 'tiny']}

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)

verts_a = side_a['vertices']
verts_bf = side_b['vertices'][::-1].copy()

sample_a = _resample_polyline(verts_a, N_SAMPLES)
sample_bf = _resample_polyline(verts_bf, N_SAMPLES)

def compute_tangent_normal(sample_positions, piece_center):
    results = []
    for i in range(len(sample_positions)):
        pos = sample_positions[i]
        if i == 0:
            tangent = sample_positions[1] - sample_positions[0]
        elif i == len(sample_positions) - 1:
            tangent = sample_positions[-1] - sample_positions[-2]
        else:
            tangent = sample_positions[i + 1] - sample_positions[i - 1]
        tlen = np.linalg.norm(tangent)
        if tlen < 1e-6:
            results.append({'pos': pos, 'tangent': np.array([0,0]), 'normal': np.array([0,0]), 'band_pos': pos})
            continue
        tangent = tangent / tlen
        normal = np.array([-tangent[1], tangent[0]])
        to_c = piece_center - pos
        if np.dot(normal, to_c) < 0:
            normal = -normal
        band_pos = pos + normal * (INNER_OFFSET + BAND_WIDTH // 2)
        results.append({'pos': pos, 'tangent': tangent, 'normal': normal, 'band_pos': band_pos})
    return results

info_a = compute_tangent_normal(sample_a, side_a['piece_center'])
info_bf = compute_tangent_normal(sample_bf, side_b['piece_center'])

# Print debug info for points of interest
for label, info_list in [("A", info_a), ("B_flipped", info_bf)]:
    print(f"\n{label} tangent/normal at key points:")
    for i in range(N_SAMPLES):
        t = info_list[i]['tangent']
        n = info_list[i]['normal']
        t_angle = math.degrees(math.atan2(t[1], t[0]))
        n_angle = math.degrees(math.atan2(n[1], n[0]))
        print(f"  [{i:2d}] tangent=({t[0]:+.3f},{t[1]:+.3f}) angle={t_angle:+7.1f}°  "
              f"normal=({n[0]:+.3f},{n[1]:+.3f}) angle={n_angle:+7.1f}°")

# Transform for overlay
def get_transform(src_verts, tgt_verts):
    src_mid = (src_verts[0] + src_verts[-1]) / 2.0
    tgt_mid = (tgt_verts[0] + tgt_verts[-1]) / 2.0
    src_theta = math.atan2(src_verts[-1][1] - src_verts[0][1], src_verts[-1][0] - src_verts[0][0])
    tgt_theta = math.atan2(tgt_verts[-1][1] - tgt_verts[0][1], tgt_verts[-1][0] - tgt_verts[0][0])
    rot = tgt_theta + math.pi - src_theta
    return src_mid, tgt_mid, rot

def tf_pt(v, src_mid, tgt_mid, rot):
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    dx = v[0] - src_mid[0]
    dy = v[1] - src_mid[1]
    return np.array([dx*cos_r - dy*sin_r + tgt_mid[0], dx*sin_r + dy*cos_r + tgt_mid[1]])

def tf_vec(v, rot):
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    return np.array([v[0]*cos_r - v[1]*sin_r, v[0]*sin_r + v[1]*cos_r])

src_mid, tgt_mid, rot = get_transform(side_b['vertices'], verts_a)

piece_data = _load_piece_data(DEDUPED_PATH)
outline_a = _get_outline(piece_data[PID_A])
outline_b_raw = _get_outline(piece_data[PID_B])
outline_b = [tf_pt(np.array(p), src_mid, tgt_mid, rot) for p in outline_b_raw]

# Transform B's info to A's coordinate system
info_bf_aligned = []
for info in info_bf:
    info_bf_aligned.append({
        'pos': tf_pt(info['pos'], src_mid, tgt_mid, rot),
        'tangent': tf_vec(info['tangent'], rot),
        'normal': tf_vec(info['normal'], rot),
        'band_pos': tf_pt(info['band_pos'], src_mid, tgt_mid, rot),
    })

# Bounding box
all_pts = (outline_a + outline_b +
           [info['pos'].tolist() for info in info_a] +
           [info['band_pos'].tolist() for info in info_a] +
           [info['pos'].tolist() for info in info_bf_aligned] +
           [info['band_pos'].tolist() for info in info_bf_aligned])
min_x = min(p[0] for p in all_pts)
max_x = max(p[0] for p in all_pts)
min_y = min(p[1] for p in all_pts)
max_y = max(p[1] for p in all_pts)
data_w = max_x - min_x
data_h = max_y - min_y

canvas_w = 1600
margin = 80
scale = min((canvas_w - 2*margin) / data_w, (900 - 2*margin) / data_h)
canvas_h = int(data_h * scale) + 2*margin + 50

def tc(x, y):
    return ((x - min_x) * scale + margin, (y - min_y) * scale + margin + 40)

ARROW_LEN = 25

canvas = Image.new('RGBA', (canvas_w, canvas_h), (30, 30, 30, 255))
draw = ImageDraw.Draw(canvas)

draw.text((15, 8),
          f"Direction Debug: Piece {PID_A}[{SI_A}] + Piece {PID_B}[{SI_B}] — Tangent(cyan) Normal(green) Band(yellow)",
          fill=(255, 255, 255, 255), font=fonts['title'])

# Draw outlines
for outline, color, w in [
    (outline_a, (80, 120, 200, 160), 2),
    (outline_b, (200, 80, 80, 160), 2),
]:
    pts = [tc(x, y) for x, y in outline]
    if len(pts) >= 3:
        draw.polygon(pts, fill=None, outline=color, width=w)

# Draw A's edge line
a_edge = [tc(v[0], v[1]) for v in sample_a]
if len(a_edge) >= 2:
    draw.line(a_edge, fill=(100, 150, 255, 120), width=1)

# Draw B's edge line (aligned)
b_edge = [tc(info['pos'][0], info['pos'][1]) for info in info_bf_aligned]
if len(b_edge) >= 2:
    draw.line(b_edge, fill=(255, 100, 100, 120), width=1)

# Draw points with direction arrows
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

        # Tangent arrow (cyan)
        tx, ty = info['tangent'][0], info['tangent'][1]
        ex, ey = px + tx * ARROW_LEN, py + ty * ARROW_LEN
        draw.line([(px, py), (ex, ey)], fill=(0, 255, 255, 200), width=2)

        # Normal arrow (green)
        nx, ny = info['normal'][0], info['normal'][1]
        nex, ney = px + nx * ARROW_LEN, py + ny * ARROW_LEN
        draw.line([(px, py), (nex, ney)], fill=(0, 255, 0, 200), width=2)

        # Band position (yellow dot)
        if is_problem:
            bx, by = tc(info['band_pos'][0], info['band_pos'][1])
            draw.ellipse([bx-3, by-3, bx+3, by+3], fill=(255, 255, 0, 255), outline=(255,255,255,200))
            draw.line([(px, py), (bx, by)], fill=(255, 255, 0, 150), width=1)

# Legend
ly = canvas_h - 16
draw.text((15, ly), "Cyan=tangent  Green=normal  Yellow=band_pos (problem points only)", fill=(200,200,200,255), font=fonts['tiny'])
draw.text((500, ly), "Yellow numbers = problem points (8-10, 16-18)", fill=(255,255,0,255), font=fonts['tiny'])

out_path = os.path.join(CHECK_PATH, f'direction_debug_{PID_A}_{PID_B}.png')
canvas.save(out_path)
print(f"\nSaved: {out_path}")
