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
    load_side_data, _resample_polyline,
    _find_corresponding_points_on_edge, _compute_transform,
    _apply_transform, _apply_inverse_transform,
    INNER_OFFSET, BAND_WIDTH, N_SAMPLES
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
verts_b = side_b['vertices']
verts_bf = verts_b[::-1].copy()

sample_a = _resample_polyline(verts_a, N_SAMPLES)

src_mid, tgt_mid, rot = _compute_transform(verts_b, verts_a)
verts_bf_aligned = _apply_transform(verts_bf, src_mid, tgt_mid, rot)
corr_aligned = _find_corresponding_points_on_edge(sample_a, verts_bf_aligned)
corr_orig = _apply_inverse_transform(corr_aligned, src_mid, tgt_mid, rot)
sample_a_in_b = _apply_inverse_transform(sample_a, src_mid, tgt_mid, rot)

# NEW tangent for A: A[i] -> corr_bf_aligned[i]
# NEW tangent for B: corr_orig[i] -> sample_a_in_b[i]
tangent_a_new = np.zeros_like(sample_a)
tangent_b_new = np.zeros_like(corr_orig)
for i in range(N_SAMPLES):
    tangent_a_new[i] = corr_aligned[i] - sample_a[i]
    tl = np.linalg.norm(tangent_a_new[i])
    if tl > 1e-6:
        tangent_a_new[i] /= tl

    tangent_b_new[i] = sample_a_in_b[i] - corr_orig[i]
    tl = np.linalg.norm(tangent_b_new[i])
    if tl > 1e-6:
        tangent_b_new[i] /= tl

# Compute normals
def normal_from_tangent(tangent, pos, piece_center):
    normal = np.array([-tangent[1], tangent[0]])
    to_c = piece_center - pos
    if np.dot(normal, to_c) < 0:
        normal = -normal
    return normal

normals_a = np.array([normal_from_tangent(tangent_a_new[i], sample_a[i], side_a['piece_center']) for i in range(N_SAMPLES)])
normals_b_orig = np.array([normal_from_tangent(tangent_b_new[i], corr_orig[i], side_b['piece_center']) for i in range(N_SAMPLES)])

# Transform B's normals to A's system for visualization
def tf_vec(v, r):
    cos_r, sin_r = math.cos(r), math.sin(r)
    return np.array([v[0]*cos_r - v[1]*sin_r, v[0]*sin_r + v[1]*cos_r])

normals_b_aligned = np.array([tf_vec(n, rot) for n in normals_b_orig])
tangent_b_aligned = np.array([tf_vec(t, rot) for t in tangent_b_new])

# Transform for overlay
def tf_pt(v, sm, tm, r):
    cos_r, sin_r = math.cos(r), math.sin(r)
    dx = v[0] - sm[0]
    dy = v[1] - sm[1]
    return np.array([dx*cos_r - dy*sin_r + tm[0], dx*sin_r + dy*cos_r + tm[1]])

piece_data = _load_piece_data(DEDUPED_PATH)
outline_a = _get_outline(piece_data[PID_A])
outline_b = [tf_pt(np.array(p), src_mid, tgt_mid, rot) for p in _get_outline(piece_data[PID_B])]
corr_aligned_back = _apply_transform(corr_orig, src_mid, tgt_mid, rot)

all_pts = outline_a + outline_b + [p.tolist() for p in sample_a] + [p.tolist() for p in corr_aligned_back]
min_x = min(p[0] for p in all_pts)
max_x = max(p[0] for p in all_pts)
min_y = min(p[1] for p in all_pts)
max_y = max(p[1] for p in all_pts)
data_w = max_x - min_x
data_h = max_y - min_y

canvas_w = 1600
margin = 80
scale = min((canvas_w - 2*margin) / data_w, (900 - 2*margin) / data_h)
canvas_h = int(data_h * scale) + 2*margin + 80

def tc(x, y):
    return ((x - min_x) * scale + margin, (y - min_y) * scale + margin + 50)

canvas = Image.new('RGBA', (canvas_w, canvas_h), (30, 30, 30, 255))
draw = ImageDraw.Draw(canvas)

draw.text((15, 8),
          f"Normal via opposite points: tangent = line to opposite point, normal = perpendicular toward center",
          fill=(255, 255, 255, 255), font=fonts['title'])

for outline, color in [(outline_a, (80,120,200,160)), (outline_b, (200,80,80,160))]:
    pts = [tc(x, y) for x, y in outline]
    if len(pts) >= 3:
        draw.polygon(pts, fill=None, outline=color, width=2)

a_edge = [tc(v[0], v[1]) for v in sample_a]
draw.line(a_edge, fill=(100, 150, 255, 100), width=1)
b_edge = [tc(p[0], p[1]) for p in corr_aligned_back]
draw.line(b_edge, fill=(255, 100, 100, 100), width=1)

ARROW = 35

for i in range(N_SAMPLES):
    is_problem = i in [8, 9, 10, 16, 17, 18]
    r = 6 if is_problem else 3

    # A point
    ax, ay = tc(sample_a[i][0], sample_a[i][1])
    draw.ellipse([ax-r, ay-r, ax+r, ay+r], fill=(50, 140, 255, 255), outline=(255,255,255,180))
    draw.text((ax+7, ay-9), str(i), fill=(0, 255, 100, 255), font=fonts['idx'])

    # A tangent (cyan) and normal (green)
    draw.line([(ax, ay), (ax + tangent_a_new[i][0]*ARROW, ay + tangent_a_new[i][1]*ARROW)],
              fill=(0, 255, 255, 180), width=2)
    draw.line([(ax, ay), (ax + normals_a[i][0]*ARROW, ay + normals_a[i][1]*ARROW)],
              fill=(0, 255, 0, 180), width=2)

    # A band position
    if is_problem:
        band_a_pos = sample_a[i] + normals_a[i] * (INNER_OFFSET + BAND_WIDTH // 2)
        bax, bay = tc(band_a_pos[0], band_a_pos[1])
        draw.ellipse([bax-3, bay-3, bax+3, bay+3], fill=(0, 255, 0, 255))

    # B corr point
    bx, by = tc(corr_aligned_back[i][0], corr_aligned_back[i][1])
    draw.ellipse([bx-r, by-r, bx+r, by+r], fill=(255, 80, 80, 255), outline=(255,255,255,180))

    # B tangent (orange) and normal (yellow)
    draw.line([(bx, by), (bx + tangent_b_aligned[i][0]*ARROW, by + tangent_b_aligned[i][1]*ARROW)],
              fill=(255, 165, 0, 180), width=2)
    draw.line([(bx, by), (bx + normals_b_aligned[i][0]*ARROW, by + normals_b_aligned[i][1]*ARROW)],
              fill=(255, 255, 0, 180), width=2)

    # B band position
    if is_problem:
        band_b_pos = corr_aligned_back[i] + normals_b_aligned[i] * (INNER_OFFSET + BAND_WIDTH // 2)
        bbx, bby = tc(band_b_pos[0], band_b_pos[1])
        draw.ellipse([bbx-3, bby-3, bbx+3, bby+3], fill=(255, 255, 0, 255))

ly = canvas_h - 18
draw.text((15, ly), "A: cyan=tangent green=normal  |  B: orange=tangent yellow=normal  |  Problem points highlighted",
          fill=(200,200,200,255), font=fonts['tiny'])

out_path = os.path.join(CHECK_PATH, f'normal_opposite_{PID_A}_{PID_B}.png')
canvas.save(out_path)
print(f"Saved: {out_path}")

print("\nTangent angles (from opposite point direction):")
for i in range(N_SAMPLES):
    a_angle = math.degrees(math.atan2(tangent_a_new[i][1], tangent_a_new[i][0]))
    b_angle = math.degrees(math.atan2(tangent_b_aligned[i][1], tangent_b_aligned[i][0]))
    marker = " <<<" if i in [8,9,10,16,17,18] else ""
    print(f"  [{i:2d}] A_tangent={a_angle:+7.1f}°  B_tangent={b_angle:+7.1f}°{marker}")
