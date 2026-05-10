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
    _apply_transform, _apply_inverse_transform,
    _edge_tangent_at,
    INNER_OFFSET, BAND_WIDTH, N_SAMPLES
)
from pipline.show_connectivity import _get_outline, _load_piece_data

from pipline.config import get_vector_path, get_check_path
from pipline.debug_utils import load_fonts, transform_point, rotate_vector, CanvasViewport

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

def compute_normals(points, piece_center, edge_vertices=None):
    n = len(points)
    normals = np.zeros_like(points)
    for i in range(n):
        if edge_vertices is not None:
            tangent = _edge_tangent_at(edge_vertices, points[i])
        else:
            tangents = []
            if i > 0:
                d = points[i] - points[i - 1]
                dl = np.linalg.norm(d)
                if dl > 1e-6:
                    tangents.append(d / dl)
            if i < n - 1:
                d = points[i + 1] - points[i]
                dl = np.linalg.norm(d)
                if dl > 1e-6:
                    tangents.append(d / dl)
            if not tangents:
                normals[i] = np.array([0.0, 0.0])
                continue
            tangent = np.mean(tangents, axis=0)
        tl = np.linalg.norm(tangent)
        if tl < 1e-6:
            normals[i] = np.array([0.0, 0.0])
            continue
        tangent = tangent / tl
        normal = np.array([-tangent[1], tangent[0]])
        to_c = piece_center - points[i]
        if np.dot(normal, to_c) < 0:
            normal = -normal
        normals[i] = normal
    return normals

normals_a = compute_normals(sample_a, side_a['piece_center'])
normals_b_orig = compute_normals(corr_orig, side_b['piece_center'], edge_vertices=verts_bf)

normals_b_aligned = np.array([rotate_vector(n, rot) for n in normals_b_orig])

piece_data = _load_piece_data(DEDUPED_PATH)
outline_a = _get_outline(piece_data[PID_A])
outline_b = [transform_point(np.array(p), src_mid, tgt_mid, rot) for p in _get_outline(piece_data[PID_B])]
corr_aligned_back = _apply_transform(corr_orig, src_mid, tgt_mid, rot)

all_pts = outline_a + outline_b + [p.tolist() for p in sample_a] + [p.tolist() for p in corr_aligned_back]
vp = CanvasViewport(all_pts, canvas_w=1600, target_h=900, margin=80, top_offset=50)
tc = vp.tc
draw = vp.draw

draw.text((15, 8),
          f"Normal from own adjacent points (smoothed)  |  Blue=A normal  Red=B normal  Green=index",
          fill=(255, 255, 255, 255), font=fonts['title'])

for outline, color in [(outline_a, (80,120,200,160)), (outline_b, (200,80,80,160))]:
    pts = [tc(x, y) for x, y in outline]
    if len(pts) >= 3:
        draw.polygon(pts, fill=None, outline=color, width=2)

a_edge = [tc(v[0], v[1]) for v in sample_a]
draw.line(a_edge, fill=(100, 150, 255, 100), width=1)
b_edge = [tc(p[0], p[1]) for p in corr_aligned_back]
draw.line(b_edge, fill=(255, 100, 100, 100), width=1)

ARROW = 40

for i in range(N_SAMPLES):
    is_problem = i in [8, 9, 10, 16, 17, 18]
    r = 6 if is_problem else 3

    ax, ay = tc(sample_a[i][0], sample_a[i][1])
    draw.ellipse([ax-r, ay-r, ax+r, ay+r], fill=(50, 140, 255, 255), outline=(255,255,255,180))
    draw.text((ax+7, ay-9), str(i), fill=(0, 255, 100, 255), font=fonts['idx'])

    draw.line([(ax, ay), (ax + normals_a[i][0]*ARROW, ay + normals_a[i][1]*ARROW)],
              fill=(50, 140, 255, 220), width=2)

    band_a = sample_a[i] + normals_a[i] * (INNER_OFFSET + BAND_WIDTH // 2)
    bax, bay = tc(band_a[0], band_a[1])
    draw.ellipse([bax-3, bay-3, bax+3, bay+3], fill=(80, 180, 255, 255))

    bx, by = tc(corr_aligned_back[i][0], corr_aligned_back[i][1])
    draw.ellipse([bx-r, by-r, bx+r, by+r], fill=(255, 80, 80, 255), outline=(255,255,255,180))

    draw.line([(bx, by), (bx + normals_b_aligned[i][0]*ARROW, by + normals_b_aligned[i][1]*ARROW)],
              fill=(255, 80, 80, 220), width=2)

    band_b = corr_aligned_back[i] + normals_b_aligned[i] * (INNER_OFFSET + BAND_WIDTH // 2)
    bbx, bby = tc(band_b[0], band_b[1])
    draw.ellipse([bbx-3, bby-3, bbx+3, bby+3], fill=(255, 140, 140, 255))

ly = vp.canvas.size[1] - 18
draw.text((15, ly), "Blue arrows=A normals  Red arrows=B normals  Dots=band positions  Green=index",
          fill=(200,200,200,255), font=fonts['tiny'])

out_path = os.path.join(CHECK_PATH, f'normal_own_adj_{PID_A}_{PID_B}.png')
vp.canvas.save(out_path)
print(f"Saved: {out_path}")
