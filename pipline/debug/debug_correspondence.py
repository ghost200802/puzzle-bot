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
    load_side_data, _resample_polyline, _resample_by_chord, N_SAMPLES
)

from config import get_vector_path, get_check_path
from pipline.debug_utils import load_fonts

DEDUPED_PATH = get_vector_path()
CHECK_PATH = get_check_path()

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)

verts_a = side_a['vertices']
verts_b = side_b['vertices']
verts_bf = verts_b[::-1].copy()

def _side_angle(verts):
    p1, p2 = verts[0], verts[-1]
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def _side_mid(verts):
    p1, p2 = verts[0], verts[-1]
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

def _transform_edge(src_verts, tgt_verts):
    src_mid = np.array(_side_mid(src_verts))
    tgt_mid = np.array(_side_mid(tgt_verts))
    src_theta = _side_angle(src_verts)
    tgt_theta = _side_angle(tgt_verts)
    rot = tgt_theta + math.pi - src_theta
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    transformed = []
    for v in src_verts:
        dx = v[0] - src_mid[0]
        dy = v[1] - src_mid[1]
        fx = dx * cos_r - dy * sin_r + tgt_mid[0]
        fy = dx * sin_r + dy * cos_r + tgt_mid[1]
        transformed.append([fx, fy])
    return np.array(transformed)

transformed_bf = _transform_edge(verts_bf, verts_a)

arc_a = _resample_polyline(verts_a, N_SAMPLES)
arc_bf = _resample_polyline(verts_bf, N_SAMPLES)
arc_bf_aligned = _transform_edge(arc_bf, verts_a)

chord_a = _resample_by_chord(verts_a, N_SAMPLES)
chord_bf = _resample_by_chord(verts_bf, N_SAMPLES)
chord_bf_aligned = _transform_edge(chord_bf, verts_a)

print("Arc-length sampling distances (after alignment):")
arc_dists = [np.linalg.norm(arc_a[i] - arc_bf_aligned[i]) for i in range(N_SAMPLES)]
for i in range(N_SAMPLES):
    print(f"  [{i:2d}] dist={arc_dists[i]:.1f}")
print(f"  Mean: {np.mean(arc_dists):.1f}, Median: {np.median(arc_dists):.1f}, Max: {np.max(arc_dists):.1f}")

print(f"\nChord-based sampling distances (after alignment):")
chord_dists = [np.linalg.norm(chord_a[i] - chord_bf_aligned[i]) for i in range(N_SAMPLES)]
for i in range(N_SAMPLES):
    print(f"  [{i:2d}] dist={chord_dists[i]:.1f}")
print(f"  Mean: {np.mean(chord_dists):.1f}, Median: {np.median(chord_dists):.1f}, Max: {np.max(chord_dists):.1f}")

fonts = load_fonts(title_size=20, idx_size=11)
font = fonts['title']
small_font = fonts['idx']
title_font = fonts['title']

for method, pts_a, pts_bf_al, dists in [
    ("arc-length", arc_a, arc_bf_aligned, arc_dists),
    ("chord-based", chord_a, chord_bf_aligned, chord_dists),
]:
    all_pts = list(pts_a) + list(pts_bf_al) + list(verts_a) + list(transformed_bf)
    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_y = max(p[1] for p in all_pts)
    data_w = max_x - min_x
    data_h = max_y - min_y
    margin = 40
    cw = 1200
    ch = 600
    scale = min((cw - 2*margin) / data_w, (ch - 2*margin) / data_h)

    img = Image.new('RGBA', (cw, ch + 40), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 5), f"{method}: Correspondence distances (mean={np.mean(dists):.1f})", fill=(0,0,0), font=title_font)

    def tc(x, y):
        return ((x - min_x) * scale + margin, (y - min_y) * scale + margin + 30)

    edge_a_pts = [tc(v[0], v[1]) for v in verts_a[::5]]
    edge_b_pts = [tc(v[0], v[1]) for v in transformed_bf[::5]]
    if len(edge_a_pts) >= 2:
        draw.line(edge_a_pts, fill=(0, 0, 200, 100), width=2)
    if len(edge_b_pts) >= 2:
        draw.line(edge_b_pts, fill=(200, 0, 0, 100), width=2)

    for i in range(N_SAMPLES):
        pa = tc(pts_a[i][0], pts_a[i][1])
        pb = tc(pts_bf_al[i][0], pts_bf_al[i][1])
        d = dists[i]

        if d < 20:
            line_color = (0, 180, 0, 200)
        elif d < 50:
            line_color = (255, 165, 0, 200)
        else:
            line_color = (255, 0, 0, 200)

        draw.line([pa, pb], fill=line_color, width=1)

        r_a = 5
        draw.ellipse([pa[0]-r_a, pa[1]-r_a, pa[0]+r_a, pa[1]+r_a],
                     fill=(0, 0, 200, 255), outline=(0, 0, 0, 255))
        draw.text((pa[0]+r_a+1, pa[1]-6), str(i), fill=(0, 0, 150), font=small_font)

        r_b = 4
        draw.ellipse([pb[0]-r_b, pb[1]-r_b, pb[0]+r_b, pb[1]+r_b],
                     fill=(200, 0, 0, 255), outline=(0, 0, 0, 255))

    ly = ch + 10
    draw.text((10, ly), "Blue=A  Red=B(flipped,aligned)", fill=(0,0,0), font=font)
    draw.text((350, ly), "Green=<20px  Orange=<50px  Red=>50px", fill=(0,0,0), font=font)

    out_path = os.path.join(CHECK_PATH, f'correspondence_{method}_{PID_A}_{PID_B}.png')
    img.save(out_path)
    print(f"\nSaved: {out_path}")
