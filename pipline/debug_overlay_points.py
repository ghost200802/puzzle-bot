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
    load_side_data, load_color_image, extract_inner_band,
    compute_pattern_ncc, compute_seam_color_diff,
    compute_texture_richness, compute_gradient_consistency,
    _resample_polyline, _compute_transform,
    INNER_OFFSET, BAND_WIDTH, N_SAMPLES, SAMPLE_RADIUS
)

from config import get_vector_path, get_color_path, get_check_path
from debug_utils import load_fonts, transform_point, compute_tangent_normal, CanvasViewport

DEDUPED_PATH = get_vector_path()
COLOR_PATH = get_color_path()
CHECK_PATH = get_check_path()

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

fonts = load_fonts(title_size=24, idx_size=12)
try:
    fonts['label'] = ImageFont.truetype("arialbd.ttf", 14)
    fonts['small'] = ImageFont.truetype("arial.ttf", 11)
except Exception:
    for k in ['label', 'small']:
        fonts[k] = ImageFont.load_default()

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)
color_a, mask_a = load_color_image(COLOR_PATH, PID_A)
color_b, mask_b = load_color_image(COLOR_PATH, PID_B)

verts_a = side_a['vertices']
verts_bf = side_b['vertices'][::-1].copy()

sample_a = _resample_polyline(verts_a, N_SAMPLES)
sample_bf = _resample_polyline(verts_bf, N_SAMPLES)

band_a = compute_tangent_normal(sample_a, side_a['piece_center'])
band_bf = compute_tangent_normal(sample_bf, side_b['piece_center'])

ba_colors, ba_gray = extract_inner_band(color_a, sample_a, side_a['piece_center'], mask_a)
bb_colors, bb_gray = extract_inner_band(color_b, sample_bf, side_b['piece_center'], mask_b)

n = min(len(ba_gray), len(bb_gray))
ncc = compute_pattern_ncc(ba_gray[:n], bb_gray[:n])
cd_mean, cd_median = compute_seam_color_diff(ba_colors[:n], bb_colors[:n])
grad = compute_gradient_consistency(ba_gray[:n], bb_gray[:n])

print(f"NCC (arc-length both): {ncc:.4f}")
print(f"Color diff: {cd_mean:.2f}")
print(f"Grad score: {grad}")

src_mid, tgt_mid, rot = _compute_transform(side_b['vertices'], verts_a)

from show_connectivity import _get_outline, _load_piece_data
piece_data = _load_piece_data(DEDUPED_PATH)

outline_a_full = _get_outline(piece_data[PID_A])
outline_b_raw = _get_outline(piece_data[PID_B])
outline_b_aligned = [transform_point(np.array(p), src_mid, tgt_mid, rot) for p in outline_b_raw]

sample_bf_aligned = [transform_point(p, src_mid, tgt_mid, rot) for p in sample_bf]
band_bf_aligned = []
for b in band_bf:
    ep = transform_point(b['pos'], src_mid, tgt_mid, rot)
    bp = transform_point(b['band_pos'], src_mid, tgt_mid, rot)
    band_bf_aligned.append({'edge_pos': ep, 'band_pos': bp})

all_pts = (outline_a_full + outline_b_aligned +
           [b['pos'].tolist() for b in band_a] +
           [b['band_pos'].tolist() for b in band_a] +
           [b['edge_pos'].tolist() for b in band_bf_aligned] +
           [b['band_pos'].tolist() for b in band_bf_aligned])

vp = CanvasViewport(all_pts, canvas_w=1400, target_h=800, margin=60, top_offset=40)
tc = vp.tc
draw = vp.draw

draw.text((15, 8),
          f"Overlay: Piece {PID_A}[{SI_A}] + Piece {PID_B}[{SI_B}] \u2014 Arc-length sampling both edges  NCC={ncc:.4f}",
          fill=(255, 255, 255, 255), font=fonts['title'])

for outline, color, width in [
    (outline_a_full, (100, 150, 255, 180), 2),
    (outline_b_aligned, (255, 100, 100, 180), 2),
]:
    canvas_pts = [tc(x, y) for x, y in outline]
    if len(canvas_pts) >= 3:
        draw.polygon(canvas_pts, fill=None, outline=color, width=width)

side_pts = [tc(v[0], v[1]) for v in verts_a[::max(1, len(verts_a)//50)]]
if len(side_pts) >= 2:
    draw.line(side_pts, fill=(255, 200, 0, 200), width=2)

bf_edge_pts = [tc(v[0], v[1]) for v in sample_bf_aligned]
if len(bf_edge_pts) >= 2:
    draw.line(bf_edge_pts, fill=(255, 80, 80, 150), width=1)

for i in range(N_SAMPLES):
    ep = tc(band_a[i]['pos'][0], band_a[i]['pos'][1])
    bp = tc(band_a[i]['band_pos'][0], band_a[i]['band_pos'][1])

    draw.ellipse([ep[0]-5, ep[1]-5, ep[0]+5, ep[1]+5], fill=(50, 140, 255, 255), outline=(255,255,255,180))
    draw.ellipse([bp[0]-3, bp[1]-3, bp[0]+3, bp[1]+3], fill=(50, 140, 255, 180))
    draw.text((ep[0]+6, ep[1]-8), str(i), fill=(0, 255, 100, 255), font=fonts['idx'])

for i in range(N_SAMPLES):
    ep = tc(band_bf_aligned[i]['edge_pos'][0], band_bf_aligned[i]['edge_pos'][1])
    bp = tc(band_bf_aligned[i]['band_pos'][0], band_bf_aligned[i]['band_pos'][1])

    draw.ellipse([ep[0]-5, ep[1]-5, ep[0]+5, ep[1]+5], fill=(255, 80, 80, 255), outline=(255,255,255,180))
    draw.ellipse([bp[0]-3, bp[1]-3, bp[0]+3, bp[1]+3], fill=(255, 80, 80, 180))
    draw.text((ep[0]-22, ep[1]+4), str(i), fill=(0, 255, 100, 255), font=fonts['idx'])

for i in range(N_SAMPLES):
    a_ep = tc(band_a[i]['pos'][0], band_a[i]['pos'][1])
    b_ep = tc(band_bf_aligned[i]['edge_pos'][0], band_bf_aligned[i]['edge_pos'][1])
    draw.line([a_ep, b_ep], fill=(255, 255, 0, 80), width=1)

ly = vp.canvas.size[1] - 18
draw.ellipse([15, ly-4, 23, ly+4], fill=(50, 140, 255, 255))
draw.text((28, ly-7), f"Piece {PID_A} (arc-length)", fill=(100, 180, 255, 255), font=fonts['small'])

draw.ellipse([220, ly-4, 228, ly+4], fill=(255, 80, 80, 255))
draw.text((233, ly-7), f"Piece {PID_B} flipped (arc-length)", fill=(255, 130, 130, 255), font=fonts['small'])

draw.text((470, ly-7), "Green = index number", fill=(0, 255, 100, 255), font=fonts['small'])
draw.text((650, ly-7), f"NCC={ncc:.4f}  ColorDiff={cd_mean:.1f}", fill=(200, 200, 200, 255), font=fonts['small'])

out_path = os.path.join(CHECK_PATH, f'overlay_arclength_{PID_A}_{PID_B}.png')
vp.canvas.save(out_path)
print(f"Saved: {out_path}")

dists = []
for i in range(N_SAMPLES):
    d = np.linalg.norm(band_a[i]['pos'] - band_bf_aligned[i]['edge_pos'])
    dists.append((i, d))
dists.sort(key=lambda x: x[1])
print("\nPoint distances (sorted):")
for idx, d in dists:
    marker = "OK" if d < 20 else ("~" if d < 50 else "!!")
    print(f"  [{idx:2d}] dist={d:.1f} {marker}")
print(f"\n  <20px: {sum(1 for _,d in dists if d<20)}/{len(dists)}")
print(f"  <50px: {sum(1 for _,d in dists if d<50)}/{len(dists)}")
print(f"  Mean: {np.mean([d for _,d in dists]):.1f}")
