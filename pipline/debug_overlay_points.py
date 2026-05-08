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
    _resample_polyline, INNER_OFFSET, BAND_WIDTH, N_SAMPLES, SAMPLE_RADIUS
)

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '3_vector')
COLOR_PATH = os.path.join(OUTPUT_DIR, '2_piece_colors')
CHECK_PATH = os.path.join(OUTPUT_DIR, 'check')

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

try:
    fonts = {
        'title': ImageFont.truetype("arialbd.ttf", 24),
        'label': ImageFont.truetype("arialbd.ttf", 14),
        'small': ImageFont.truetype("arial.ttf", 11),
        'idx': ImageFont.truetype("arialbd.ttf", 12),
    }
except:
    default = ImageFont.load_default()
    fonts = {k: default for k in ['title', 'label', 'small', 'idx']}

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)
color_a, mask_a = load_color_image(COLOR_PATH, PID_A)
color_b, mask_b = load_color_image(COLOR_PATH, PID_B)

verts_a = side_a['vertices']
verts_bf = side_b['vertices'][::-1].copy()

sample_a = _resample_polyline(verts_a, N_SAMPLES)
sample_bf = _resample_polyline(verts_bf, N_SAMPLES)

def compute_band_pts(sample_positions, piece_center):
    center = piece_center
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
            results.append({'edge_pos': pos.copy(), 'band_pos': pos.copy(), 'normal': np.array([0,0])})
            continue
        tangent = tangent / tlen
        normal = np.array([-tangent[1], tangent[0]])
        to_c = center - pos
        if np.dot(normal, to_c) < 0:
            normal = -normal
        band_pos = pos + normal * (INNER_OFFSET + BAND_WIDTH // 2)
        results.append({'edge_pos': pos.copy(), 'band_pos': band_pos, 'normal': normal})
    return results

band_a = compute_band_pts(sample_a, side_a['piece_center'])
band_bf = compute_band_pts(sample_bf, side_b['piece_center'])

# Extract inner bands for NCC
ba_colors, ba_gray = extract_inner_band(color_a, sample_a, side_a['piece_center'], mask_a)
bb_colors, bb_gray = extract_inner_band(color_b, sample_bf, side_b['piece_center'], mask_b)

n = min(len(ba_gray), len(bb_gray))
ncc = compute_pattern_ncc(ba_gray[:n], bb_gray[:n])
cd_mean, cd_median = compute_seam_color_diff(ba_colors[:n], bb_colors[:n])
grad = compute_gradient_consistency(ba_gray[:n], bb_gray[:n])

print(f"NCC (arc-length both): {ncc:.4f}")
print(f"Color diff: {cd_mean:.2f}")
print(f"Grad score: {grad}")

# Transform B_flipped to align with A for overlay
def get_transform(src_verts, tgt_verts):
    src_mid = (src_verts[0] + src_verts[-1]) / 2.0
    tgt_mid = (tgt_verts[0] + tgt_verts[-1]) / 2.0
    src_theta = math.atan2(src_verts[-1][1] - src_verts[0][1], src_verts[-1][0] - src_verts[0][0])
    tgt_theta = math.atan2(tgt_verts[-1][1] - tgt_verts[0][1], tgt_verts[-1][0] - tgt_verts[0][0])
    rot = tgt_theta + math.pi - src_theta
    return src_mid, tgt_mid, rot

def apply_tf(pts, src_mid, tgt_mid, rot):
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    result = []
    for v in pts:
        dx = v[0] - src_mid[0]
        dy = v[1] - src_mid[1]
        result.append(np.array([dx*cos_r - dy*sin_r + tgt_mid[0],
                                dx*sin_r + dy*cos_r + tgt_mid[1]]))
    return result

src_mid, tgt_mid, rot = get_transform(side_b['vertices'], verts_a)

# Transform B's full outline
from show_connectivity import _get_outline, _load_piece_data
piece_data = _load_piece_data(DEDUPED_PATH)

outline_a_full = _get_outline(piece_data[PID_A])

outline_b_raw = _get_outline(piece_data[PID_B])
# Transform outline_b: rotate around B_flipped's side midpoint, align to A
outline_b_aligned = apply_tf(outline_b_raw, src_mid, tgt_mid, rot)

# Transform B_flipped's sample and band points to A's coordinate system
sample_bf_aligned = apply_tf(sample_bf, src_mid, tgt_mid, rot)
band_bf_aligned = []
for b in band_bf:
    ep = apply_tf([b['edge_pos']], src_mid, tgt_mid, rot)[0]
    bp = apply_tf([b['band_pos']], src_mid, tgt_mid, rot)[0]
    band_bf_aligned.append({'edge_pos': ep, 'band_pos': bp})

# Compute bounding box
all_pts = outline_a_full + outline_b_aligned + [b['edge_pos'].tolist() for b in band_a] + [b['edge_pos'].tolist() for b in band_bf_aligned]
min_x = min(p[0] for p in all_pts)
max_x = max(p[0] for p in all_pts)
min_y = min(p[1] for p in all_pts)
max_y = max(p[1] for p in all_pts)
data_w = max_x - min_x
data_h = max_y - min_y

canvas_w = 1400
margin = 60
scale = min((canvas_w - 2*margin) / data_w, (800 - 2*margin) / data_h)
canvas_h = int(data_h * scale) + 2*margin + 60

def tc(x, y):
    return ((x - min_x) * scale + margin, (y - min_y) * scale + margin + 40)

canvas = Image.new('RGBA', (canvas_w, canvas_h), (40, 40, 40, 255))
draw = ImageDraw.Draw(canvas)

draw.text((15, 8),
          f"Overlay: Piece {PID_A}[{SI_A}] + Piece {PID_B}[{SI_B}] — Arc-length sampling both edges  NCC={ncc:.4f}",
          fill=(255, 255, 255, 255), font=fonts['title'])

# Draw piece outlines
for outline, color, width in [
    (outline_a_full, (100, 150, 255, 180), 2),
    (outline_b_aligned, (255, 100, 100, 180), 2),
]:
    canvas_pts = [tc(x, y) for x, y in outline]
    if len(canvas_pts) >= 3:
        draw.polygon(canvas_pts, fill=None, outline=color, width=width)

# Draw shared side (A's edge)
side_pts = [tc(v[0], v[1]) for v in verts_a[::max(1, len(verts_a)//50)]]
if len(side_pts) >= 2:
    draw.line(side_pts, fill=(255, 200, 0, 200), width=2)

# Draw B_flipped's edge (aligned)
bf_edge_pts = [tc(v[0], v[1]) for v in sample_bf_aligned]
if len(bf_edge_pts) >= 2:
    draw.line(bf_edge_pts, fill=(255, 80, 80, 150), width=1)

# Draw A's sample points (blue dots, green numbers)
for i in range(N_SAMPLES):
    ep = tc(band_a[i]['edge_pos'][0], band_a[i]['edge_pos'][1])
    bp = tc(band_a[i]['band_pos'][0], band_a[i]['band_pos'][1])

    draw.ellipse([ep[0]-5, ep[1]-5, ep[0]+5, ep[1]+5], fill=(50, 140, 255, 255), outline=(255,255,255,180))
    draw.ellipse([bp[0]-3, bp[1]-3, bp[0]+3, bp[1]+3], fill=(50, 140, 255, 180))
    draw.text((ep[0]+6, ep[1]-8), str(i), fill=(0, 255, 100, 255), font=fonts['idx'])

# Draw B_flipped's sample points (red dots, green numbers)
for i in range(N_SAMPLES):
    ep = tc(band_bf_aligned[i]['edge_pos'][0], band_bf_aligned[i]['edge_pos'][1])
    bp = tc(band_bf_aligned[i]['band_pos'][0], band_bf_aligned[i]['band_pos'][1])

    draw.ellipse([ep[0]-5, ep[1]-5, ep[0]+5, ep[1]+5], fill=(255, 80, 80, 255), outline=(255,255,255,180))
    draw.ellipse([bp[0]-3, bp[1]-3, bp[0]+3, bp[1]+3], fill=(255, 80, 80, 180))
    draw.text((ep[0]-22, ep[1]+4), str(i), fill=(0, 255, 100, 255), font=fonts['idx'])

# Draw correspondence lines between same-index points
for i in range(N_SAMPLES):
    a_ep = tc(band_a[i]['edge_pos'][0], band_a[i]['edge_pos'][1])
    b_ep = tc(band_bf_aligned[i]['edge_pos'][0], band_bf_aligned[i]['edge_pos'][1])
    draw.line([a_ep, b_ep], fill=(255, 255, 0, 80), width=1)

# Legend
ly = canvas_h - 18
draw.ellipse([15, ly-4, 23, ly+4], fill=(50, 140, 255, 255))
draw.text((28, ly-7), f"Piece {PID_A} (arc-length)", fill=(100, 180, 255, 255), font=fonts['small'])

draw.ellipse([220, ly-4, 228, ly+4], fill=(255, 80, 80, 255))
draw.text((233, ly-7), f"Piece {PID_B} flipped (arc-length)", fill=(255, 130, 130, 255), font=fonts['small'])

draw.text((470, ly-7), "Green = index number", fill=(0, 255, 100, 255), font=fonts['small'])
draw.text((650, ly-7), f"NCC={ncc:.4f}  ColorDiff={cd_mean:.1f}", fill=(200, 200, 200, 255), font=fonts['small'])

out_path = os.path.join(CHECK_PATH, f'overlay_arclength_{PID_A}_{PID_B}.png')
canvas.save(out_path)
print(f"Saved: {out_path}")

# Distance stats
dists = []
for i in range(N_SAMPLES):
    d = np.linalg.norm(band_a[i]['edge_pos'] - band_bf_aligned[i]['edge_pos'])
    dists.append((i, d))
dists.sort(key=lambda x: x[1])
print("\nPoint distances (sorted):")
for idx, d in dists:
    marker = "OK" if d < 20 else ("~" if d < 50 else "!!")
    print(f"  [{idx:2d}] dist={d:.1f} {marker}")
print(f"\n  <20px: {sum(1 for _,d in dists if d<20)}/{len(dists)}")
print(f"  <50px: {sum(1 for _,d in dists if d<50)}/{len(dists)}")
print(f"  Mean: {np.mean([d for _,d in dists]):.1f}")
