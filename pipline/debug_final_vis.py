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
    _resample_polyline, _find_corresponding_points_on_edge,
    _compute_transform, _apply_transform, _apply_inverse_transform,
    _edge_tangent_at, verify_match,
    INNER_OFFSET, BAND_WIDTH, N_SAMPLES
)
from show_connectivity import _load_piece_data, _get_outline

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '3_vector')
COLOR_PATH = os.path.join(OUTPUT_DIR, '2_piece_colors')
CHECK_PATH = os.path.join(OUTPUT_DIR, 'check')

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

try:
    fonts = {
        'title': ImageFont.truetype("arialbd.ttf", 26),
        'section': ImageFont.truetype("arialbd.ttf", 18),
        'label': ImageFont.truetype("arial.ttf", 14),
        'small': ImageFont.truetype("arial.ttf", 12),
        'idx': ImageFont.truetype("arialbd.ttf", 10),
    }
except:
    default = ImageFont.load_default()
    fonts = {k: default for k in ['title', 'section', 'label', 'small', 'idx']}

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)
color_a, mask_a = load_color_image(COLOR_PATH, PID_A)
color_b, mask_b = load_color_image(COLOR_PATH, PID_B)

img_a = Image.open(os.path.join(COLOR_PATH, f"piece_{PID_A}.png")).convert('RGBA')
img_b = Image.open(os.path.join(COLOR_PATH, f"piece_{PID_B}.png")).convert('RGBA')

verts_a = side_a['vertices']
verts_b = side_b['vertices']
verts_bf = verts_b[::-1].copy()

sample_a = _resample_polyline(verts_a, N_SAMPLES)

src_mid, tgt_mid, rot = _compute_transform(verts_b, verts_a)
verts_bf_aligned = _apply_transform(verts_bf, src_mid, tgt_mid, rot)
corr_aligned = _find_corresponding_points_on_edge(sample_a, verts_bf_aligned)
corr_orig = _apply_inverse_transform(corr_aligned, src_mid, tgt_mid, rot)

ba_c, ba_g_raw = extract_inner_band(color_a, sample_a, mask_a, normal_side='left')
bb_c, bb_g_raw = extract_inner_band(color_b, corr_orig, mask_b, normal_side='right', edge_vertices=verts_bf)

valid = [i for i in range(min(len(ba_g_raw), len(bb_g_raw)))
         if ba_g_raw[i] is not None and bb_g_raw[i] is not None]
valid_set = set(valid)
ba_g = np.array([ba_g_raw[i] for i in valid])
bb_g = np.array([bb_g_raw[i] for i in valid])
ba_c_arr = np.array([ba_c[i] for i in valid])
bb_c_arr = np.array([bb_c[i] for i in valid])

n = len(valid)
ncc = compute_pattern_ncc(ba_g, bb_g)
cd_mean, cd_median = compute_seam_color_diff(ba_c_arr, bb_c_arr)
grad = compute_gradient_consistency(ba_g[:n], bb_g[:n])
tex_a = compute_texture_richness(ba_g[:n])
tex_b = compute_texture_richness(bb_g[:n])

print(f"NCC={ncc:.4f}  ColorDiff={cd_mean:.2f}  Grad={grad}  Samples={n}")

piece_data = _load_piece_data(DEDUPED_PATH)

def tf_pt(v, sm, tm, r):
    cos_r, sin_r = math.cos(r), math.sin(r)
    dx = v[0] - sm[0]
    dy = v[1] - sm[1]
    return (dx*cos_r - dy*sin_r + tm[0], dx*sin_r + dy*cos_r + tm[1])

def tf_pt_arr(v, sm, tm, r):
    cos_r, sin_r = math.cos(r), math.sin(r)
    dx = v[0] - sm[0]
    dy = v[1] - sm[1]
    return np.array([dx*cos_r - dy*sin_r + tm[0], dx*sin_r + dy*cos_r + tm[1]])

outline_a = _get_outline(piece_data[PID_A])
outline_b_raw = _get_outline(piece_data[PID_B])
outline_b = [tf_pt(np.array(p), src_mid, tgt_mid, rot) for p in outline_b_raw]

sample_a_list = [tuple(p) for p in sample_a]
corr_aligned_list = [tuple(p) for p in corr_aligned]

all_pts = (outline_a + outline_b + sample_a_list + corr_aligned_list)
min_x = min(p[0] for p in all_pts) - 30
max_x = max(p[0] for p in all_pts) + 30
min_y = min(p[1] for p in all_pts) - 30
max_y = max(p[1] for p in all_pts) + 30
data_w = max_x - min_x
data_h = max_y - min_y

canvas_w = 1200
margin = 60
scale = min((canvas_w - 2*margin) / data_w, (700 - 2*margin) / data_h)
canvas_h = int(data_h * scale) + 2*margin + 80

def tc(x, y):
    return ((x - min_x) * scale + margin, (y - min_y) * scale + margin + 50)

canvas = Image.new('RGBA', (canvas_w, canvas_h), (40, 40, 40, 255))
draw = ImageDraw.Draw(canvas)

draw.text((15, 8),
          f"Piece {PID_A}[{SI_A}] + Piece {PID_B}[{SI_B}] aligned | NCC={ncc:.4f}  ColorDiff={cd_mean:.1f}  Grad={grad:.3f}",
          fill=(255, 255, 255, 255), font=fonts['title'])

# Paste color images
for pid, img_src, sides, si in [
    (PID_A, img_a, piece_data[PID_A], SI_A),
    (PID_B, img_b, piece_data[PID_B], SI_B),
]:
    src_verts = np.array(sides[si]['vertices'])
    tgt_verts = verts_a

    s_mid = (src_verts[0] + src_verts[-1]) / 2.0
    t_mid = (tgt_verts[0] + tgt_verts[-1]) / 2.0
    s_theta = math.atan2(src_verts[-1][1] - src_verts[0][1], src_verts[-1][0] - src_verts[0][0])
    t_theta = math.atan2(tgt_verts[-1][1] - tgt_verts[0][1], tgt_verts[-1][0] - tgt_verts[0][0])
    r2 = t_theta + math.pi - s_theta

    w_img, h_img = img_src.size

    if pid == PID_A:
        out_corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
        o_min_x, o_min_y = 0.0, 0.0
        out_w, out_h = w_img, h_img
        t_img = img_src
    else:
        corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
        out_corners = []
        for x, y in corners:
            dx = x - s_mid[0]
            dy = y - s_mid[1]
            ox = dx * math.cos(r2) - dy * math.sin(r2) + t_mid[0]
            oy = dx * math.sin(r2) + dy * math.cos(r2) + t_mid[1]
            out_corners.append((ox, oy))

        all_x = [c[0] for c in out_corners]
        all_y = [c[1] for c in out_corners]
        o_min_x = min(all_x)
        o_min_y = min(all_y)
        o_max_x = max(all_x)
        o_max_y = max(all_y)
        out_w = int(math.ceil(o_max_x - o_min_x)) + 1
        out_h = int(math.ceil(o_max_y - o_min_y)) + 1

        cos_neg = math.cos(-r2)
        sin_neg = math.sin(-r2)
        a = cos_neg
        b_ = -sin_neg
        c_ = cos_neg * (o_min_x - t_mid[0]) - sin_neg * (o_min_y - t_mid[1]) + s_mid[0]
        d = sin_neg
        e = cos_neg
        f_ = sin_neg * (o_min_x - t_mid[0]) + cos_neg * (o_min_y - t_mid[1]) + s_mid[1]

        t_img = img_src.transform((out_w, out_h), Image.AFFINE, (a, b_, c_, d, e, f_), resample=Image.BICUBIC)

    tw = int(t_img.size[0] * scale)
    th = int(t_img.size[1] * scale)
    if tw > 0 and th > 0:
        resized = t_img.resize((tw, th), Image.LANCZOS)
        px = int(tc(o_min_x, o_min_y)[0])
        py = int(tc(o_min_x, o_min_y)[1])
        canvas.paste(resized, (px, py), resized)

# Draw outlines
for outline, color in [(outline_a, (80,140,255,180)), (outline_b, (255,100,100,180))]:
    pts = [tc(x, y) for x, y in outline]
    if len(pts) >= 3:
        draw.polygon(pts, fill=None, outline=color, width=2)

# Draw shared side (orange)
side_pts = [tc(v[0], v[1]) for v in verts_a[::max(1, len(verts_a)//60)]]
if len(side_pts) >= 2:
    draw.line(side_pts, fill=(255, 200, 0, 200), width=2)

# Draw A sample points with band colors
for i in range(N_SAMPLES):
    ax, ay = tc(sample_a[i][0], sample_a[i][1])

    is_valid = i in valid_set
    if is_valid:
        bgr = ba_c[i]
        r_col, g_col, b_col = int(bgr[2]), int(bgr[1]), int(bgr[0])

        tangents = []
        if i > 0:
            d = sample_a[i] - sample_a[i-1]
            dl = np.linalg.norm(d)
            if dl > 1e-6: tangents.append(d / dl)
        if i < N_SAMPLES - 1:
            d = sample_a[i+1] - sample_a[i]
            dl = np.linalg.norm(d)
            if dl > 1e-6: tangents.append(d / dl)
        if tangents:
            tangent = np.mean(tangents, axis=0)
            tl = np.linalg.norm(tangent)
            if tl > 1e-6:
                tangent = tangent / tl
                normal = np.array([-tangent[1], tangent[0]])
                band_pos = sample_a[i] + normal * (INNER_OFFSET + BAND_WIDTH // 2)
                bpx, bpy = tc(band_pos[0], band_pos[1])
                draw.ellipse([bpx-4, bpy-4, bpx+4, bpy+4], fill=(r_col, g_col, b_col, 255), outline=(255,255,255,200))

        draw.ellipse([ax-5, ay-5, ax+5, ay+5], fill=(50, 140, 255, 255), outline=(255,255,0,255))
        draw.text((ax+6, ay-9), str(i), fill=(100, 200, 255, 255), font=fonts['idx'])
    else:
        draw.ellipse([ax-4, ay-4, ax+4, ay+4], fill=None, outline=(50, 140, 255, 80))
        draw.line([ax-6, ay-6, ax+6, ay+6], fill=(255, 80, 80, 200), width=1)
        draw.line([ax-6, ay+6, ax+6, ay-6], fill=(255, 80, 80, 200), width=1)
        draw.text((ax+6, ay-9), str(i), fill=(120, 120, 120, 180), font=fonts['idx'])

# Draw B correspondence points with band colors
for i in range(N_SAMPLES):
    bx, by = tc(corr_aligned[i][0], corr_aligned[i][1])

    is_valid = i in valid_set
    if is_valid:
        bgr = bb_c[i]
        r_col, g_col, b_col = int(bgr[2]), int(bgr[1]), int(bgr[0])

        tangent = _edge_tangent_at(verts_bf, corr_orig[i])
        normal = np.array([tangent[1], -tangent[0]])
        band_pos = corr_orig[i] + normal * (INNER_OFFSET + BAND_WIDTH // 2)
        band_aligned = tf_pt_arr(band_pos, src_mid, tgt_mid, rot)
        bpx, bpy = tc(band_aligned[0], band_aligned[1])
        draw.ellipse([bpx-4, bpy-4, bpx+4, bpy+4], fill=(r_col, g_col, b_col, 255), outline=(255,255,255,200))

        draw.ellipse([bx-5, by-5, bx+5, by+5], fill=(255, 80, 80, 255), outline=(255,255,0,255))
        draw.text((bx-18, by+4), str(i), fill=(255, 150, 150, 255), font=fonts['idx'])
    else:
        draw.ellipse([bx-4, by-4, bx+4, by+4], fill=None, outline=(255, 80, 80, 80))
        draw.line([bx-6, by-6, bx+6, by+6], fill=(255, 80, 80, 200), width=1)
        draw.line([bx-6, by+6, bx+6, by-6], fill=(255, 80, 80, 200), width=1)
        draw.text((bx-18, by+4), str(i), fill=(120, 120, 120, 180), font=fonts['idx'])

# Info panel at bottom
info_y = canvas_h - 65
draw.rectangle([10, info_y, canvas_w - 10, canvas_h - 5], fill=(20, 20, 20, 230), outline=(100,100,100,255))
info_lines = [
    f"NCC: {ncc:.4f}  |  ColorDiff: mean={cd_mean:.1f} median={cd_median:.1f}  |  GradScore: {grad:.4f}  |  Texture: A={tex_a:.3f} B={tex_b:.3f}  |  Samples: {n}",
    f"Solid=valid points (used in NCC)  X=skipped (band out of mask)  |  reject=False",
]
for j, line in enumerate(info_lines):
    draw.text((20, info_y + 5 + j * 22), line, fill=(220, 220, 220, 255), font=fonts['label'])

# Gray comparison strip
strip_y = info_y - 55
strip_margin = 80
strip_w = canvas_w - 2 * strip_margin
sw = strip_w / n if n > 0 else 1
for row, (grays, colors, label, color) in enumerate([
    (ba_g, ba_c_arr, f"A (Piece {PID_A})", (80,140,255)),
    (bb_g, bb_c_arr, f"B (Piece {PID_B})", (255,100,100)),
]):
    ry = strip_y + row * 24
    draw.text((15, ry + 2), label, fill=color, font=fonts['small'])
    for i in range(n):
        bgr = colors[i]
        r_col, g_col, b_col = int(bgr[2]), int(bgr[1]), int(bgr[0])
        sx = strip_margin + i * sw
        draw.rectangle([sx, ry, sx + sw + 1, ry + 20], fill=(r_col, g_col, b_col, 255))

out_path = os.path.join(CHECK_PATH, f'ncc_final_{PID_A}_{PID_B}.png')
canvas.save(out_path)
print(f"Saved: {out_path}")
