import os
import sys
import json
import math
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))

from PIL import Image, ImageDraw, ImageFont
import importlib
import common.texture_verify
importlib.reload(common.texture_verify)
from common.texture_verify import (
    load_side_data, load_color_image, extract_inner_band,
    compute_pattern_ncc, compute_seam_color_diff,
    compute_texture_richness, compute_gradient_consistency,
    _resample_polyline, _find_corresponding_points_on_edge,
    _compute_transform, _apply_transform, _apply_inverse_transform,
    INNER_OFFSET, BAND_WIDTH, N_SAMPLES, SAMPLE_RADIUS
)
from show_connectivity import (
    _side_angle, _side_midpoint, _get_outline, _get_centroid,
    _load_piece_data, _load_piece_images, _SIDE_NAMES
)

from config import get_vector_path, get_color_path, get_connectivity_path, get_check_path
from pipline.debug_utils import load_fonts

DEDUPED_PATH = get_vector_path()
COLOR_PATH = get_color_path()
CONNECTIVITY_PATH = get_connectivity_path()
CHECK_PATH = get_check_path()

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3


def _compute_band_sample_points(side_data, color_image, binary_mask):
    verts = side_data['vertices']
    center = side_data['piece_center']
    h, w = color_image.shape[:2]
    resampled = _resample_polyline(verts, N_SAMPLES)
    points = []
    for i in range(N_SAMPLES):
        if i == 0:
            tangent = resampled[1] - resampled[0]
        elif i == N_SAMPLES - 1:
            tangent = resampled[-1] - resampled[-2]
        else:
            tangent = resampled[i + 1] - resampled[i - 1]
        tlen = np.linalg.norm(tangent)
        if tlen < 1e-6:
            points.append(None)
            continue
        tangent = tangent / tlen
        normal = np.array([-tangent[1], tangent[0]])
        to_c = center - resampled[i]
        if np.dot(normal, to_c) < 0:
            normal = -normal

        band_positions = []
        band_colors = []
        for d in range(INNER_OFFSET, INNER_OFFSET + BAND_WIDTH):
            pt = resampled[i] + normal * d
            px_c, py_c = int(round(pt[0])), int(round(pt[1]))
            patch_pixels = []
            for dy in range(-SAMPLE_RADIUS, SAMPLE_RADIUS + 1):
                for dx in range(-SAMPLE_RADIUS, SAMPLE_RADIUS + 1):
                    py = py_c + dy
                    px = px_c + dx
                    if 0 <= py < h and 0 <= px < w and binary_mask[py, px] > 0:
                        patch_pixels.append(color_image[py, px].astype(np.float64))
            if patch_pixels:
                avg_color = np.mean(patch_pixels, axis=0)
                band_positions.append((pt[0], pt[1]))
                band_colors.append(avg_color)

        if band_positions:
            avg_pos = np.mean(band_positions, axis=0)
            avg_color = np.mean(band_colors, axis=0)
            gray = 0.114 * avg_color[0] + 0.587 * avg_color[1] + 0.299 * avg_color[2]
            points.append({
                'edge_pos': resampled[i].copy(),
                'band_pos': avg_pos,
                'normal': normal,
                'avg_color_bgr': avg_color,
                'gray': gray,
            })
        else:
            points.append(None)
    return points


def _compute_corresponding_band_points(side_data_a, side_data_b, color_image_b, binary_mask_b):
    verts_a = side_data_a['vertices']
    verts_bf = side_data_b['vertices'][::-1].copy()
    center_b = side_data_b['piece_center']
    h, w = color_image_b.shape[:2]

    sample_a = _resample_polyline(verts_a, N_SAMPLES)
    src_mid, tgt_mid, rot = _compute_transform(verts_bf, verts_a)
    verts_bf_aligned = _apply_transform(verts_bf, src_mid, tgt_mid, rot)
    corr_aligned = _find_corresponding_points_on_edge(sample_a, verts_bf_aligned)
    corr_original = _apply_inverse_transform(corr_aligned, src_mid, tgt_mid, rot)

    points = []
    for i in range(N_SAMPLES):
        pos = corr_original[i]
        if i == 0:
            tangent = corr_original[1] - corr_original[0]
        elif i == N_SAMPLES - 1:
            tangent = corr_original[-1] - corr_original[-2]
        else:
            tangent = corr_original[i + 1] - corr_original[i - 1]
        tlen = np.linalg.norm(tangent)
        if tlen < 1e-6:
            points.append(None)
            continue
        tangent = tangent / tlen
        normal = np.array([-tangent[1], tangent[0]])
        to_c = center_b - pos
        if np.dot(normal, to_c) < 0:
            normal = -normal

        band_positions = []
        band_colors = []
        for d in range(INNER_OFFSET, INNER_OFFSET + BAND_WIDTH):
            pt = pos + normal * d
            px_c, py_c = int(round(pt[0])), int(round(pt[1]))
            patch_pixels = []
            for dy in range(-SAMPLE_RADIUS, SAMPLE_RADIUS + 1):
                for dx in range(-SAMPLE_RADIUS, SAMPLE_RADIUS + 1):
                    py = py_c + dy
                    px = px_c + dx
                    if 0 <= py < h and 0 <= px < w and binary_mask_b[py, px] > 0:
                        patch_pixels.append(color_image_b[py, px].astype(np.float64))
            if patch_pixels:
                avg_color = np.mean(patch_pixels, axis=0)
                band_positions.append((pt[0], pt[1]))
                band_colors.append(avg_color)

        if band_positions:
            avg_pos = np.mean(band_positions, axis=0)
            avg_color = np.mean(band_colors, axis=0)
            gray = 0.114 * avg_color[0] + 0.587 * avg_color[1] + 0.299 * avg_color[2]
            points.append({
                'edge_pos': pos.copy(),
                'band_pos': avg_pos,
                'normal': normal,
                'avg_color_bgr': avg_color,
                'gray': gray,
            })
        else:
            points.append(None)
    return points


def main():
    fonts = load_fonts(title_size=28, idx_size=11)
    try:
        fonts['section'] = ImageFont.truetype("arialbd.ttf", 22)
        fonts['label'] = ImageFont.truetype("arial.ttf", 16)
        fonts['small'] = ImageFont.truetype("arial.ttf", 14)
        fonts['tiny'] = ImageFont.truetype("arial.ttf", 12)
        fonts['metric'] = ImageFont.truetype("arialbd.ttf", 18)
    except Exception:
        for k in ['section', 'label', 'small', 'tiny', 'metric']:
            fonts[k] = ImageFont.load_default()

    print("Loading data...")
    piece_data = _load_piece_data(DEDUPED_PATH)

    side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
    side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)

    color_a_bgr, mask_a = load_color_image(COLOR_PATH, PID_A)
    color_b_bgr, mask_b = load_color_image(COLOR_PATH, PID_B)

    img_a_rgba = Image.open(os.path.join(COLOR_PATH, f"piece_{PID_A}.png")).convert('RGBA')
    img_b_rgba = Image.open(os.path.join(COLOR_PATH, f"piece_{PID_B}.png")).convert('RGBA')

    print("Computing band sample points...")
    pts_a = _compute_band_sample_points(side_a, color_a_bgr, mask_a)
    pts_b_corr = _compute_corresponding_band_points(side_a, side_b, color_b_bgr, mask_b)

    band_a_gray = np.array([p['gray'] for p in pts_a if p is not None])
    band_b_corr_gray = np.array([p['gray'] for p in pts_b_corr if p is not None])

    band_a_colors = np.array([p['avg_color_bgr'] for p in pts_a if p is not None])
    band_b_corr_colors = np.array([p['avg_color_bgr'] for p in pts_b_corr if p is not None])

    n = min(len(band_a_gray), len(band_b_corr_gray))
    band_a = band_a_gray[:n]
    band_b = band_b_corr_gray[:n]

    ncc = compute_pattern_ncc(band_a, band_b)
    color_diff_mean, color_diff_median = compute_seam_color_diff(band_a_colors[:n], band_b_corr_colors[:n])
    grad_score = compute_gradient_consistency(band_a, band_b)
    tex_a = compute_texture_richness(band_a)
    tex_b = compute_texture_richness(band_b)

    print(f"NCC (correspondence-based): {ncc:.4f}")
    print(f"Color diff: {color_diff_mean:.2f}")
    print(f"Grad score: {grad_score}")

    canvas_w = 2000
    header_h = 50

    piece_strip_h = 450
    plot_h = 300
    detail_h = 180
    gap = 12
    section_hdr = 28

    total_h = (header_h + gap +
               section_hdr + piece_strip_h + gap +
               section_hdr + plot_h + gap +
               section_hdr + detail_h + 20)

    canvas = Image.new('RGBA', (canvas_w, total_h), (235, 235, 235, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((20, 12),
              f"NCC Debug: Piece {PID_A}[{SI_A}] ({_SIDE_NAMES[SI_A]}) <-> "
              f"Piece {PID_B}[{SI_B}] ({_SIDE_NAMES[SI_B]})",
              fill=(0, 0, 0, 255), font=fonts['title'])

    y = header_h + gap
    draw.rectangle([10, y, canvas_w - 10, y + section_hdr],
                   fill=(50, 50, 110, 255))
    draw.text((20, y + 2),
              "Band Sampling Points on Color Images (dots = NCC sample positions, orange = shared side)",
              fill=(255, 255, 255, 255), font=fonts['section'])
    y += section_hdr

    half_w = canvas_w // 2 - 10

    for col, (pid, si, side_d, img_rgba, pts_list, title_color, piece_label) in enumerate([
        (PID_A, SI_A, side_a, img_a_rgba, pts_a, (0, 60, 200), f"Piece {PID_A}"),
        (PID_B, SI_B, side_b, img_b_rgba, pts_b_corr, (180, 0, 0), f"Piece {PID_B} (corr)"),
    ]):
        x0 = 10 + col * (half_w + 10)
        cell = Image.new('RGBA', (half_w, piece_strip_h), (255, 255, 255, 255))

        w_img, h_img = img_rgba.size
        scale = min((half_w - 20) / w_img, (piece_strip_h - 40) / h_img)
        tw = int(w_img * scale)
        th = int(h_img * scale)
        resized = img_rgba.resize((tw, th), Image.LANCZOS)
        paste_x = (half_w - tw) // 2
        paste_y = 30
        cell.paste(resized, (paste_x, paste_y), resized)

        cell_draw = ImageDraw.Draw(cell)
        cell_draw.text((10, 5), piece_label, fill=title_color, font=fonts['section'])

        side_verts = side_d['vertices']
        s_p1 = side_verts[0]
        s_p2 = side_verts[-1]
        sp1x = paste_x + s_p1[0] * scale
        sp1y = paste_y + s_p1[1] * scale
        sp2x = paste_x + s_p2[0] * scale
        sp2y = paste_y + s_p2[1] * scale
        cell_draw.line([(sp1x, sp1y), (sp2x, sp2y)], fill=(255, 165, 0, 255), width=3)

        edge_line = []
        valid_pts = []
        for p in pts_list:
            if p is None:
                if edge_line:
                    edge_line.append(None)
                continue
            ex = paste_x + p['edge_pos'][0] * scale
            ey = paste_y + p['edge_pos'][1] * scale
            edge_line.append((ex, ey))
            bx = paste_x + p['band_pos'][0] * scale
            by = paste_y + p['band_pos'][1] * scale
            valid_pts.append((bx, by, p))

        if edge_line:
            segments = []
            seg = []
            for pt in edge_line:
                if pt is None:
                    if seg:
                        segments.append(seg)
                        seg = []
                else:
                    seg.append(pt)
            if seg:
                segments.append(seg)
            for seg in segments:
                if len(seg) >= 2:
                    cell_draw.line(seg, fill=(255, 165, 0, 200), width=2)

        for idx, (bx, by, p) in enumerate(valid_pts):
            bgr = p['avg_color_bgr']
            r, g, b_col = int(bgr[2]), int(bgr[1]), int(bgr[0])
            dot_r = max(4, int(6 * scale))
            cell_draw.ellipse([bx - dot_r, by - dot_r, bx + dot_r, by + dot_r],
                              fill=(r, g, b_col, 255), outline=(0, 0, 0, 255))
            if idx % 3 == 0:
                cell_draw.text((bx + dot_r + 1, by - 6), str(idx),
                               fill=(0, 0, 0, 200), font=fonts['idx'])

        band_line = [(bx, by) for bx, by, _ in valid_pts]
        if len(band_line) >= 2:
            cell_draw.line(band_line, fill=(0, 200, 0, 120), width=1)

        canvas.paste(cell, (x0, y))

    y += piece_strip_h + gap

    draw.rectangle([10, y, canvas_w - 10, y + section_hdr],
                   fill=(50, 50, 110, 255))
    draw.text((20, y + 2),
              "Gray Value Along Band (dots colored by actual pixel color)",
              fill=(255, 255, 255, 255), font=fonts['section'])
    y += section_hdr

    all_vals = np.concatenate([band_a, band_b])
    vmin = max(0, all_vals.min() - 5)
    vmax = all_vals.max() + 5
    if vmax - vmin < 1:
        vmax = vmin + 1

    plot_left = 55
    plot_right = canvas_w - 15
    plot_top = y + 25
    plot_bottom = y + plot_h - 25
    pw = plot_right - plot_left
    ph = plot_bottom - plot_top

    draw.rectangle([10, y, canvas_w - 10, y + plot_h], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))

    for i in range(5):
        val = vmin + (vmax - vmin) * i / 4
        py = plot_bottom - ph * i / 4
        draw.line([(plot_left, py), (plot_right, py)], fill=(230, 230, 230, 255))
        draw.text((12, py - 7), f"{val:.0f}", fill=(120, 120, 120, 255), font=fonts['tiny'])

    draw.text((canvas_w // 2 - 80, y + 4), "Gray Value Along Inner Band (Correspondence-based)",
              fill=(0, 0, 0, 255), font=fonts['label'])

    def val_to_py(v):
        return plot_bottom - ph * (v - vmin) / (vmax - vmin)

    def draw_colored_line(band_gray, band_colors_bgr, line_color, line_w, draw_dots):
        pts = []
        for i in range(n):
            px = plot_left + pw * i / (n - 1)
            py = val_to_py(band_gray[i])
            pts.append((px, py))
        if len(pts) >= 2:
            draw.line(pts, fill=line_color, width=line_w)
        if draw_dots:
            for i in range(n):
                px = pts[i][0]
                py = pts[i][1]
                bgr = band_colors_bgr[i]
                r, g, b_c = int(bgr[2]), int(bgr[1]), int(bgr[0])
                draw.ellipse([px - 4, py - 4, px + 4, py + 4],
                             fill=(r, g, b_c, 255), outline=(0, 0, 0, 180))

    draw_colored_line(band_a, band_a_colors[:n], (0, 0, 200, 255), 3, True)
    draw_colored_line(band_b, band_b_corr_colors[:n], (200, 0, 0, 255), 2, True)

    legend_y = y + plot_h - 18
    lx = plot_left
    draw.ellipse([lx, legend_y - 4, lx + 8, legend_y + 4], fill=(0, 0, 200, 255))
    draw.line([(lx + 12, legend_y), (lx + 32, legend_y)], fill=(0, 0, 200, 255), width=3)
    draw.text((lx + 36, legend_y - 7), f"Band A (piece {PID_A}[{SI_A}], dots=actual color)",
              fill=(0, 0, 200, 255), font=fonts['tiny'])

    lx += 350
    draw.ellipse([lx, legend_y - 4, lx + 8, legend_y + 4], fill=(200, 0, 0, 255))
    draw.line([(lx + 12, legend_y), (lx + 32, legend_y)], fill=(200, 0, 0, 255), width=2)
    draw.text((lx + 36, legend_y - 7), f"Band B corr (NCC={ncc:.4f}, dots=actual color)",
              fill=(200, 0, 0, 255), font=fonts['tiny'])

    lx += 380
    draw.text((lx, legend_y - 7),
              f"| ColorDiff={color_diff_mean:.1f}  GradScore={grad_score:.3f}" if grad_score else
              f"| ColorDiff={color_diff_mean:.1f}",
              fill=(80, 80, 80, 255), font=fonts['tiny'])

    y += plot_h + gap

    draw.rectangle([10, y, canvas_w - 10, y + section_hdr],
                   fill=(50, 50, 110, 255))
    draw.text((20, y + 2),
              "Band Color Strip Comparison (actual sampled colors side by side)",
              fill=(255, 255, 255, 255), font=fonts['section'])
    y += section_hdr

    strip_h = 40
    label_h = 18
    strip_margin = 100
    strip_w = canvas_w - 2 * strip_margin
    sw = strip_w / n

    for row, (band_colors_bgr, label, label_color) in enumerate([
        (band_a_colors[:n], f"Band A (Piece {PID_A}[{SI_A}])", (0, 0, 200)),
        (band_b_corr_colors[:n], f"Band B correspondence (NCC={ncc:.4f})", (200, 0, 0)),
    ]):
        ry = y + row * (strip_h + label_h + 8)
        draw.text((15, ry + strip_h // 2 - 7), label, fill=label_color, font=fonts['small'])

        for i in range(n):
            bgr = band_colors_bgr[i]
            r, g, b_c = int(bgr[2]), int(bgr[1]), int(bgr[0])
            sx = strip_margin + i * sw
            draw.rectangle([sx, ry, sx + sw + 1, ry + strip_h],
                           fill=(r, g, b_c, 255), outline=(200, 200, 200, 255))

        for i in range(0, n, 5):
            sx = strip_margin + i * sw
            draw.text((sx, ry + strip_h + 1), str(i), fill=(100, 100, 100, 255), font=fonts['tiny'])

    y += 2 * (strip_h + label_h + 8) + gap

    draw.rectangle([10, y, canvas_w - 10, y + section_hdr],
                   fill=(50, 50, 110, 255))
    draw.text((20, y + 2),
              "Summary Metrics",
              fill=(255, 255, 255, 255), font=fonts['section'])
    y += section_hdr

    box_w = (canvas_w - 50) // 3
    box_h = detail_h

    bx1 = 10
    draw.rectangle([bx1, y, bx1 + box_w, y + box_h],
                   fill=(230, 255, 230, 255), outline=(0, 160, 0, 255), width=2)
    draw.text((bx1 + 10, y + 5), "Correspondence-based NCC", fill=(0, 140, 0, 255), font=fonts['label'])
    lines = [
        f"NCC:                {ncc:.4f}",
        f"Color diff (mean):  {color_diff_mean:.2f}",
        f"Color diff (median):{color_diff_median:.2f}",
        f"Gradient score:     {grad_score:.4f}" if grad_score else "Gradient score:     N/A",
        f"Texture richness A: {tex_a:.4f}",
        f"Texture richness B: {tex_b:.4f}",
        f"Samples:            {n}",
    ]
    for i, line in enumerate(lines):
        draw.text((bx1 + 15, y + 28 + i * 20), line, fill=(0, 60, 0, 255), font=fonts['small'])

    bx2 = 10 + box_w + 15
    draw.rectangle([bx2, y, bx2 + box_w, y + box_h],
                   fill=(255, 235, 235, 255), outline=(200, 0, 0, 255), width=2)
    draw.text((bx2 + 10, y + 5), "Old (arc-length, NCC=0.2376)", fill=(200, 0, 0, 255), font=fonts['label'])
    old_lines = [
        f"NCC:                0.2376",
        f"Reason: independent arc-length sampling",
        f"per piece => spatial misalignment",
        f"",
        f"Mean correspondence distance: 412px",
        f"",
        f"",
    ]
    for i, line in enumerate(old_lines):
        draw.text((bx2 + 15, y + 28 + i * 18), line, fill=(80, 0, 0, 255), font=fonts['tiny'])

    bx3 = 10 + 2 * (box_w + 15)
    draw.rectangle([bx3, y, bx3 + box_w, y + box_h],
                   fill=(235, 235, 255, 255), outline=(0, 0, 200, 255), width=2)
    draw.text((bx3 + 10, y + 5), "Method Description", fill=(0, 0, 180, 255), font=fonts['label'])
    geo_lines = [
        f"1. Arc-length sample A's edge (covers tabs)",
        f"2. Transform B_flipped to align with A",
        f"3. For each A sample, find closest on B",
        f"4. Map back to B's original coordinates",
        f"5. Extract inner band at paired positions",
        f"",
        f"Mean corr distance: 54px (vs 412px old)",
    ]
    for i, line in enumerate(geo_lines):
        draw.text((bx3 + 15, y + 28 + i * 18), line, fill=(0, 0, 100, 255), font=fonts['tiny'])

    out_path = os.path.join(CHECK_PATH, f'ncc_debug_{PID_A}_{PID_B}.png')
    canvas.save(out_path)
    print(f"\nSaved: {out_path}")
    print(f"Canvas size: {canvas_w}x{total_h}")


if __name__ == '__main__':
    main()
