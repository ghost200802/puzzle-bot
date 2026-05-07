import os, sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))
from common.config import VECTOR_DIR

OUTPUT_DIR = os.path.join(_here, '..', '..', 'output', 'puzzle_new')
VECTOR_PATH = os.path.join(OUTPUT_DIR, VECTOR_DIR)
COLOR_DIR = os.path.join(OUTPUT_DIR, '2_piece_colors')

try:
    FONT = ImageFont.truetype("arial.ttf", 13)
    FONT_L = ImageFont.truetype("arial.ttf", 15)
    FONT_XL = ImageFont.truetype("arial.ttf", 17)
except Exception:
    FONT = ImageFont.load_default()
    FONT_L = FONT
    FONT_XL = FONT


def draw_corners(img_rgba, corners, color):
    pil = Image.fromarray(img_rgba.copy())
    draw = ImageDraw.Draw(pil)
    for i, c in enumerate(corners):
        cx, cy = int(c[0]), int(c[1])
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=color)
        draw.text((cx + 6, cy - 8), f"C{i}", fill=color, font=FONT)
    for i in range(len(corners)):
        c1 = corners[i]
        c2 = corners[(i + 1) % len(corners)]
        draw.line([(int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1]))],
                  fill=color, width=2)
    return np.array(pil)


def make_histogram_panel(vals_a, vals_b, title, w=400, h=150):
    panel = Image.new('RGBA', (w, h), (40, 40, 50, 255))
    draw = ImageDraw.Draw(panel)

    bins = np.arange(257)
    hist_a, _ = np.histogram(vals_a, bins=bins)
    hist_b, _ = np.histogram(vals_b, bins=bins)
    max_val = max(hist_a.max(), hist_b.max(), 1)

    bar_w = w / 256
    for i in range(256):
        x0 = int(i * bar_w)
        x1 = int((i + 1) * bar_w)
        ha = int(hist_a[i] / max_val * (h - 30))
        hb = int(hist_b[i] / max_val * (h - 30))
        if ha > 0:
            draw.rectangle([x0, h - 15 - ha, x1, h - 15], fill=(100, 150, 255, 180))
        if hb > 0:
            draw.rectangle([x0, h - 15 - hb, x1, h - 15], fill=(255, 100, 100, 120))

    draw.text((4, 2), title, fill=(255, 255, 200), font=FONT_L)
    draw.text((4, h - 14), f"mean_a={vals_a.mean():.1f} std_a={vals_a.std():.1f}", fill=(100, 150, 255), font=FONT)
    draw.text((w // 2, h - 14), f"mean_b={vals_b.mean():.1f} std_b={vals_b.std():.1f}", fill=(255, 100, 100), font=FONT)
    return panel


def make_scatter_panel(vals_a, vals_b, title, ncc_val, w=300, h=300):
    panel = Image.new('RGBA', (w, h), (40, 40, 50, 255))
    draw = ImageDraw.Draw(panel)

    margin = 30
    plot_w = w - 2 * margin
    plot_h = h - 2 * margin - 15

    draw.rectangle([margin, margin, margin + plot_w, margin + plot_h], outline=(80, 80, 80))

    step = max(1, len(vals_a) // 3000)
    va = vals_a[::step]
    vb = vals_b[::step]

    for a, b in zip(va, vb):
        px = margin + int(a / 255 * plot_w)
        py = margin + plot_h - int(b / 255 * plot_h)
        px = max(margin, min(margin + plot_w, px))
        py = max(margin, min(margin + plot_h, py))
        draw.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(100, 200, 255, 60))

    diag_y0 = margin + plot_h
    diag_y1 = margin
    draw.line([(margin, diag_y0), (margin + plot_w, diag_y1)], fill=(255, 255, 100, 100), width=1)

    draw.text((4, 2), f"{title}  NCC={ncc_val:.4f}", fill=(255, 255, 200), font=FONT_L)
    draw.text((margin, margin + plot_h + 3), "gray_a ->", fill=(180, 180, 180), font=FONT)
    draw.text((2, margin), "b", fill=(180, 180, 180), font=FONT)
    return panel


def compute_ncc_raw(ga, gb):
    ga_n = ga - ga.mean()
    gb_n = gb - gb.mean()
    denom = np.sqrt(np.sum(ga_n ** 2) * np.sum(gb_n ** 2))
    if denom < 1e-6:
        return 0.0
    return float(np.sum(ga_n * gb_n) / denom)


def compute_ncc_normalized(ga, gb):
    mean_a = ga.mean()
    mean_b = gb.mean()
    gb_shifted = gb - mean_b + mean_a
    ga_n = ga - mean_a
    gb_n = gb_shifted - mean_a
    denom = np.sqrt(np.sum(ga_n ** 2) * np.sum(gb_n ** 2))
    if denom < 1e-6:
        return 0.0
    return float(np.sum(ga_n * gb_n) / denom)


def match_histogram(src_vals, ref_vals):
    src_vals = np.clip(src_vals, 0, 255).astype(np.uint8)
    ref_vals = np.clip(ref_vals, 0, 255).astype(np.uint8)

    src_hist, _ = np.histogram(src_vals, bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(ref_vals, bins=256, range=(0, 256))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1
    ref_cdf /= ref_cdf[-1] if ref_cdf[-1] > 0 else 1

    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(ref_cdf - src_cdf[i])
        mapping[i] = int(np.argmin(diff))

    matched = mapping[src_vals].astype(np.float64)
    return matched, mapping


def match_histogram_2d(gray_img, ref_gray, mask):
    result = gray_img.copy().astype(np.float64)
    for c in range(3):
        src_ch = gray_img[:, :, c][mask].ravel().astype(np.uint8) if gray_img.ndim == 3 else None
    src_vals = gray_img[mask].ravel().astype(np.uint8)
    ref_vals = ref_gray[mask].ravel().astype(np.uint8)
    matched_flat, mapping = match_histogram(src_vals.astype(np.float64), ref_vals.astype(np.float64))
    full = gray_img.copy().astype(np.uint8)
    full_matched = mapping[full]
    return full_matched.astype(np.float64), mapping


def debug_ncc(pid_a, pid_b, rot):
    path_a = os.path.join(COLOR_DIR, f'piece_{pid_a}.png')
    path_b = os.path.join(COLOR_DIR, f'piece_{pid_b}.png')
    img_a = np.array(Image.open(path_a).convert('RGBA'))
    img_b = np.array(Image.open(path_b).convert('RGBA'))

    vp = Path(VECTOR_PATH)
    corners_a = []
    corners_b_raw = []
    for j in range(4):
        with open(vp / f'side_{pid_a}_{j}.json') as f:
            corners_a.append(np.array(json.load(f)['vertices'][0], dtype=np.float64))
        with open(vp / f'side_{pid_b}_{j}.json') as f:
            corners_b_raw.append(np.array(json.load(f)['vertices'][0], dtype=np.float64))

    src_pts = np.array([corners_b_raw[(i + rot) % 4] for i in range(4)], dtype=np.float32)
    dst_pts = np.array(corners_a, dtype=np.float32)

    H, mask_h = cv2.findHomography(src_pts, dst_pts)
    print(f"Homography:\n{H}")
    print(f"Inliers: {mask_h.ravel().tolist() if mask_h is not None else 'None'}")

    h, w = img_a.shape[:2]
    warped_b = cv2.warpPerspective(img_b, H, (w, h))

    mask_a = img_a[:, :, 3] > 128
    mask_b = warped_b[:, :, 3] > 128
    overlap = mask_a & mask_b

    n_overlap = int(np.sum(overlap))
    n_a = int(np.sum(mask_a))
    n_b_warped = int(np.sum(mask_b))
    print(f"\nPixel counts: A={n_a}, B_warped={n_b_warped}, Overlap={n_overlap} ({n_overlap / min(n_a, n_b_warped) * 100:.1f}%)")

    gray_a = cv2.cvtColor(img_a[:, :, :3], cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(warped_b[:, :, :3], cv2.COLOR_RGB2GRAY)

    ga_raw = gray_a[overlap].astype(np.float64)
    gb_raw = gray_b[overlap].astype(np.float64)

    ncc_raw = compute_ncc_raw(ga_raw, gb_raw)

    gb_matched, _ = match_histogram(gb_raw, ga_raw)
    ncc_matched = compute_ncc_raw(ga_raw, gb_matched)

    gray_b_matched_full, _ = match_histogram_2d(gray_b, gray_a, overlap)
    gb_matched_full = gray_b_matched_full[overlap]

    edges_a = cv2.Canny(gray_a, 50, 150)
    edges_b_raw = cv2.Canny(gray_b, 50, 150)
    edges_b_matched = cv2.Canny(gray_b_matched_full.astype(np.uint8), 50, 150)

    ea = edges_a[overlap].astype(np.float64)
    eb_raw = edges_b_raw[overlap].astype(np.float64)
    eb_matched = edges_b_matched[overlap].astype(np.float64)
    ncc_edge_raw = compute_ncc_raw(ea, eb_raw)
    ncc_edge_matched = compute_ncc_raw(ea, eb_matched)

    ncc_final = max(ncc_matched, ncc_edge_matched)

    print(f"\nNCC comparison:")
    print(f"  Raw NCC gray:              {ncc_raw:.4f}")
    print(f"  HistMatch NCC gray:        {ncc_matched:.4f}")
    print(f"  Raw NCC edge:              {ncc_edge_raw:.4f}")
    print(f"  HistMatch NCC edge:        {ncc_edge_matched:.4f}")
    print(f"  Best (histMatch pipeline): {ncc_final:.4f}")

    diff_raw = np.abs(ga_raw - gb_raw)
    diff_matched = np.abs(ga_raw - gb_matched)

    PADDING = 16
    LABEL_H = 28
    CELL_W = w + PADDING
    CELL_H = h + LABEL_H + PADDING
    HIST_W, HIST_H = 400, 150
    SCATTER_W, SCATTER_H = 280, 280

    col_count = 3
    top_h = CELL_H * 2 + PADDING
    stats_h = max(HIST_H, SCATTER_H) + LABEL_H + PADDING
    canvas_w = col_count * CELL_W + PADDING + SCATTER_W + PADDING * 2
    canvas_h = top_h + stats_h + PADDING * 2 + 40

    canvas = Image.new('RGBA', (canvas_w, canvas_h), (30, 30, 38, 255))
    draw = ImageDraw.Draw(canvas)

    def put_img(x, y, img_arr, label, border=(100, 100, 100)):
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        if img_arr.shape[2] == 4:
            bg = np.full_like(img_arr[:, :, :3], 50)
            alpha = img_arr[:, :, 3:4].astype(np.float32) / 255.0
            comp = (img_arr[:, :, :3].astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
            img_arr = comp
        if img_arr.dtype != np.uint8:
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        draw.text((x + 4, y), label, fill=(255, 255, 200), font=FONT_L)
        canvas.paste(Image.fromarray(img_arr), (x, y + LABEL_H))
        draw.rectangle([x - 1, y + LABEL_H - 1, x + w, y + LABEL_H + h], outline=border, width=1)

    img_a_c = draw_corners(img_a, corners_a, (0, 255, 0))
    src_labeled = [corners_b_raw[(i + rot) % 4] for i in range(4)]
    img_b_c = draw_corners(img_b, src_labeled, (255, 128, 0))
    img_b_c2 = draw_corners(warped_b, corners_a, (255, 128, 0))

    row1_y = PADDING
    put_img(PADDING, row1_y, img_a_c, f"Piece #{pid_a} (reference)", (0, 200, 0))
    put_img(PADDING + CELL_W, row1_y, img_b_c, f"Piece #{pid_b} corners (rot={rot})", (255, 128, 0))
    put_img(PADDING + CELL_W * 2, row1_y, img_b_c2, f"#{pid_b} warped (overlay)", (200, 128, 0))

    row2_y = PADDING + CELL_H
    put_img(PADDING, row2_y, warped_b, f"#{pid_b} warped to #{pid_a}", (200, 128, 0))

    overlap_vis = np.zeros((h, w, 3), dtype=np.uint8)
    overlap_vis[mask_a] = [0, 0, 180]
    overlap_vis[mask_b] = [180, 0, 0]
    overlap_vis[overlap] = [0, 200, 0]
    put_img(PADDING + CELL_W, row2_y, overlap_vis,
            f"Overlap mask (G=both {n_overlap}px {n_overlap / min(n_a, n_b_warped) * 100:.1f}%)", (0, 200, 0))

    diff_vis = np.zeros((h, w), dtype=np.uint8)
    diff_vis[overlap] = np.clip(diff_raw, 0, 255).astype(np.uint8)
    diff_jet = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
    diff_jet[~overlap] = 40
    put_img(PADDING + CELL_W * 2, row2_y, diff_jet,
            f"|Diff| raw (mean={diff_raw.mean():.1f}) NCC={ncc_raw:.3f}", (200, 200, 0))

    stats_y = row2_y + CELL_H + PADDING
    draw.text((PADDING, stats_y), "Histogram Matching & NCC Analysis", fill=(255, 220, 100), font=FONT_XL)
    stats_y += 22

    hist1 = make_histogram_panel(ga_raw, gb_raw,
                                  f"Raw gray (NCC={ncc_raw:.3f})",
                                  w=HIST_W, h=HIST_H)
    hist2 = make_histogram_panel(ga_raw, gb_matched,
                                  f"After histMatch (NCC={ncc_matched:.3f})",
                                  w=HIST_W, h=HIST_H)
    hist3 = make_histogram_panel(ga_raw, gb_matched_full,
                                  f"HistMatch (full image) (NCC={ncc_matched:.3f})",
                                  w=HIST_W, h=HIST_H)

    canvas.paste(hist1, (PADDING, stats_y))
    canvas.paste(hist2, (PADDING + HIST_W + PADDING, stats_y))
    canvas.paste(hist3, (PADDING + (HIST_W + PADDING) * 2, stats_y))

    scatter1 = make_scatter_panel(ga_raw, gb_raw, "Raw gray", ncc_raw, SCATTER_W, SCATTER_H)
    scatter2 = make_scatter_panel(ga_raw, gb_matched, "HistMatch", ncc_matched, SCATTER_W, SCATTER_H)
    scatter3 = make_scatter_panel(ga_raw, gb_matched_full, "HistMatch full", ncc_matched, SCATTER_W, SCATTER_H)

    scatter_x = PADDING + col_count * CELL_W + PADDING
    canvas.paste(scatter1, (scatter_x, row1_y))
    canvas.paste(scatter2, (scatter_x, row1_y + SCATTER_H + PADDING))
    canvas.paste(scatter3, (scatter_x, row1_y + (SCATTER_H + PADDING) * 2))

    summary_y = canvas_h - 38
    summary = (f"#{pid_a} vs #{pid_b}  rot={rot}  |  "
               f"Raw NCC={ncc_raw:.4f}  HistMatch NCC={ncc_matched:.4f}  "
               f"Edge_raw={ncc_edge_raw:.4f}  Edge_matched={ncc_edge_matched:.4f}")
    draw.text((PADDING, summary_y), summary, fill=(255, 255, 100), font=FONT_L)

    result_color = (100, 255, 100) if ncc_final >= 0.70 else (255, 100, 100)
    draw.text((PADDING, summary_y + 18),
              f"Pipeline result: NCC={ncc_final:.4f}  {'PASS' if ncc_final >= 0.70 else 'FAIL (< 0.70)'}",
              fill=result_color, font=FONT_XL)

    out_path = os.path.join(OUTPUT_DIR, f'debug_ncc_{pid_a}_{pid_b}.png')
    canvas.save(out_path)
    print(f"\nSaved debug image to {out_path}")
    print(f"Image size: {canvas.width}x{canvas.height}")
    return ncc_final


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=int, default=30)
    parser.add_argument('--b', type=int, default=55)
    parser.add_argument('--rot', type=int, default=2)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Debug NCC for piece {args.a} vs {args.b} (rot={args.rot})")
    print("=" * 60)
    debug_ncc(args.a, args.b, args.rot)
