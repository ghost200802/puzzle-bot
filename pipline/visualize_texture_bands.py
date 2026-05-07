import os
import sys
import json
import math
import argparse

import numpy as np
import cv2

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR, VECTOR_DIR
from common.texture_verify import (
    load_side_data, load_color_image, extract_inner_band,
    compute_texture_richness, compute_seam_color_diff,
    compute_gradient_consistency, verify_match,
    INNER_OFFSET, BAND_WIDTH, N_SAMPLES,
)

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)
COLOR_PATH = os.path.join(OUTPUT_DIR, '2_piece_colors')
if not os.path.isdir(COLOR_PATH):
    COLOR_PATH = os.path.join(OUTPUT_DIR, VECTOR_DIR)


def _draw_band_on_image(img, side_vertices, piece_center, mask, color=(0, 255, 0)):
    vis = img.copy()
    h, w = vis.shape[:2]

    diffs = np.diff(side_vertices, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]
    if total_length < 1e-6:
        return vis

    sample_dists = np.linspace(0, total_length, N_SAMPLES)
    resampled = np.zeros((N_SAMPLES, 2))
    for i, d in enumerate(sample_dists):
        idx = np.searchsorted(cum_lengths, d, side='right') - 1
        idx = max(0, min(idx, len(seg_lengths) - 1))
        seg_start = cum_lengths[idx]
        seg_len = seg_lengths[idx]
        t = (d - seg_start) / seg_len if seg_len > 1e-6 else 0.0
        resampled[i] = side_vertices[idx] * (1 - t) + side_vertices[idx + 1] * t

    for i in range(N_SAMPLES):
        if i == 0:
            tangent = resampled[1] - resampled[0]
        elif i == N_SAMPLES - 1:
            tangent = resampled[-1] - resampled[-2]
        else:
            tangent = resampled[i + 1] - resampled[i - 1]

        tlen = np.sqrt(tangent[0] ** 2 + tangent[1] ** 2)
        if tlen < 1e-6:
            continue
        tangent = tangent / tlen

        normal = np.array([-tangent[1], tangent[0]])
        to_center = piece_center - resampled[i]
        if np.dot(normal, to_center) < 0:
            normal = -normal

        for d in range(INNER_OFFSET, INNER_OFFSET + BAND_WIDTH):
            pt = resampled[i] + normal * d
            px, py = int(round(pt[0])), int(round(pt[1]))
            if 0 <= py < h and 0 <= px < w:
                cv2.circle(vis, (px, py), 1, color, -1)

        start_pt = resampled[i] + normal * INNER_OFFSET
        end_pt = resampled[i] + normal * (INNER_OFFSET + BAND_WIDTH)
        cv2.line(vis,
                 (int(start_pt[0]), int(start_pt[1])),
                 (int(end_pt[0]), int(end_pt[1])),
                 color, 1)

    return vis


def visualize(pid_a, si_a, pid_b, si_b, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(_here, '..', 'output', 'texture_vis')
    os.makedirs(output_dir, exist_ok=True)

    side_a = load_side_data(DEDUPED_PATH, pid_a, si_a)
    side_b = load_side_data(DEDUPED_PATH, pid_b, si_b)
    color_a, mask_a = load_color_image(COLOR_PATH, pid_a)
    color_b, mask_b = load_color_image(COLOR_PATH, pid_b)

    if side_a is None or side_b is None:
        print("Error: side data not found")
        return
    if color_a is None or color_b is None:
        print("Error: color images not found")
        return

    vis_a = _draw_band_on_image(color_a, side_a['vertices'], side_a['piece_center'], mask_a, (0, 255, 0))
    vis_b = _draw_band_on_image(color_b, side_b['vertices'][::-1].copy(), side_b['piece_center'], mask_b, (0, 0, 255))

    cv2.imwrite(os.path.join(output_dir, f'band_a_{pid_a}_{si_a}.png'), vis_a)
    cv2.imwrite(os.path.join(output_dir, f'band_b_{pid_b}_{si_b}.png'), vis_b)

    band_a_colors, band_a_gray = extract_inner_band(
        color_a, side_a['vertices'], side_a['piece_center'], mask_a
    )
    band_b_colors, band_b_gray = extract_inner_band(
        color_b, side_b['vertices'][::-1].copy(), side_b['piece_center'], mask_b
    )

    n = min(len(band_a_colors), len(band_b_colors))
    if n < 5:
        print(f"Warning: only {n} valid sample points")
        if n == 0:
            return

    band_a_colors = band_a_colors[:n]
    band_b_colors = band_b_colors[:n]
    band_a_gray = band_a_gray[:n]
    band_b_gray = band_b_gray[:n]

    band_height = 40
    band_img_a = np.zeros((band_height, n * 4, 3), dtype=np.uint8)
    band_img_b = np.zeros((band_height, n * 4, 3), dtype=np.uint8)
    for i in range(n):
        c = band_a_colors[i].astype(np.uint8)
        band_img_a[:, i * 4:(i + 1) * 4] = c
    for i in range(n):
        c = band_b_colors[i].astype(np.uint8)
        band_img_b[:, i * 4:(i + 1) * 4] = c

    compare = np.vstack([band_img_a, band_img_b])
    cv2.imwrite(os.path.join(output_dir, f'band_compare_{pid_a}_{si_a}_vs_{pid_b}_{si_b}.png'), compare)

    color_diff_mean, color_diff_median = compute_seam_color_diff(band_a_colors, band_b_colors)
    tex_a = compute_texture_richness(band_a_gray)
    tex_b = compute_texture_richness(band_b_gray)
    grad_score = compute_gradient_consistency(band_a_gray, band_b_gray)

    result = verify_match(COLOR_PATH, DEDUPED_PATH, pid_a, si_a, pid_b, si_b)

    print(f"\n{'=' * 50}")
    print(f"  {pid_a}[{si_a}] -> {pid_b}[{si_b}]")
    print(f"{'=' * 50}")
    print(f"  Samples:      {n}")
    print(f"  Color ΔE:     mean={color_diff_mean:.2f}  median={color_diff_median:.2f}")
    print(f"  Texture A:    {tex_a:.4f}")
    print(f"  Texture B:    {tex_b:.4f}")
    print(f"  Texture level: {result['texture_level']}")
    print(f"  Grad score:   {grad_score}")
    print(f"  Reject:       {result['reject']}")
    print(f"  Reason:       {result['reason']}")
    print(f"{'=' * 50}")

    grad_plot_h = 200
    grad_plot_w = max(n * 4, 400)
    grad_img = np.ones((grad_plot_h, grad_plot_w, 3), dtype=np.uint8) * 255

    if n >= 3:
        grad_a = np.diff(band_a_gray)
        grad_b = np.diff(band_b_gray)
        all_grads = np.concatenate([grad_a, grad_b])
        max_abs = max(np.max(np.abs(all_grads)), 1)

        mid_y = grad_plot_h // 2
        cv2.line(grad_img, (0, mid_y), (grad_plot_w, mid_y), (200, 200, 200), 1)

        scale_y = (grad_plot_h // 2 - 10) / max_abs
        scale_x = grad_plot_w / max(len(grad_a) - 1, 1)

        for i in range(len(grad_a) - 1):
            x1 = int(i * scale_x)
            x2 = int((i + 1) * scale_x)
            y1 = int(mid_y - grad_a[i] * scale_y)
            y2 = int(mid_y - grad_a[i + 1] * scale_y)
            cv2.line(grad_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for i in range(len(grad_b) - 1):
            x1 = int(i * scale_x)
            x2 = int((i + 1) * scale_x)
            y1 = int(mid_y - grad_b[i] * scale_y)
            y2 = int(mid_y - grad_b[i + 1] * scale_y)
            cv2.line(grad_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(output_dir, f'gradient_{pid_a}_{si_a}_vs_{pid_b}_{si_b}.png'), grad_img)

    delta_e_plot_h = 200
    delta_e_plot_w = max(n * 4, 400)
    delta_e_img = np.ones((delta_e_plot_h, delta_e_plot_w, 3), dtype=np.uint8) * 255

    if n > 0:
        a_bgr = band_a_colors.reshape(1, -1, 3).astype(np.uint8)
        b_bgr = band_b_colors.reshape(1, -1, 3).astype(np.uint8)
        a_lab = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2Lab).astype(np.float64).reshape(-1, 3)
        b_lab = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2Lab).astype(np.float64).reshape(-1, 3)
        delta_e_vals = np.sqrt(np.sum((a_lab - b_lab) ** 2, axis=1))

        max_de = max(np.max(delta_e_vals), 1)
        scale_y = (delta_e_plot_h - 20) / max_de
        scale_x = delta_e_plot_w / max(n - 1, 1)

        threshold_line_y = int(delta_e_plot_h - 10 - 60.0 * scale_y)
        cv2.line(delta_e_img, (0, threshold_line_y),
                 (delta_e_plot_w, threshold_line_y), (0, 0, 255), 1)
        cv2.putText(delta_e_img, "60", (5, threshold_line_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        for i in range(n - 1):
            x1 = int(i * scale_x)
            x2 = int((i + 1) * scale_x)
            y1 = int(delta_e_plot_h - 10 - delta_e_vals[i] * scale_y)
            y2 = int(delta_e_plot_h - 10 - delta_e_vals[i + 1] * scale_y)
            cv2.line(delta_e_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite(os.path.join(output_dir, f'delta_e_{pid_a}_{si_a}_vs_{pid_b}_{si_b}.png'), delta_e_img)

    print(f"\n  Visualization saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize texture bands for a match pair')
    parser.add_argument('pid_a', type=int, help='Piece A ID')
    parser.add_argument('si_a', type=int, help='Piece A side index (0-3)')
    parser.add_argument('pid_b', type=int, help='Piece B ID')
    parser.add_argument('si_b', type=int, help='Piece B side index (0-3)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory')

    args = parser.parse_args()
    visualize(args.pid_a, args.si_a, args.pid_b, args.si_b, args.output)


if __name__ == '__main__':
    main()
