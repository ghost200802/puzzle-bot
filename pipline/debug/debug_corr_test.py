import os
import sys
import math
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))

import importlib
import common.texture_verify as tv
importlib.reload(tv)
from common.texture_verify import (
    load_side_data, load_color_image, extract_inner_band,
    compute_pattern_ncc, _resample_polyline, _find_corresponding_points_on_edge,
    _compute_transform, _apply_transform, _apply_inverse_transform,
    N_SAMPLES
)

from pipline.config import get_output_dir, get_vector_path, get_color_path

OUTPUT_DIR = get_output_dir()
DEDUPED_PATH = get_vector_path()
COLOR_PATH = get_color_path()

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)
color_a, mask_a = load_color_image(COLOR_PATH, PID_A)
color_b, mask_b = load_color_image(COLOR_PATH, PID_B)

verts_a = side_a['vertices']
verts_bf = side_b['vertices'][::-1].copy()

sample_a = _resample_polyline(verts_a, N_SAMPLES)

src_mid, tgt_mid, rot = _compute_transform(verts_bf, verts_a)
verts_bf_aligned = _apply_transform(verts_bf, src_mid, tgt_mid, rot)

corr_aligned = _find_corresponding_points_on_edge(sample_a, verts_bf_aligned)

dists = [np.linalg.norm(sample_a[i] - corr_aligned[i]) for i in range(N_SAMPLES)]

print("Correspondence distances (arc-length A -> closest on B aligned):")
for i in range(N_SAMPLES):
    print(f"  [{i:2d}] dist={dists[i]:.1f}")
print(f"  Mean: {np.mean(dists):.1f}, Median: {np.median(dists):.1f}, Max: {np.max(dists):.1f}")
print(f"  Points with dist<10:  {sum(1 for d in dists if d < 10)}/{N_SAMPLES}")
print(f"  Points with dist<20:  {sum(1 for d in dists if d < 20)}/{N_SAMPLES}")
print(f"  Points with dist<50:  {sum(1 for d in dists if d < 50)}/{N_SAMPLES}")

corr_original = _apply_inverse_transform(corr_aligned, src_mid, tgt_mid, rot)

band_a_colors, band_a_gray = extract_inner_band(color_a, sample_a, side_a['piece_center'], mask_a)
band_b_colors, band_b_gray = extract_inner_band(color_b, corr_original, side_b['piece_center'], mask_b)

n = min(len(band_a_gray), len(band_b_gray))
print(f"\nBand samples: A={len(band_a_gray)}, B={len(band_b_gray)}, common={n}")

ncc = compute_pattern_ncc(band_a_gray[:n], band_b_gray[:n])
print(f"NCC (correspondence-based): {ncc:.4f}")

print(f"\nGray values comparison (first 10):")
print(f"  {'idx':>3}  {'A_gray':>8}  {'B_gray':>8}")
for i in range(min(10, n)):
    print(f"  {i:>3}  {band_a_gray[i]:>8.1f}  {band_b_gray[i]:>8.1f}")
