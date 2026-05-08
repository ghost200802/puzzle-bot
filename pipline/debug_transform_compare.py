import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

import importlib
import common.texture_verify as tv
importlib.reload(tv)
from common.texture_verify import (
    load_side_data, load_color_image, extract_inner_band,
    compute_pattern_ncc, _resample_polyline,
    _find_corresponding_points_on_edge, _compute_transform,
    _apply_transform, _apply_inverse_transform, N_SAMPLES
)

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '3_vector')
COLOR_PATH = os.path.join(OUTPUT_DIR, '2_piece_colors')

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)
color_a, mask_a = load_color_image(COLOR_PATH, PID_A)
color_b, mask_b = load_color_image(COLOR_PATH, PID_B)

verts_a = side_a['vertices']
verts_b = side_b['vertices']
verts_bf = verts_b[::-1].copy()

sample_a = _resample_polyline(verts_a, N_SAMPLES)

# Method 1: transform from B_flipped to A (original debug_corr_test)
sm1, tm1, r1 = _compute_transform(verts_bf, verts_a)
bf_al1 = _apply_transform(verts_bf, sm1, tm1, r1)
corr_al1 = _find_corresponding_points_on_edge(sample_a, bf_al1)
corr_orig1 = _apply_inverse_transform(corr_al1, sm1, tm1, r1)

# Method 2: transform from B original to A (current verify_match)
sm2, tm2, r2 = _compute_transform(verts_b, verts_a)
bf_al2 = _apply_transform(verts_bf, sm2, tm2, r2)
corr_al2 = _find_corresponding_points_on_edge(sample_a, bf_al2)
corr_orig2 = _apply_inverse_transform(corr_al2, sm2, tm2, r2)

for method, corr_orig in [("B_flipped->A", corr_orig1), ("B_original->A", corr_orig2)]:
    ba_c, ba_g = extract_inner_band(color_a, sample_a, side_a['piece_center'], mask_a)
    bb_c, bb_g = extract_inner_band(color_b, corr_orig, side_b['piece_center'], mask_b)
    n = min(len(ba_g), len(bb_g))
    ncc = compute_pattern_ncc(ba_g[:n], bb_g[:n])
    print(f"\nMethod: {method}")
    print(f"  NCC: {ncc:.4f}, samples: {n}")
    print(f"  corr_orig first 5: {corr_orig[:5].tolist()}")
    print(f"  corr_orig[8:11]: {corr_orig[8:11].tolist()}")

    dists = [np.linalg.norm(sample_a[i] - (_apply_transform(corr_orig[i:i+1], sm1 if method=="B_flipped->A" else sm2, 
                          tm1 if method=="B_flipped->A" else tm2,
                          r1 if method=="B_flipped->A" else r2))[0]) for i in range(N_SAMPLES)]
    print(f"  Distances: mean={np.mean(dists):.1f}, min={np.min(dists):.1f}, max={np.max(dists):.1f}")
