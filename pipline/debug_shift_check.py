import os
import sys
import math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import importlib
import common.texture_verify as tv
importlib.reload(tv)
from common.texture_verify import (
    load_side_data, load_color_image, extract_inner_band,
    compute_pattern_ncc, _resample_polyline,
    _find_corresponding_points_on_edge, _compute_transform,
    _apply_transform, _apply_inverse_transform, N_SAMPLES
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '3_vector')
COLOR_PATH = os.path.join(OUTPUT_DIR, '2_piece_colors')

side_a = load_side_data(DEDUPED_PATH, 4, 3)
side_b = load_side_data(DEDUPED_PATH, 137, 3)
color_a, mask_a = load_color_image(COLOR_PATH, 4)
color_b, mask_b = load_color_image(COLOR_PATH, 137)

verts_a = side_a['vertices']
verts_b = side_b['vertices']
verts_bf = verts_b[::-1].copy()

sample_a = _resample_polyline(verts_a, N_SAMPLES)
src_mid, tgt_mid, rot = _compute_transform(verts_b, verts_a)
verts_bf_aligned = _apply_transform(verts_bf, src_mid, tgt_mid, rot)
corr_aligned = _find_corresponding_points_on_edge(sample_a, verts_bf_aligned)
corr_orig = _apply_inverse_transform(corr_aligned, src_mid, tgt_mid, rot)

# Check distance between A[i] and B[i], A[i] and B[i+1], A[i] and B[i-1]
print("Distance check: A[i] vs B[i], B[i+1], B[i-1]")
print(f"  {'i':>3}  {'A-B[i]':>8}  {'A-B[i-1]':>10}  {'A-B[i+1]':>10}  {'best':>5}")
for i in range(N_SAMPLES):
    d_same = np.linalg.norm(sample_a[i] - corr_aligned[i])
    d_prev = np.linalg.norm(sample_a[i] - corr_aligned[i-1]) if i > 0 else 999
    d_next = np.linalg.norm(sample_a[i] - corr_aligned[i+1]) if i < N_SAMPLES-1 else 999
    best = 'same'
    if d_prev < d_same and d_prev < d_next:
        best = 'i-1'
    elif d_next < d_same:
        best = 'i+1'
    print(f"  [{i:2d}]  {d_same:>8.1f}  {d_prev:>10.1f}  {d_next:>10.1f}  {best:>5}")

# Now compute NCC with and without shift
ba_c, ba_g = extract_inner_band(color_a, sample_a, mask_a, normal_side='left')
bb_c, bb_g = extract_inner_band(color_b, corr_orig, mask_b, normal_side='right', edge_vertices=verts_bf)

n = min(len(ba_g), len(bb_g))
print(f"\nNCC (no manual shift): {compute_pattern_ncc(ba_g[:n], bb_g[:n]):.4f}")

# Try manual shift of +1
print(f"NCC (shift A by +1, skip A[0]): {compute_pattern_ncc(ba_g[1:n], bb_g[:n-1]):.4f}")
print(f"NCC (shift B by +1, skip B[0]): {compute_pattern_ncc(ba_g[:n-1], bb_g[1:n]):.4f}")

# Try all shifts
print("\nNCC per shift value:")
a = ba_g[:n].astype(np.float64)
b = bb_g[:n].astype(np.float64)
for s in range(-5, 6):
    if s >= 0:
        a_sub = a[s:]
        b_sub = b[:n - s]
    else:
        a_sub = a[:n + s]
        b_sub = b[-s:]
    m = len(a_sub)
    if m < 5:
        continue
    a_mean = np.mean(a_sub)
    b_mean = np.mean(b_sub)
    a_c = a_sub - a_mean
    b_c = b_sub - b_mean
    a_norm = np.sqrt(np.sum(a_c ** 2))
    b_norm = np.sqrt(np.sum(b_c ** 2))
    if a_norm < 1e-6 or b_norm < 1e-6:
        continue
    ncc = float(np.dot(a_c, b_c) / (a_norm * b_norm))
    print(f"  shift={s:+d}: NCC={ncc:.4f} (compare A[{-s if s<0 else 0}:] vs B[{s if s>0 else 0}:])")
