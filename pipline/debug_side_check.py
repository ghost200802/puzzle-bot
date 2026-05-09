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
    _resample_polyline, _find_corresponding_points_on_edge,
    _compute_transform, _apply_transform, _apply_inverse_transform,
    N_SAMPLES, INNER_OFFSET, BAND_WIDTH
)

from config import get_output_dir, get_vector_path, get_color_path

OUTPUT_DIR = get_output_dir()
DEDUPED_PATH = get_vector_path()
COLOR_PATH = get_color_path()

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

# Test each side for A
for side in ['left', 'right']:
    colors, gray = extract_inner_band(color_a, sample_a, mask_a, normal_side=side)
    print(f"A normal_side='{side}': {len(gray)} samples, gray range=[{gray.min():.1f}, {gray.max():.1f}]" if len(gray) > 0 else f"A normal_side='{side}': 0 samples")

# Test each side for B
for side in ['left', 'right']:
    colors, gray = extract_inner_band(color_b, corr_orig, mask_b, normal_side=side, edge_vertices=verts_bf)
    print(f"B normal_side='{side}': {len(gray)} samples, gray range=[{gray.min():.1f}, {gray.max():.1f}]" if len(gray) > 0 else f"B normal_side='{side}': 0 samples")

# Check: what side is correct?
# A's tangent goes CCW around piece -> piece is on RIGHT side of tangent
# B's tangent (flipped) goes CW around piece -> piece is on LEFT side of tangent
# But we verified visually: A uses (ty, -tx) = right, B uses (-ty, tx) = left

# Let's verify by checking dot product with to_center for a few points
print("\nVerify directions:")
center_a = side_a['piece_center']
center_b = side_b['piece_center']

for i in [0, 5, 15, 25]:
    pos = sample_a[i]
    tangent = sample_a[min(i+1, N_SAMPLES-1)] - sample_a[max(i-1, 0)]
    tl = np.linalg.norm(tangent)
    if tl > 1e-6:
        tangent = tangent / tl
    left_n = np.array([-tangent[1], tangent[0]])
    right_n = np.array([tangent[1], -tangent[0]])
    to_c = center_a - pos
    print(f"  A[{i}] left_dot={np.dot(left_n, to_c):+.1f} right_dot={np.dot(right_n, to_c):+.1f} -> {'right' if np.dot(right_n, to_c) > 0 else 'left'} is toward center")

for i in [0, 5, 15, 25]:
    pos = corr_orig[i]
    tangent_raw = corr_orig[min(i+1, N_SAMPLES-1)] - corr_orig[max(i-1, 0)]
    tl = np.linalg.norm(tangent_raw)
    if tl > 1e-6:
        tangent_raw = tangent_raw / tl
    left_n = np.array([-tangent_raw[1], tangent_raw[0]])
    right_n = np.array([tangent_raw[1], -tangent_raw[0]])
    to_c = center_b - pos
    print(f"  B[{i}] left_dot={np.dot(left_n, to_c):+.1f} right_dot={np.dot(right_n, to_c):+.1f} -> {'right' if np.dot(right_n, to_c) > 0 else 'left'} is toward center")
