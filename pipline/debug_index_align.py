import os
import sys
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

# Extract bands individually per point to track which ones succeed
from common.texture_verify import INNER_OFFSET, BAND_WIDTH, SAMPLE_RADIUS

def extract_single_band(color_image, pos, tangent, normal_side, binary_mask, edge_vertices=None):
    h, w = color_image.shape[:2]
    if edge_vertices is not None:
        from common.texture_verify import _edge_tangent_at
        tangent = _edge_tangent_at(edge_vertices, pos)
    tlen = np.linalg.norm(tangent)
    if tlen < 1e-6:
        return None
    tangent = tangent / tlen
    if normal_side == 'left':
        normal = np.array([-tangent[1], tangent[0]])
    else:
        normal = np.array([tangent[1], -tangent[0]])
    
    colors = []
    for d in range(INNER_OFFSET, INNER_OFFSET + BAND_WIDTH):
        pt = pos + normal * d
        px_c, py_c = int(round(pt[0])), int(round(pt[1]))
        patch_pixels = []
        for dy in range(-SAMPLE_RADIUS, SAMPLE_RADIUS + 1):
            for dx in range(-SAMPLE_RADIUS, SAMPLE_RADIUS + 1):
                py = py_c + dy
                px = px_c + dx
                if 0 <= py < h and 0 <= px < w and binary_mask[py, px] > 0:
                    patch_pixels.append(color_image[py, px].astype(np.float64))
        if patch_pixels:
            colors.append(np.mean(patch_pixels, axis=0))
    if not colors:
        return None
    avg = np.mean(colors, axis=0)
    gray = 0.114 * avg[0] + 0.587 * avg[1] + 0.299 * avg[2]
    return gray

print("Per-point band extraction status:")
a_grays = []
b_grays = []
for i in range(N_SAMPLES):
    # A
    tangents = []
    if i > 0:
        d = sample_a[i] - sample_a[i-1]
        dl = np.linalg.norm(d)
        if dl > 1e-6: tangents.append(d / dl)
    if i < N_SAMPLES - 1:
        d = sample_a[i+1] - sample_a[i]
        dl = np.linalg.norm(d)
        if dl > 1e-6: tangents.append(d / dl)
    t_a = np.mean(tangents, axis=0) if tangents else np.array([0,0])
    ga = extract_single_band(color_a, sample_a[i], t_a, 'left', mask_a)
    
    # B
    gb = extract_single_band(color_b, corr_orig[i], None, 'right', mask_b, edge_vertices=verts_bf)
    
    a_ok = ga is not None
    b_ok = gb is not None
    status = "OK" if (a_ok and b_ok) else f"{'A' if not a_ok else ''}{'B' if not b_ok else ''} FAIL"
    print(f"  [{i:2d}] A={'%6.1f'%ga if a_ok else ' FAIL'}  B={'%6.1f'%gb if b_ok else ' FAIL'}  {status}")
    
    if a_ok: a_grays.append(ga)
    if b_ok: b_grays.append(gb)

# Current method: just concatenate all valid, losing index alignment
ba_c, ba_g = extract_inner_band(color_a, sample_a, mask_a, normal_side='left')
bb_c, bb_g = extract_inner_band(color_b, corr_orig, mask_b, normal_side='right', edge_vertices=verts_bf)
print(f"\nextract_inner_band: A={len(ba_g)} B={len(bb_g)}")
print(f"NCC (current, misaligned): {compute_pattern_ncc(ba_g, bb_g):.4f}")

# Now compute with index-aligned data
n_aligned = min(len(a_grays), len(b_grays))
if n_aligned >= 10:
    a_arr = np.array(a_grays[:n_aligned])
    b_arr = np.array(b_grays[:n_aligned])
    print(f"\nIndex-aligned NCC: {compute_pattern_ncc(a_arr, b_arr):.4f}")
