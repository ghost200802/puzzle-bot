import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))

import importlib
import common.texture_verify
importlib.reload(common.texture_verify)
from common.texture_verify import (
    load_side_data, load_color_image, extract_inner_band,
    compute_pattern_ncc, _resample_polyline, _resample_by_chord
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

# Test chord resampling
verts_a = side_a['vertices']
p1_a = verts_a[0]
p2_a = verts_a[-1]
chord_a = p2_a - p1_a
chord_len_a = np.linalg.norm(chord_a)
chord_dir_a = chord_a / chord_len_a

arc_pts = _resample_polyline(verts_a, 30)
chord_pts = _resample_by_chord(verts_a, 30)

proj_arc = [np.dot(v - p1_a, chord_dir_a) for v in arc_pts]
proj_chord = [np.dot(v - p1_a, chord_dir_a) for v in chord_pts]

print("Arc-length vs Chord-based projections for Side A:")
print(f"  {'idx':>3}  {'arc_proj':>10}  {'chord_proj':>10}")
for i in range(30):
    print(f"  {i:>3}  {proj_arc[i]:>10.2f}  {proj_chord[i]:>10.2f}")

# Now compute NCC with new method
band_a_colors, band_a_gray = extract_inner_band(
    color_a, side_a['vertices'], side_a['piece_center'], mask_a
)
vertices_b_flipped = side_b['vertices'][::-1].copy()
band_b_flip_colors, band_b_flip_gray = extract_inner_band(
    color_b, vertices_b_flipped, side_b['piece_center'], mask_b
)
band_b_orig_colors, band_b_orig_gray = extract_inner_band(
    color_b, side_b['vertices'], side_b['piece_center'], mask_b
)

n = min(len(band_a_gray), len(band_b_flip_gray), len(band_b_orig_gray))
print(f"\nBand A samples: {len(band_a_gray)}, Band B flip: {len(band_b_flip_gray)}, Band B orig: {len(band_b_orig_gray)}, common: {n}")

ncc_flip = compute_pattern_ncc(band_a_gray[:n], band_b_flip_gray[:n])
ncc_orig = compute_pattern_ncc(band_a_gray[:n], band_b_orig_gray[:n])

print(f"\nNCC (B flipped, chord-based):  {ncc_flip:.4f}")
print(f"NCC (B original, chord-based): {ncc_orig:.4f}")

# Also check chord projections of the band samples
band_a_proj = [np.dot(p - p1_a, chord_dir_a) for p in arc_pts[:n]]
print(f"\nBand A first 10 gray: {band_a_gray[:min(10,n)].tolist()}")
print(f"Band B flip first 10 gray: {band_b_flip_gray[:min(10,n)].tolist()}")
