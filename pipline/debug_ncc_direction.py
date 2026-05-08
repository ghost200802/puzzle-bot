import os
import sys
import json
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.texture_verify import (
    load_side_data, load_color_image, extract_inner_band,
    compute_pattern_ncc, _resample_polyline
)

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '3_vector')
COLOR_PATH = os.path.join(OUTPUT_DIR, '2_piece_colors')

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

print("=" * 70)
print("NCC Direction Analysis: Piece 4[3] <-> Piece 137[3]")
print("=" * 70)

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)

print("\n[1] Side vertex endpoints and geometry")
print(f"  Piece {PID_A} side {SI_A}:")
print(f"    p1 = {side_a['vertices'][0].tolist()}")
print(f"    p2 = {side_a['vertices'][-1].tolist()}")
print(f"    center = {side_a['piece_center'].tolist()}")
print(f"    vertex count = {len(side_a['vertices'])}")

print(f"\n  Piece {PID_B} side {SI_B}:")
print(f"    p1 = {side_b['vertices'][0].tolist()}")
print(f"    p2 = {side_b['vertices'][-1].tolist()}")
print(f"    center = {side_b['piece_center'].tolist()}")
print(f"    vertex count = {len(side_b['vertices'])}")

print("\n[2] Geometric alignment analysis")
print("  When two pieces connect, based on _can_potentially_match:")
print("    adj_map = { (si-1)%4: (sj+1)%4, (si+1)%4: (sj-1)%4 }")
print(f"    A[{SI_A}]: adjacent sides are A[{(SI_A-1)%4}] and A[{(SI_A+1)%4}]")
print(f"    B[{SI_B}]: adjacent sides are B[{(SI_B-1)%4}] and B[{(SI_B+1)%4}]")
print(f"    A[{(SI_A-1)%4}] maps to B[{(SI_B+1)%4}]  (A side before si -> B side after sj)")
print(f"    A[{(SI_A+1)%4}] maps to B[{(SI_B-1)%4}]  (A side after si -> B side before sj)")
print()
print("  Sides are ordered counterclockwise (CCW):")
print("    Side N ends where Side N+1 begins (shared vertex)")
print(f"    A[{SI_A}].p1 = end of A[{(SI_A-1)%4}], A[{SI_A}].p2 = start of A[{(SI_A+1)%4}]")
print(f"    B[{SI_B}].p1 = end of B[{(SI_B-1)%4}], B[{SI_B}].p2 = start of B[{(SI_B+1)%4}]")
print()
print("  Alignment mapping:")
print(f"    A[{(SI_A-1)%4}] (ends at A[{SI_A}].p1) <-> B[{(SI_B+1)%4}] (starts at B[{SI_B}].p2)")
print(f"    => A[{SI_A}].p1 aligns with B[{SI_B}].p2")
print(f"    A[{(SI_A+1)%4}] (starts at A[{SI_A}].p2) <-> B[{(SI_B-1)%4}] (ends at B[{SI_B}].p1)")
print(f"    => A[{SI_A}].p2 aligns with B[{SI_B}].p1")

a_p1 = side_a['vertices'][0]
a_p2 = side_a['vertices'][-1]
b_p1 = side_b['vertices'][0]
b_p2 = side_b['vertices'][-1]
print(f"\n  Concrete points:")
print(f"    A.p1 = {a_p1.tolist()} should align with B.p2 = {b_p2.tolist()}")
print(f"    A.p2 = {a_p2.tolist()} should align with B.p1 = {b_p1.tolist()}")

print("\n[3] Inner band traversal direction analysis")
print("  Band A: traverses from A.p1 to A.p2 (CCW around piece)")
print("  Band B (current code, flipped): traverses from B.p2 to B.p1")
print("    B.p2 = first point of flipped = spatially aligns with A.p1 ✓")
print("    B.p1 = last point of flipped = spatially aligns with A.p2 ✓")
print("  => Band A[0] and Band B_flipped[0] are at the same spatial position")
print("  => Current flipping is GEOMETRICALLY CORRECT")

print("\n[4] Verify with normal direction")
center_a = side_a['piece_center']
center_b = side_b['piece_center']

resampled_a = _resample_polyline(side_a['vertices'], 30)
resampled_b_orig = _resample_polyline(side_b['vertices'], 30)
resampled_b_flip = _resample_polyline(side_b['vertices'][::-1], 30)

def compute_normal(resampled, i, center):
    n = len(resampled)
    if i == 0:
        tangent = resampled[1] - resampled[0]
    elif i == n - 1:
        tangent = resampled[-1] - resampled[-2]
    else:
        tangent = resampled[i + 1] - resampled[i - 1]
    tlen = np.linalg.norm(tangent)
    if tlen < 1e-6:
        return None
    tangent = tangent / tlen
    normal = np.array([-tangent[1], tangent[0]])
    to_center = center - resampled[i]
    if np.dot(normal, to_center) < 0:
        normal = -normal
    return normal

print(f"  Sample normals at start/middle/end:")
for label, resampled, center in [
    ("A (piece 4)", resampled_a, center_a),
    ("B original (piece 137)", resampled_b_orig, center_b),
    ("B flipped (piece 137)", resampled_b_flip, center_b),
]:
    print(f"\n  {label}:")
    for idx in [0, 15, 29]:
        n = compute_normal(resampled, idx, center)
        if n is not None:
            print(f"    [{idx}] pos={resampled[idx].tolist()}, normal={n.tolist()}")

print("\n[5] Load color images and extract bands")
color_a, mask_a = load_color_image(COLOR_PATH, PID_A)
color_b, mask_b = load_color_image(COLOR_PATH, PID_B)

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
print(f"  Band A samples: {len(band_a_gray)}")
print(f"  Band B (flipped) samples: {len(band_b_flip_gray)}")
print(f"  Band B (original) samples: {len(band_b_orig_gray)}")
print(f"  Common length: {n}")

band_a = band_a_gray[:n]
band_b_f = band_b_flip_gray[:n]
band_b_o = band_b_orig_gray[:n]

print("\n[6] Gray value comparison (first 15 samples)")
print(f"  {'idx':>3}  {'Band A':>8}  {'B flipped':>10}  {'B original':>10}  {'B orig reversed':>14}")
for i in range(min(15, n)):
    print(f"  {i:>3}  {band_a[i]:>8.1f}  {band_b_f[i]:>10.1f}  {band_b_o[i]:>10.1f}  {band_b_o[n-1-i]:>14.1f}")

print("\n[7] NCC computation for all configurations")
configs = {
    "A vs B_flipped (CURRENT CODE)": (band_a, band_b_f),
    "A vs B_original": (band_a, band_b_o),
    "A vs B_original_reversed": (band_a, band_b_o[::-1]),
}

for desc, (ba, bb) in configs.items():
    ncc = compute_pattern_ncc(ba, bb)
    pearson = np.corrcoef(ba, bb)[0, 1]
    print(f"  {desc}:")
    print(f"    NCC (with shift search) = {ncc:.4f}")
    print(f"    Pearson (no shift)      = {pearson:.4f}")

print("\n[8] Direct correlation: Band A vs Band B_flipped (element-wise)")
diff = band_a - band_b_f
print(f"  Mean diff: {np.mean(diff):.2f}")
print(f"  Std diff:  {np.std(diff):.2f}")
print(f"  Direction: A tends {'higher' if np.mean(diff) > 0 else 'lower'} than B_flipped")

a_trend = "increasing" if (band_a[-1] - band_a[0]) > 0 else "decreasing"
bf_trend = "increasing" if (band_b_f[-1] - band_b_f[0]) > 0 else "decreasing"
bo_trend = "increasing" if (band_b_o[-1] - band_b_o[0]) > 0 else "decreasing"
print(f"  Band A trend: {a_trend} ({band_a[0]:.1f} -> {band_a[-1]:.1f})")
print(f"  Band B_flipped trend: {bf_trend} ({band_b_f[0]:.1f} -> {band_b_f[-1]:.1f})")
print(f"  Band B_original trend: {bo_trend} ({band_b_o[0]:.1f} -> {band_b_o[-1]:.1f})")

print("\n[9] Checking if B_original reversed matches better spatially")
print("  B_original goes B.p1 -> B.p2 (CCW)")
print("  B_original_reversed goes B.p2 -> B.p1")
print("  B_flipped vertices go B.p2 -> B.p1")
print("  Both B_original_reversed and B_flipped traverse in same spatial direction")
print("  But the NORMALS may differ because:")
print("    B_flipped: tangent from B.p2->B.p1, normal auto-corrected to center")
print("    B_original_reversed: band extracted along B.p1->B.p2, then reversed")

ncc_orig_rev = compute_pattern_ncc(band_a, band_b_o[::-1])
print(f"\n  NCC(A, B_original_reversed) = {ncc_orig_rev:.4f}")
print(f"  NCC(A, B_flipped)           = {compute_pattern_ncc(band_a, band_b_f):.4f}")
print(f"  Difference: {abs(ncc_orig_rev - compute_pattern_ncc(band_a, band_b_f)):.4f}")

print("\n[10] Key insight: flipped vertices vs reversed band")
print("  The critical difference:")
print("    B_flipped: vertices reversed BEFORE band extraction")
print("      -> tangent direction changes -> normal rotates -> but to_center corrects it")
print("      -> band is extracted along reversed path, inward from REVERSED edge")
print("    B_original_reversed: band extracted THEN reversed")
print("      -> band is extracted along original path, inward from ORIGINAL edge")
print("      -> only the ORDER of samples changes, not the pixel content")

band_b_f_start = band_b_f[:5]
band_b_o_rev_end = band_b_o[::-1][:5]
print(f"\n  B_flipped first 5:     {band_b_f_start.tolist()}")
print(f"  B_orig_rev first 5:    {band_b_o_rev_end.tolist()}")
print(f"  Are they same? {np.allclose(band_b_f, band_b_o[::-1], atol=0.1)}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
