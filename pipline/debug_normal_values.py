import os
import sys
import math
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

import importlib
import common.texture_verify as tv
importlib.reload(tv)
from common.texture_verify import (
    load_side_data, _resample_polyline,
    _find_corresponding_points_on_edge, _compute_transform,
    _apply_transform, _apply_inverse_transform, N_SAMPLES
)

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '3_vector')

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)

verts_a = side_a['vertices']
verts_b = side_b['vertices']
verts_bf = verts_b[::-1].copy()

sample_a = _resample_polyline(verts_a, N_SAMPLES)

src_mid, tgt_mid, rot = _compute_transform(verts_b, verts_a)
verts_bf_aligned = _apply_transform(verts_bf, src_mid, tgt_mid, rot)
corr_aligned = _find_corresponding_points_on_edge(sample_a, verts_bf_aligned)
corr_orig = _apply_inverse_transform(corr_aligned, src_mid, tgt_mid, rot)

center_a = side_a['piece_center']
center_b = side_b['piece_center']

print("=== B correspondence points (corr_orig) - checking ordering ===")
print(f"B center: {center_b}")
for i in [6,7,8,9,10,11,12, 15,16,17,18,19,20]:
    p = corr_orig[i]
    to_c = center_b - p
    dist_to_center = np.linalg.norm(to_c)
    angle_to_center = math.degrees(math.atan2(to_c[1], to_c[0]))

    if i > 0:
        prev_d = corr_orig[i] - corr_orig[i-1]
        prev_dl = np.linalg.norm(prev_d)
        prev_angle = math.degrees(math.atan2(prev_d[1], prev_d[0]))
    else:
        prev_angle = 0
        prev_dl = 0

    if i < N_SAMPLES - 1:
        next_d = corr_orig[i+1] - corr_orig[i]
        next_dl = np.linalg.norm(next_d)
        next_angle = math.degrees(math.atan2(next_d[1], next_d[0]))
    else:
        next_angle = 0
        next_dl = 0

    # Compute tangent (average of prev and next directions)
    tangents = []
    if i > 0 and prev_dl > 1e-6:
        tangents.append(prev_d / prev_dl)
    if i < N_SAMPLES - 1 and next_dl > 1e-6:
        tangents.append(next_d / next_dl)
    
    if tangents:
        tangent = np.mean(tangents, axis=0)
        tl = np.linalg.norm(tangent)
        if tl > 1e-6:
            tangent = tangent / tl
        normal = np.array([-tangent[1], tangent[0]])
        n_angle = math.degrees(math.atan2(normal[1], normal[0]))
        dot = np.dot(normal, to_c)
        flipped = dot < 0
        if flipped:
            normal = -normal
            n_angle = math.degrees(math.atan2(normal[1], normal[0]))
    else:
        tangent = np.array([0,0])
        normal = np.array([0,0])
        n_angle = 0
        dot = 0
        flipped = False

    t_angle = math.degrees(math.atan2(tangent[1], tangent[0])) if np.linalg.norm(tangent) > 1e-6 else 0

    marker = " <<<" if i in [8,9,10,16,17,18] else ""
    print(f"\n  [{i}]{marker}")
    print(f"    pos=({p[0]:.1f}, {p[1]:.1f})")
    print(f"    prev_dir: angle={prev_angle:+.1f}° len={prev_dl:.1f}")
    print(f"    next_dir: angle={next_angle:+.1f}° len={next_dl:.1f}")
    print(f"    tangent:  angle={t_angle:+.1f}°")
    print(f"    normal:   angle={n_angle:+.1f}°  dot_with_center={dot:+.3f}  flipped={flipped}")
    print(f"    to_center: angle={angle_to_center:+.1f}° dist={dist_to_center:.1f}")

print("\n=== Also check A points for comparison ===")
print(f"A center: {center_a}")
for i in [8,9,10,16,17,18]:
    p = sample_a[i]
    to_c = center_a - p
    angle_to_center = math.degrees(math.atan2(to_c[1], to_c[0]))

    tangents = []
    if i > 0:
        d = sample_a[i] - sample_a[i-1]
        dl = np.linalg.norm(d)
        if dl > 1e-6:
            tangents.append(d / dl)
    if i < N_SAMPLES - 1:
        d = sample_a[i+1] - sample_a[i]
        dl = np.linalg.norm(d)
        if dl > 1e-6:
            tangents.append(d / dl)
    
    tangent = np.mean(tangents, axis=0)
    tl = np.linalg.norm(tangent)
    if tl > 1e-6:
        tangent = tangent / tl
    normal = np.array([-tangent[1], tangent[0]])
    dot = np.dot(normal, to_c)
    flipped = dot < 0
    if flipped:
        normal = -normal

    t_angle = math.degrees(math.atan2(tangent[1], tangent[0]))
    n_angle = math.degrees(math.atan2(normal[1], normal[0]))
    print(f"  [{i}] tangent={t_angle:+.1f}° normal={n_angle:+.1f}° to_center={angle_to_center:+.1f}° flipped={flipped}")
