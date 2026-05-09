import os
import sys
import numpy as np
import json

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.texture_verify import load_side_data, _resample_polyline, N_SAMPLES

from config import get_output_dir, get_vector_path

OUTPUT_DIR = get_output_dir()
DEDUPED_PATH = get_vector_path()

PID_A, SI_A = 4, 3
PID_B, SI_B = 137, 3

side_a = load_side_data(DEDUPED_PATH, PID_A, SI_A)
side_b = load_side_data(DEDUPED_PATH, PID_B, SI_B)

for label, side in [("A (piece 4)", side_a), ("B (piece 137)", side_b)]:
    verts = side['vertices']
    p1 = verts[0]
    p2 = verts[-1]
    chord = p2 - p1
    chord_len = np.linalg.norm(chord)
    chord_dir = chord / chord_len

    diffs = np.diff(verts, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    arc_len = np.sum(seg_lengths)

    projections = np.array([np.dot(v - p1, chord_dir) for v in verts])
    perp_dists = np.array([np.linalg.norm((v - p1) - np.dot(v - p1, chord_dir) * chord_dir) for v in verts])

    resampled_arc = _resample_polyline(verts, N_SAMPLES)
    proj_arc = np.array([np.dot(v - p1, chord_dir) for v in resampled_arc])
    perp_arc = np.array([np.linalg.norm((v - p1) - np.dot(v - p1, chord_dir) * chord_dir) for v in resampled_arc])

    print(f"\n{'='*60}")
    print(f"Side {label}")
    print(f"  Chord length:  {chord_len:.2f}")
    print(f"  Arc length:    {arc_len:.2f}")
    print(f"  Ratio arc/chord: {arc_len/chord_len:.2f}")
    print(f"  Vertices:      {len(verts)}")
    print(f"  Projection range: [{projections.min():.2f}, {projections.max():.2f}]")
    print(f"  Max perp dist:   {perp_dists.max():.2f} (at vertex {np.argmax(perp_dists)})")

    print(f"\n  Arc-length resampled ({N_SAMPLES} points) - chord projections:")
    print(f"    Range: [{proj_arc.min():.2f}, {proj_arc.max():.2f}]")
    proj_bins = np.linspace(0, chord_len, 11)
    hist, _ = np.histogram(proj_arc, bins=proj_bins)
    for i in range(10):
        bar = '#' * hist[i]
        print(f"    [{proj_bins[i]:6.1f}-{proj_bins[i+1]:6.1f}]: {hist[i]:3d} {bar}")

    print(f"\n  Arc-length resampled - perp distances:")
    print(f"    min={perp_arc.min():.2f}, max={perp_arc.max():.2f}, mean={perp_arc.mean():.2f}")

    n_near_zero = np.sum(perp_arc < 5)
    n_large = np.sum(perp_arc > 30)
    print(f"    Points with perp<5:  {n_near_zero}/{N_SAMPLES} ({n_near_zero/N_SAMPLES*100:.0f}%)")
    print(f"    Points with perp>30: {n_large}/{N_SAMPLES} ({n_large/N_SAMPLES*100:.0f}%)")

print(f"\n{'='*60}")
print("Overlap analysis:")
p1_a = side_a['vertices'][0]
p2_a = side_a['vertices'][-1]
chord_a = p2_a - p1_a
chord_len_a = np.linalg.norm(chord_a)
chord_dir_a = chord_a / chord_len_a

resampled_a = _resample_polyline(side_a['vertices'], N_SAMPLES)
resampled_b = _resample_polyline(side_b['vertices'][::-1], N_SAMPLES)

proj_a = np.array([np.dot(v - p1_a, chord_dir_a) for v in resampled_a])
proj_b = np.array([np.dot(v - p1_a, chord_dir_a) for v in resampled_b])

print(f"  Arc-sampled A chord projections: [{proj_a.min():.2f}, {proj_a.max():.2f}]")
print(f"  Arc-sampled B chord projections: [{proj_b.min():.2f}, {proj_b.max():.2f}]")
print(f"  These should overlap for NCC to work!")

print(f"\n  Per-sample chord projection comparison (first 10):")
print(f"  {'idx':>3}  {'proj_A':>8}  {'proj_B':>8}  {'diff':>8}")
for i in range(min(10, N_SAMPLES)):
    print(f"  {i:>3}  {proj_a[i]:>8.2f}  {proj_b[i]:>8.2f}  {proj_a[i]-proj_b[i]:>8.2f}")

mean_diff = np.mean(np.abs(proj_a - proj_b))
max_diff = np.max(np.abs(proj_a - proj_b))
print(f"\n  Mean |diff|: {mean_diff:.2f}")
print(f"  Max  |diff|: {max_diff:.2f}")
print(f"  => Arc-length sampling gives MISALIGNED positions on the chord")
