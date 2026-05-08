import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

import importlib
import common.texture_verify as tv
importlib.reload(tv)
from common.texture_verify import load_side_data, _resample_by_chord, N_SAMPLES

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '3_vector')

side_a = load_side_data(DEDUPED_PATH, 4, 3)
side_b = load_side_data(DEDUPED_PATH, 137, 3)

chord_a = side_a['vertices'][-1] - side_a['vertices'][0]
chord_len_a = np.linalg.norm(chord_a)
chord_dir_a = chord_a / chord_len_a

chord_b = side_b['vertices'][-1] - side_b['vertices'][0]
chord_len_b = np.linalg.norm(chord_b)
chord_dir_b = chord_b / chord_len_b

flipped_b = side_b['vertices'][::-1].copy()
chord_bf = flipped_b[-1] - flipped_b[0]
chord_len_bf = np.linalg.norm(chord_bf)
chord_dir_bf = chord_bf / chord_len_bf

resampled_a = _resample_by_chord(side_a['vertices'], N_SAMPLES)
resampled_bf = _resample_by_chord(flipped_b, N_SAMPLES)

proj_a = np.array([np.dot(v - side_a['vertices'][0], chord_dir_a) for v in resampled_a])
proj_bf_on_a = np.array([np.dot(v - side_a['vertices'][0], chord_dir_a) for v in resampled_bf])

print("Chord comparison:")
print(f"  A chord: len={chord_len_a:.2f}, dir={chord_dir_a}")
print(f"  B chord: len={chord_len_b:.2f}, dir={chord_dir_b}")
print(f"  B flipped chord: len={chord_len_bf:.2f}, dir={chord_dir_bf}")
print(f"  A vs B_flipped chord dir dot: {np.dot(chord_dir_a, chord_dir_bf):.4f}")
print(f"  Angle between chords: {np.degrees(np.arccos(np.clip(np.dot(chord_dir_a, chord_dir_bf), -1, 1))):.1f} deg")

print(f"\nChord projection comparison (resampled onto A's chord):")
print(f"  {'idx':>3}  {'proj_A':>10}  {'proj_Bf_on_A':>13}  {'diff':>8}")
for i in range(N_SAMPLES):
    print(f"  {i:>3}  {proj_a[i]:>10.2f}  {proj_bf_on_a[i]:>13.2f}  {proj_a[i]-proj_bf_on_a[i]:>8.2f}")

mean_diff = np.mean(np.abs(proj_a - proj_bf_on_a))
print(f"\n  Mean |diff|: {mean_diff:.2f} (out of chord {chord_len_a:.2f})")
print(f"  Mean |diff| / chord: {mean_diff/chord_len_a*100:.1f}%")

print(f"\n  For reference, arc-length sampling had Mean |diff|: 376.77 ({376.77/chord_len_a*100:.1f}%)")
