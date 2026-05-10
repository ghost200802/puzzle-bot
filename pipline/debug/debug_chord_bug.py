import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))

from common.texture_verify import load_side_data

from pipline.config import get_output_dir, get_vector_path

OUTPUT_DIR = get_output_dir()
DEDUPED_PATH = get_vector_path()

side_a = load_side_data(DEDUPED_PATH, 4, 3)
verts = side_a['vertices']
p1 = verts[0]
p2 = verts[-1]
chord = p2 - p1
chord_len = np.linalg.norm(chord)
chord_dir = chord / chord_len

projections = np.array([np.dot(v - p1, chord_dir) for v in verts])

print(f"Chord length: {chord_len:.2f}")
print(f"Vertices: {len(verts)}")
print(f"\nProjection monotonicity check:")
violations = 0
for i in range(1, len(projections)):
    if projections[i] < projections[i-1] - 0.01:
        violations += 1
        if violations <= 10:
            print(f"  Fold-back at vertex {i-1}->{i}: proj {projections[i-1]:.2f} -> {projections[i]:.2f}")
print(f"  Total fold-backs: {violations}")

print(f"\nProjection around tab region (vertices 150-200):")
for i in range(150, min(200, len(projections))):
    marker = " <-- fold" if i > 0 and projections[i] < projections[i-1] - 0.01 else ""
    print(f"  v[{i}]: proj={projections[i]:.2f}{marker}")

print(f"\nsearchsorted test on NON-MONOTONIC array:")
targets = [260.0, 267.0, 270.0, 280.0, 290.0]
for t in targets:
    idx = np.searchsorted(projections, t, side='right') - 1
    idx = max(0, min(idx, len(projections) - 2))
    print(f"  target={t:.1f}: searchsorted finds vertex {idx}, proj[{idx}]={projections[idx]:.2f}, proj[{idx+1}]={projections[idx+1]:.2f}")
    if projections[idx+1] < projections[idx]:
        print(f"    WARNING: found a fold-back segment!")
