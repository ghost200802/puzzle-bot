import os
import sys
import json
import math

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common import sides, util

DEDUPED_DIR = os.path.join(_here, '..', 'output', 'puzzle_new', '4_deduped')

def analyze_convexity(pid, si):
    path = os.path.join(DEDUPED_DIR, f'side_{pid}_{si}.json')
    with open(path) as f:
        data = json.load(f)
    vertices = np.array(data['vertices'])
    piece_center = np.array(data.get('piece_center', [0, 0]))
    is_edge = data.get('is_edge', False)

    p1 = vertices[0]
    p2 = vertices[-1]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    cross_center = dx * (piece_center[1] - p1[1]) - dy * (piece_center[0] - p1[0])
    length = math.sqrt(dx*dx + dy*dy)
    center_sd = cross_center / length if length > 0.001 else 0

    total_sd = 0
    for v in vertices:
        cross = dx * (v[1] - p1[1]) - dy * (v[0] - p1[0])
        total_sd += cross / length if length > 0.001 else 0
    avg_sd = total_sd / len(vertices)

    is_convex_computed = (avg_sd * center_sd) > 0

    print(f"  Piece {pid} Side {si}:")
    print(f"    p1={p1}, p2={p2}")
    print(f"    piece_center={piece_center}")
    print(f"    direction=({dx:.1f}, {dy:.1f}), length={length:.1f}")
    print(f"    center_sd={center_sd:.4f}")
    print(f"    avg_sd={avg_sd:.4f}")
    print(f"    center_sd * avg_sd = {center_sd * avg_sd:.4f}")
    print(f"    is_convex={is_convex_computed} (avg_sd and center_sd same side = convex)")
    print(f"    is_edge={is_edge}")
    print(f"    num_vertices={len(vertices)}")
    return is_convex_computed

print("=" * 60)
print("Convexity analysis: 137[1] and 141[3]")
print("=" * 60)

print("\n--- 137 Side 1 (should be convex=True) ---")
analyze_convexity(137, 1)

print("\n--- 141 Side 3 (should be concave=False, but is convex=True) ---")
analyze_convexity(141, 3)

print("\n--- 141 all sides ---")
for si in range(4):
    analyze_convexity(141, si)
    print()

print("\n--- 137 all sides ---")
for si in range(4):
    analyze_convexity(137, si)
    print()
