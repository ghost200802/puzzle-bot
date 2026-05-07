import os
import sys
import json
import math

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common import sides, util

DEDUPED_DIR = os.path.join(_here, '..', 'output', 'puzzle_new', '4_deduped')


def detailed_convexity_analysis(pid, si):
    path = os.path.join(DEDUPED_DIR, f'side_{pid}_{si}.json')
    with open(path) as f:
        data = json.load(f)
    vertices = np.array(data['vertices'])
    piece_center = np.array(data.get('piece_center', [0, 0]))
    is_edge = data.get('is_edge', False)

    p1 = tuple(vertices[0])
    p2 = tuple(vertices[-1])
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)

    center_sd = sides.Side._signed_distance_to_line(piece_center, p1, p2)

    sd_values = []
    for v in vertices:
        sd = sides.Side._signed_distance_to_line(tuple(v), p1, p2)
        sd_values.append(sd)

    sd_arr = np.array(sd_values)
    avg_sd = np.mean(sd_arr)
    max_sd = np.max(sd_arr)
    min_sd = np.min(sd_arr)
    median_sd = np.median(sd_arr)
    std_sd = np.std(sd_arr)

    n_positive = np.sum(sd_arr > 0.5)
    n_negative = np.sum(sd_arr < -0.5)
    n_near_zero = np.sum(np.abs(sd_arr) <= 0.5)

    print(f"  Piece {pid} Side {si}: p1={p1}, p2={p2}")
    print(f"    direction=({dx:.0f}, {dy:.0f}), length={length:.1f}")
    print(f"    piece_center={tuple(piece_center)}")
    print(f"    center_sd = {center_sd:.4f}")
    print(f"    num_vertices = {len(vertices)}")
    print(f"    signed distance stats:")
    print(f"      min={min_sd:.4f}, max={max_sd:.4f}")
    print(f"      mean={avg_sd:.4f}, median={median_sd:.4f}, std={std_sd:.4f}")
    print(f"      n_positive(>0.5)={n_positive}, n_negative(<-0.5)={n_negative}, n_near_zero={n_near_zero}")

    n = len(sd_values)
    print(f"    first 10 sd: {[f'{v:.2f}' for v in sd_values[:10]]}")
    print(f"    last  10 sd: {[f'{v:.2f}' for v in sd_values[-10:]]}")
    print(f"    mid   10 sd: {[f'{v:.2f}' for v in sd_values[n//2-5:n//2+5]]}")

    print(f"    top 20 max sd vertices:")
    top_idx = np.argsort(sd_arr)[-20:][::-1]
    for idx in top_idx:
        print(f"      [{idx}] sd={sd_arr[idx]:.4f} vertex={tuple(vertices[idx])}")

    print(f"    bottom 20 min sd vertices:")
    bot_idx = np.argsort(sd_arr)[:20]
    for idx in bot_idx:
        print(f"      [{idx}] sd={sd_arr[idx]:.4f} vertex={tuple(vertices[idx])}")

    is_convex = (avg_sd * center_sd) > 0
    print(f"    avg_sd * center_sd = {avg_sd * center_sd:.4f}")
    print(f"    is_convex = {is_convex}")
    print()


print("=" * 70)
print("DETAILED convexity analysis")
print("=" * 70)

print("\n===== 137 Side 1 (user says: convex/凸) =====")
detailed_convexity_analysis(137, 1)

print("\n===== 141 Side 3 (user says: concave/凹) =====")
detailed_convexity_analysis(141, 3)

print("\n===== 141 Side 0 (reference: clearly concave, avg_sd=-54.7) =====")
detailed_convexity_analysis(141, 0)

print("\n===== 141 Side 1 (reference: clearly convex, avg_sd=8.0) =====")
detailed_convexity_analysis(141, 1)
