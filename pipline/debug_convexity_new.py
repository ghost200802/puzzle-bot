import os
import sys
import json
import math

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common import sides, util
from common.sides import CONVEXITY_TRENDLINE_FRACTION

DEDUPED_DIR = os.path.join(_here, '..', 'output', 'puzzle_new', '4_deduped')


def analyze_piece_side(pid, si):
    path = os.path.join(DEDUPED_DIR, f'side_{pid}_{si}.json')
    with open(path) as f:
        data = json.load(f)
    vertices = np.array(data['vertices'])
    piece_center = np.array(data.get('piece_center', [0, 0]))
    is_edge = data.get('is_edge', False)

    if is_edge:
        print(f"  Piece {pid} Side {si}: EDGE, skip")
        return

    p1 = tuple(vertices[0])
    p2 = tuple(vertices[-1])
    n = len(vertices)

    center_sd = sides.Side._signed_distance_to_line(piece_center, p1, p2)

    trend_len = max(5, int(n * CONVEXITY_TRENDLINE_FRACTION))
    middle_start = trend_len
    middle_end = n - trend_len
    middle_vertices = vertices[middle_start:middle_end]

    total_sd = 0
    for v in middle_vertices:
        total_sd += sides.Side._signed_distance_to_line(tuple(v), p1, p2)
    avg_middle_sd = total_sd / len(middle_vertices)

    new_convex = (avg_middle_sd * center_sd) < 0

    all_total = 0
    for v in vertices:
        all_total += sides.Side._signed_distance_to_line(tuple(v), p1, p2)
    avg_all_sd = all_total / n
    old_convex = (avg_all_sd * center_sd) > 0

    pos_count = sum(1 for v in middle_vertices if sides.Side._signed_distance_to_line(tuple(v), p1, p2) > 0)
    neg_count = sum(1 for v in middle_vertices if sides.Side._signed_distance_to_line(tuple(v), p1, p2) <= 0)

    print(f"  Piece {pid} Side {si}: n={n}, trend_len={trend_len}, middle={len(middle_vertices)} verts")
    print(f"    center_sd = {center_sd:.2f}")
    print(f"    avg_middle_sd = {avg_middle_sd:.2f} (pos={pos_count}, neg={neg_count})")
    print(f"    product = {avg_middle_sd * center_sd:.2f}")
    print(f"    NEW => is_convex = {new_convex} ({'凸/tab' if new_convex else '凹/slot'})")
    print(f"    [old] avg_all_sd = {avg_all_sd:.2f}, old_convex = {old_convex}")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Piece 137 (side 1 should be 凸/convex)")
    print("=" * 60)
    for si in range(4):
        analyze_piece_side(137, si)

    print("=" * 60)
    print("Piece 141 (side 3 should be 凹/concave)")
    print("=" * 60)
    for si in range(4):
        analyze_piece_side(141, si)
