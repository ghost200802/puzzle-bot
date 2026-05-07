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


def get_convexity(data):
    vertices = np.array(data['vertices'])
    piece_center = np.array(data.get('piece_center', [0, 0]))
    is_edge = data.get('is_edge', False)

    if is_edge:
        return None, None, None

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

    if abs(avg_middle_sd) < 0.5:
        new_convex = None
    else:
        new_convex = (avg_middle_sd * center_sd) < 0

    all_total = 0
    for v in vertices:
        all_total += sides.Side._signed_distance_to_line(tuple(v), p1, p2)
    avg_all_sd = all_total / n

    if abs(avg_all_sd) < 1.0:
        old_convex = None
    else:
        old_convex = (avg_all_sd * center_sd) > 0

    return old_convex, new_convex, avg_all_sd


if __name__ == '__main__':
    files = sorted([f for f in os.listdir(DEDUPED_DIR) if f.startswith('side_') and f.endswith('.json')])

    total = 0
    same = 0
    changed = 0
    old_none = 0
    new_none = 0
    both_none = 0
    changes = []

    for fname in files:
        parts = fname.replace('side_', '').replace('.json', '').split('_')
        pid = int(parts[0])
        si = int(parts[1])

        with open(os.path.join(DEDUPED_DIR, fname)) as f:
            data = json.load(f)

        old_c, new_c, avg_all = get_convexity(data)

        if old_c is None and new_c is None:
            both_none += 1
            total += 1
            continue

        total += 1

        if old_c is None:
            old_none += 1
            changes.append((pid, si, old_c, new_c, avg_all))
        elif new_c is None:
            new_none += 1
            changes.append((pid, si, old_c, new_c, avg_all))
        elif old_c != new_c:
            changed += 1
            changes.append((pid, si, old_c, new_c, avg_all))
        else:
            same += 1

    print(f"Total non-edge sides: {total}")
    print(f"Same classification: {same}")
    print(f"Changed (flip): {changed}")
    print(f"Old=None, New=valid: {old_none}")
    print(f"Old=valid, New=None: {new_none}")
    print(f"Both None (edges): {both_none}")

    if changes:
        print(f"\n{'='*60}")
        print("Changed/affected sides:")
        print(f"{'='*60}")
        for pid, si, old_c, new_c, avg_all in sorted(changes):
            old_str = 'None' if old_c is None else ('凸' if old_c else '凹')
            new_str = 'None' if new_c is None else ('凸' if new_c else '凹')
            print(f"  piece {pid:3d} side {si}: {old_str} -> {new_str}  (old avg_all_sd={avg_all:.2f})")
