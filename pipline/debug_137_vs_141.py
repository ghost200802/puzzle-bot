import os
import sys
import json
import math

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common import pieces, sides, util

DEDUPED_DIR = os.path.join(_here, '..', 'output', 'puzzle_new', '4_deduped')

ps_raw = pieces.Piece.load_all(DEDUPED_DIR, resample=False)
ps_res = pieces.Piece.load_all(DEDUPED_DIR, resample=True)

print("=" * 60)
print("Debug: Why 137[1] does not match 141")
print("=" * 60)

p137 = ps_raw[137]
p141 = ps_raw[141]

print("\n--- Piece 137 sides ---")
for si in range(4):
    s = p137.sides[si]
    print(f"  Side {si}: angle={math.degrees(s.original_angle):.2f} deg, "
          f"len={s.original_length:.2f}, is_edge={s.is_edge}, "
          f"is_convex={s.is_convex}, center_sign={s.center_side_sign():.4f}")

print("\n--- Piece 141 sides ---")
for si in range(4):
    s = p141.sides[si]
    print(f"  Side {si}: angle={math.degrees(s.original_angle):.2f} deg, "
          f"len={s.original_length:.2f}, is_edge={s.is_edge}, "
          f"is_convex={s.is_convex}, center_sign={s.center_side_sign():.4f}")

print("\n--- Checking all combinations 137[si] vs 141[sj] ---")
for si in range(4):
    sa = p137.sides[si]
    if sa.is_edge:
        continue
    for sj in range(4):
        sb = p141.sides[sj]
        if sb.is_edge:
            print(f"  137[{si}] vs 141[{sj}]: SKIP - 141[{sj}] is EDGE")
            continue

        print(f"\n  === 137[{si}] vs 141[{sj}] ===")

        # Check length
        len_a = sa.original_length
        len_b = sb.original_length
        d_scale = abs(1.0 - (len_a / len_b))
        print(f"    len_a={len_a:.2f}, len_b={len_b:.2f}, d_scale={d_scale:.4f}, "
              f"threshold={sides.SIDE_MAX_LENGTH_DISCREPANCY}, "
              f"PASS={d_scale <= sides.SIDE_MAX_LENGTH_DISCREPANCY}")

        # Check convexity
        print(f"    convex_a={sa.is_convex}, convex_b={sb.is_convex}, "
              f"same={sa.is_convex == sb.is_convex if sa.is_convex is not None and sb.is_convex is not None else 'N/A'}")
        if sa.is_convex is not None and sb.is_convex is not None:
            if sa.is_convex == sb.is_convex:
                print(f"    -> REJECT: same convexity")
                continue

        # Check center side sign
        sd_a = sa.center_side_sign()
        sd_b = sb.center_side_sign()
        print(f"    center_sign_a={sd_a:.4f}, center_sign_b={sd_b:.4f}, "
              f"product={sd_a * sd_b:.4f}")
        if sd_a != 0 and sd_b != 0 and sd_a * sd_b < 0:
            print(f"    -> REJECT: center sign mismatch")
            continue

        # Check adj map
        rot_for_b = sa.original_angle + math.pi - sb.original_angle
        adj_map = {
            (si - 1) % 4: (sj + 1) % 4,
            (si + 1) % 4: (sj - 1) % 4,
        }
        for adj_a_si, adj_b_si in adj_map.items():
            adj_a = p137.sides[adj_a_si]
            adj_b = p141.sides[adj_b_si]
            print(f"    adj: 137[{adj_a_si}] is_edge={adj_a.is_edge} vs 141[{adj_b_si}] is_edge={adj_b.is_edge}")
            if adj_a.is_edge != adj_b.is_edge:
                print(f"    -> REJECT: adj edge mismatch")
                break
            if adj_a.is_edge and adj_b.is_edge:
                adj_b_rotated = adj_b.original_angle + rot_for_b
                angle_diff = util.compare_angles(adj_a.original_angle, adj_b_rotated)
                print(f"    adj edge parallel check: angle_diff={math.degrees(angle_diff):.2f} deg, "
                      f"threshold={math.degrees(sides.EDGE_PARALLEL_THRESHOLD_RAD):.2f} deg, "
                      f"PASS={angle_diff <= sides.EDGE_PARALLEL_THRESHOLD_RAD}")
                if angle_diff > sides.EDGE_PARALLEL_THRESHOLD_RAD:
                    print(f"    -> REJECT: adj edge not parallel after rotation")
                    break
        else:
            # All adj checks passed, compute error
            sa_res = ps_res[137].sides[si]
            sb_res = ps_res[141].sides[sj]
            error, shift = sa_res.error_when_fit_with(sb_res, flip=True, skip_edges=False)
            print(f"    -> ALL FILTERS PASSED! error={error:.4f}, shift=({shift[0]:.2f}, {shift[1]:.2f}), "
                  f"threshold={sides.SIDE_MAX_ERROR_TO_MATCH}, PASS={error <= sides.SIDE_MAX_ERROR_TO_MATCH}")
