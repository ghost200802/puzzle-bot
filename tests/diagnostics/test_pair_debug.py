import os
import sys
import math
import json

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..'))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))

from common import pieces, sides, util

from pipline.config import get_output_dir, get_deduped_path

OUTPUT_DIR = get_output_dir()
DEDUPED_PATH = get_deduped_path()


def signed_distance(point, line_p1, line_p2):
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    cross = dx * (point[1] - line_p1[1]) - dy * (point[0] - line_p1[0])
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.001:
        return 0
    return cross / length


def main():
    ps_raw = pieces.Piece.load_all(DEDUPED_PATH, resample=False)
    ps_res = pieces.Piece.load_all(DEDUPED_PATH, resample=True)

    pid_a, si_a = 125, 3
    pid_b, si_b = 78, 1

    pa = ps_raw[pid_a]
    pb = ps_raw[pid_b]

    print("=" * 60)
    print(f"Diagnosing: Piece {pid_a}[{si_a}] vs Piece {pid_b}[{si_b}]")
    print("=" * 60)

    print(f"\n--- Piece {pid_a} ---")
    print(f"  center: {pa.sides[0].piece_center}")
    for i in range(4):
        s = pa.sides[i]
        p1, p2 = s.original_p1, s.original_p2
        angle_deg = math.degrees(s.original_angle) % 360
        sd = signed_distance(s.piece_center, p1, p2)
        print(f"  side[{i}]: p1={p1} -> p2={p2}  angle={angle_deg:.1f}\u00b0  len={s.original_length:.0f}  is_edge={s.is_edge}  center_sd={sd:.1f}")

    print(f"\n--- Piece {pid_b} ---")
    print(f"  center: {pb.sides[0].piece_center}")
    for i in range(4):
        s = pb.sides[i]
        p1, p2 = s.original_p1, s.original_p2
        angle_deg = math.degrees(s.original_angle) % 360
        sd = signed_distance(s.piece_center, p1, p2)
        print(f"  side[{i}]: p1={p1} -> p2={p2}  angle={angle_deg:.1f}\u00b0  len={s.original_length:.0f}  is_edge={s.is_edge}  center_sd={sd:.1f}")

    sa = pa.sides[si_a]
    sb = pb.sides[si_b]

    print(f"\n--- Matching Sides Detail ---")
    print(f"  A: side[{si_a}]  p1={sa.original_p1} -> p2={sa.original_p2}")
    print(f"     angle={math.degrees(sa.original_angle) % 360:.1f}\u00b0  len={sa.original_length:.0f}")
    print(f"  B: side[{si_b}]  p1={sb.original_p1} -> p2={sb.original_p2}")
    print(f"     angle={math.degrees(sb.original_angle) % 360:.1f}\u00b0  len={sb.original_length:.0f}")

    print(f"\n--- Pre-filter Checks ---")

    print(f"\n  1) Edge check:")
    print(f"     A side[{si_a}] is_edge = {sa.is_edge}")
    print(f"     B side[{si_b}] is_edge = {sb.is_edge}")
    print(f"     PASS: neither is edge")

    print(f"\n  2) Length check:")
    d_scale = 1.0 - (sa.original_length / sb.original_length)
    print(f"     len_a={sa.original_length:.1f}  len_b={sb.original_length:.1f}  d_scale={d_scale:.4f}")
    print(f"     threshold={sides.SIDE_MAX_LENGTH_DISCREPANCY}")
    print(f"     {'PASS' if abs(d_scale) <= sides.SIDE_MAX_LENGTH_DISCREPANCY else 'FAIL'}")

    print(f"\n  3) Center constraint:")
    sd_a = sa.center_side_sign()
    sd_b = sb.center_side_sign()
    print(f"     sd_a={sd_a:.1f}  sd_b={sd_b:.1f}  product={sd_a * sd_b:.1f}")
    print(f"     Same side (product > 0) = pieces on OPPOSITE sides of joined edge = VALID")
    print(f"     Diff side (product < 0) = pieces on SAME side of joined edge = INVALID")
    if sd_a * sd_b < 0:
        print(f"     FAIL: centers on same side")
    else:
        print(f"     PASS: centers on opposite sides")

    print(f"\n  4) Adjacent edge constraint:")
    adj_map = {
        (si_a - 1) % 4: (si_b + 1) % 4,
        (si_a + 1) % 4: (si_b - 1) % 4,
    }
    for adj_a_si, adj_b_si in adj_map.items():
        adj_a = pa.sides[adj_a_si]
        adj_b = pb.sides[adj_b_si]
        angle_a = math.degrees(adj_a.original_angle) % 360
        angle_b = math.degrees(adj_b.original_angle) % 360
        angle_diff = util.compare_angles(adj_a.original_angle, adj_b.original_angle)
        angle_diff_deg = math.degrees(angle_diff)

        print(f"\n     A side[{adj_a_si}] <-> B side[{adj_b_si}]:")
        print(f"       A: is_edge={adj_a.is_edge}  angle={angle_a:.1f}\u00b0  p1={adj_a.original_p1} -> p2={adj_a.original_p2}")
        print(f"       B: is_edge={adj_b.is_edge}  angle={angle_b:.1f}\u00b0  p1={adj_b.original_p1} -> p2={adj_b.original_p2}")
        print(f"       angle_diff={angle_diff_deg:.1f}\u00b0  threshold={math.degrees(sides.EDGE_PARALLEL_THRESHOLD_RAD):.1f}\u00b0")

        if adj_a.is_edge and adj_b.is_edge:
            if angle_diff > sides.EDGE_PARALLEL_THRESHOLD_RAD:
                print(f"       BOTH EDGE + NOT PARALLEL -> FAIL")
            else:
                print(f"       BOTH EDGE + PARALLEL -> PASS")
        elif adj_a.is_edge and not adj_b.is_edge:
            print(f"       A is edge but B is NOT -> FAIL")
        elif not adj_a.is_edge and adj_b.is_edge:
            print(f"       B is edge but A is NOT -> FAIL")
        else:
            print(f"       Neither is edge -> PASS (no constraint)")

    print(f"\n--- Shape Error ---")
    sa_res = ps_res[pid_a].sides[si_a]
    sb_res = ps_res[pid_b].sides[si_b]
    error = sa_res.error_when_fit_with(sb_res, flip=True, skip_edges=False)
    print(f"  error = {error:.4f}  (threshold={sides.SIDE_MAX_ERROR_TO_MATCH})")
    print(f"  {'MATCH' if error <= sides.SIDE_MAX_ERROR_TO_MATCH else 'NO MATCH'}")

    print(f"\n--- What SHOULD the edge adjacency be? ---")
    print(f"  Piece {pid_a}: edge is side 2 (angle ~{math.degrees(pa.sides[2].original_angle) % 360:.0f}\u00b0)")
    print(f"  Piece {pid_b}: edge is side 1 (angle ~{math.degrees(pb.sides[1].original_angle) % 360:.0f}\u00b0)")
    print(f"  For these to be adjacent edge pieces, their edge sides should be")
    print(f"  adjacent to the matching sides.")
    print(f"  A side[{si_a}] adjacents: side[{(si_a-1)%4}] and side[{(si_a+1)%4}]")
    print(f"  A side[{(si_a-1)%4}] is_edge={pa.sides[(si_a-1)%4].is_edge}")
    print(f"  A side[{(si_a+1)%4}] is_edge={pa.sides[(si_a+1)%4].is_edge}")
    print(f"  B side[{si_b}] adjacents: side[{(si_b-1)%4}] and side[{(si_b+1)%4}]")
    print(f"  B side[{(si_b-1)%4}] is_edge={pb.sides[(si_b-1)%4].is_edge}")
    print(f"  B side[{(si_b+1)%4}] is_edge={pb.sides[(si_b+1)%4].is_edge}")


if __name__ == '__main__':
    main()
