import os
import sys
import math
import json
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common import pieces, sides, util

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '4_deduped')


def signed_distance(point, line_p1, line_p2):
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    cross = dx * (point[1] - line_p1[1]) - dy * (point[0] - line_p1[0])
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.001:
        return 0
    return cross / length


def perpendicular_distances(vertices, p1, p2):
    p1 = tuple(p1)
    p2 = tuple(p2)
    dists = []
    for v in vertices:
        v = tuple(v)
        d = util.distance_to_line(v, p1, p2)
        sd = signed_distance(v, p1, p2)
        dists.append((d, sd))
    return dists


def rotate_point(v, angle, around=(0, 0)):
    dx = v[0] - around[0]
    dy = v[1] - around[1]
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (dx * cos_a - dy * sin_a + around[0],
            dx * sin_a + dy * cos_a + around[1])


def main():
    ps_raw = pieces.Piece.load_all(DEDUPED_PATH, resample=False)

    pid_a, si_a = 125, 3
    pid_b, si_b = 78, 1

    pa = ps_raw[pid_a]
    pb = ps_raw[pid_b]
    sa = pa.sides[si_a]
    sb = pb.sides[si_b]

    print("=" * 60)
    print(f"Deep analysis: P{pid_a}[{si_a}] vs P{pid_b}[{si_b}]")
    print("=" * 60)

    print(f"\n--- Perpendicular extent analysis ---")
    dists_a = perpendicular_distances(sa.vertices, sa.original_p1, sa.original_p2)
    dists_b = perpendicular_distances(sb.vertices, sb.original_p1, sb.original_p2)

    max_d_a = max(d[0] for d in dists_a)
    max_d_b = max(d[0] for d in dists_b)
    mean_d_a = sum(d[0] for d in dists_a) / len(dists_a)
    mean_d_b = sum(d[0] for d in dists_b) / len(dists_b)

    sd_vals_a = [d[1] for d in dists_a]
    sd_vals_b = [d[1] for d in dists_b]

    print(f"  A side[{si_a}]: max_perp_dist={max_d_a:.1f}  mean_perp_dist={mean_d_a:.1f}")
    print(f"                 sd range=[{min(sd_vals_a):.1f}, {max(sd_vals_a):.1f}]")
    print(f"  B side[{si_b}]: max_perp_dist={max_d_b:.1f}  mean_perp_dist={mean_d_b:.1f}")
    print(f"                 sd range=[{min(sd_vals_b):.1f}, {max(sd_vals_b):.1f}]")
    print(f"  Ratio: max_a/max_b={max_d_a/max_d_b:.2f}  mean_a/mean_b={mean_d_a/mean_d_b:.2f}")

    print(f"\n--- After alignment: where do the edge sides end up? ---")
    angle_a = sa.original_angle
    angle_b = sb.original_angle
    rot_angle = angle_a + math.pi - angle_b

    mid_a = ((sa.original_p1[0] + sa.original_p2[0]) / 2,
             (sa.original_p1[1] + sa.original_p2[1]) / 2)
    mid_b = ((sb.original_p1[0] + sb.original_p2[0]) / 2,
             (sb.original_p1[1] + sb.original_p2[1]) / 2)

    print(f"  A midpoint: ({mid_a[0]:.1f}, {mid_a[1]:.1f})")
    print(f"  B midpoint: ({mid_b[0]:.1f}, {mid_b[1]:.1f})")
    print(f"  Rotation angle for B: {math.degrees(rot_angle):.1f}°")

    center_a = sa.piece_center
    center_b_rotated = rotate_point(sb.piece_center, rot_angle, around=mid_b)
    center_b_in_a_frame = (center_b_rotated[0] - mid_b[0] + mid_a[0],
                           center_b_rotated[1] - mid_b[1] + mid_a[1])

    print(f"\n  A center in A frame: ({center_a[0]}, {center_a[1]})")
    print(f"  B center rotated then translated to A frame: ({center_b_in_a_frame[0]:.1f}, {center_b_in_a_frame[1]:.1f})")

    edge_a_angle = pa.sides[(si_a + 1) % 4].original_angle
    edge_b_raw_angle = pb.sides[(si_b - 1) % 4].original_angle
    edge_b_rotated_angle = edge_b_raw_angle + rot_angle

    print(f"\n  A edge side [{(si_a+1)%4}] angle: {math.degrees(edge_a_angle):.1f}°")
    print(f"  B edge side [{(si_b-1)%4}] angle: {math.degrees(edge_b_raw_angle):.1f}° -> after rotation: {math.degrees(edge_b_rotated_angle):.1f}°")
    print(f"  Difference: {math.degrees(util.compare_angles(edge_a_angle, edge_b_rotated_angle)):.1f}°")

    print(f"\n  A center relative to matching side midpoint: ({center_a[0]-mid_a[0]:.1f}, {center_a[1]-mid_a[1]:.1f})")
    print(f"  B center relative to matching side midpoint: ({center_b_in_a_frame[0]-mid_a[0]:.1f}, {center_b_in_a_frame[1]-mid_a[1]:.1f})")

    sd_center_a_to_match = signed_distance(center_a, sa.original_p1, sa.original_p2)
    sd_center_b_to_match = signed_distance(center_b_in_a_frame, sa.original_p1, sa.original_p2)
    print(f"\n  A center signed dist to matching line: {sd_center_a_to_match:.1f}")
    print(f"  B center (rotated) signed dist to A's matching line: {sd_center_b_to_match:.1f}")
    print(f"  Same side? {sd_center_a_to_match * sd_center_b_to_match > 0}")

    print(f"\n--- What about the other adjacent edges? ---")
    adj_a_prev = pa.sides[(si_a - 1) % 4]
    adj_b_next = pb.sides[(si_b + 1) % 4]
    print(f"  A side[{(si_a-1)%4}]: is_edge={adj_a_prev.is_edge}  angle={math.degrees(adj_a_prev.original_angle):.1f}°  len={adj_a_prev.original_length:.0f}")
    print(f"  B side[{(si_b+1)%4}]: is_edge={adj_b_next.is_edge}  angle={math.degrees(adj_b_next.original_angle):.1f}°  len={adj_b_next.original_length:.0f}")

    adj_a_next = pa.sides[(si_a + 1) % 4]
    adj_b_prev = pb.sides[(si_b - 1) % 4]
    print(f"  A side[{(si_a+1)%4}]: is_edge={adj_a_next.is_edge}  angle={math.degrees(adj_a_next.original_angle):.1f}°  len={adj_a_next.original_length:.0f}")
    print(f"  B side[{(si_b-1)%4}]: is_edge={adj_b_prev.is_edge}  angle={math.degrees(adj_b_prev.original_angle):.1f}°  len={adj_b_prev.original_length:.0f}")

    print(f"\n--- KEY INSIGHT ---")
    print(f"  Both P{pid_a} and P{pid_b} are edge pieces with TOP edges.")
    print(f"  P{pid_a} side[{si_a}] is the LEFT side, P{pid_b} side[{si_b}] is the RIGHT side.")
    print(f"  When joined, both edge sides (TOP) end up on the SAME side.")
    print(f"  But P{pid_a} edge angle={math.degrees(edge_a_angle):.1f}°, rotated P{pid_b} edge angle={math.degrees(edge_b_rotated_angle):.1f}°")
    print(f"  For valid edge-piece match, edge sides should be ANTI-PARALLEL (differ by ~180°)")
    print(f"  or PARALLEL (same direction) for pieces on the same border row.")
    print(f"  They ARE parallel ({math.degrees(util.compare_angles(edge_a_angle, edge_b_rotated_angle)):.1f}° diff)")
    print(f"  BUT: edge lengths differ: P{pid_a}={adj_a_next.original_length:.0f} vs P{pid_b}={adj_b_prev.original_length:.0f}")
    print(f"  AND: matching side lengths differ: {sa.original_length:.0f} vs {sb.original_length:.0f} ({abs(1-sa.original_length/sb.original_length)*100:.1f}%)")


if __name__ == '__main__':
    main()
