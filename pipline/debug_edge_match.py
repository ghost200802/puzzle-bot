import os, sys, json, math

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common import pieces, sides, util

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '4_deduped')

ps_raw = pieces.Piece.load_all(DEDUPED_PATH, resample=False)

with open(os.path.join(OUTPUT_DIR, '5_connectivity', 'piece_edge_info.json')) as f:
    piece_edge_info = json.load(f)

def rotated_angle_check(piece_a, si, piece_b, sj):
    side_a = piece_a.sides[si]
    side_b = piece_b.sides[sj]
    rot_for_b = side_a.original_angle + math.pi - side_b.original_angle

    adj_map = {
        (si - 1) % 4: (sj + 1) % 4,
        (si + 1) % 4: (sj - 1) % 4,
    }
    results = []
    for adj_a_si, adj_b_si in adj_map.items():
        adj_a = piece_a.sides[adj_a_si]
        adj_b = piece_b.sides[adj_b_si]
        if adj_a.is_edge and adj_b.is_edge:
            adj_b_rotated = adj_b.original_angle + rot_for_b
            angle_diff = util.compare_angles(adj_a.original_angle, adj_b_rotated)
            results.append({
                'adj_a_si': adj_a_si, 'adj_b_si': adj_b_si,
                'adj_a_angle': math.degrees(adj_a.original_angle),
                'adj_b_angle_orig': math.degrees(adj_b.original_angle),
                'adj_b_angle_rotated': math.degrees(adj_b_rotated),
                'diff_old': math.degrees(util.compare_angles(adj_a.original_angle, adj_b.original_angle)),
                'diff_new': math.degrees(angle_diff),
            })
    return results

print("=" * 90)
print("Verify: After rotation, EDGE sides of 1, 26, 117 ARE parallel")
print("=" * 90)

cases = [
    (1, 1, 26, 0, "Piece 1[1] vs 26[0]"),
    (1, 1, 117, 2, "Piece 1[1] vs 117[2]"),
    (1, 3, 26, 2, "Piece 1[3] vs 26[2]"),
    (1, 3, 117, 0, "Piece 1[3] vs 117[0]"),
]

for pa, si, pb, sj, label in cases:
    piece_a = ps_raw[pa]
    piece_b = ps_raw[pb]
    side_a = piece_a.sides[si]
    side_b = piece_b.sides[sj]

    rot_for_b = side_a.original_angle + math.pi - side_b.original_angle
    print(f"\n{label}")
    print(f"  side_a angle = {math.degrees(side_a.original_angle):.2f}°")
    print(f"  side_b angle = {math.degrees(side_b.original_angle):.2f}°")
    print(f"  rotation_for_B = {math.degrees(rot_for_b):.2f}°")

    results = rotated_angle_check(piece_a, si, piece_b, sj)
    for r in results:
        print(f"  adj A[{r['adj_a_si']}](EDGE) ↔ B[{r['adj_b_si']}](EDGE):")
        print(f"    A.angle = {r['adj_a_angle']:.2f}°")
        print(f"    B.angle_orig = {r['adj_b_angle_orig']:.2f}°")
        print(f"    B.angle_rotated = {r['adj_b_angle_rotated']:.2f}°")
        print(f"    OLD diff (no rotation) = {r['diff_old']:.2f}°  {'PASS' if r['diff_old'] <= 10 else 'FAIL'}")
        print(f"    NEW diff (with rotation) = {r['diff_new']:.2f}°  {'PASS' if r['diff_new'] <= 10 else 'FAIL'}")

print("\n" + "=" * 90)
print("Comprehensive: ALL edge-piece pairs with both-EDGE adj sides")
print("=" * 90)

edge_pids = []
for pid_str, ef in piece_edge_info.items():
    pid = int(pid_str)
    flat_count = sum(1 for f in ef if f)
    if flat_count == 1:
        edge_pids.append(pid)

all_edge_angles = []
for pa in edge_pids:
    for pb in edge_pids:
        if pa >= pb:
            continue
        piece_a = ps_raw[pa]
        piece_b = ps_raw[pb]
        ef_a = piece_edge_info[str(pa)]
        ef_b = piece_edge_info[str(pb)]

        for si in range(4):
            if ef_a[si]:
                continue
            for sj in range(4):
                if ef_b[sj]:
                    continue

                side_a = piece_a.sides[si]
                side_b = piece_b.sides[sj]

                if side_a.is_convex is not None and side_b.is_convex is not None:
                    if side_a.is_convex == side_b.is_convex:
                        continue

                len_a = side_a.original_length
                len_b = side_b.original_length
                if len_a < 1 or len_b < 1:
                    continue
                d_scale = 1.0 - (len_a / len_b)
                if abs(d_scale) > sides.SIDE_MAX_LENGTH_DISCREPANCY:
                    continue

                adj_map = {
                    (si - 1) % 4: (sj + 1) % 4,
                    (si + 1) % 4: (sj - 1) % 4,
                }
                for adj_a_si, adj_b_si in adj_map.items():
                    adj_a = piece_a.sides[adj_a_si]
                    adj_b = piece_b.sides[adj_b_si]
                    if adj_a.is_edge and adj_b.is_edge:
                        rot_for_b = side_a.original_angle + math.pi - side_b.original_angle
                        adj_b_rotated = adj_b.original_angle + rot_for_b
                        diff_old = math.degrees(util.compare_angles(adj_a.original_angle, adj_b.original_angle))
                        diff_new = math.degrees(util.compare_angles(adj_a.original_angle, adj_b_rotated))
                        all_edge_angles.append({
                            'pair': f"{pa}[{si}]vs{pb}[{sj}]",
                            'adj': f"A[{adj_a_si}]↔B[{adj_b_si}]",
                            'diff_old': diff_old,
                            'diff_new': diff_new,
                        })

print(f"\nTotal both-EDGE adj cases: {len(all_edge_angles)}")
old_fail = [x for x in all_edge_angles if x['diff_old'] > 10]
new_fail = [x for x in all_edge_angles if x['diff_new'] > 10]
print(f"OLD method failures (diff > 10°): {len(old_fail)}")
print(f"NEW method failures (diff > 10°): {len(new_fail)}")

print(f"\nOLD failures (should have been PASS):")
for x in old_fail[:20]:
    print(f"  {x['pair']} adj {x['adj']}: old_diff={x['diff_old']:.2f}°, new_diff={x['diff_new']:.2f}°")

if new_fail:
    print(f"\nNEW failures (after rotation, still > 10°):")
    for x in new_fail[:20]:
        print(f"  {x['pair']} adj {x['adj']}: old_diff={x['diff_old']:.2f}°, new_diff={x['diff_new']:.2f}°")
