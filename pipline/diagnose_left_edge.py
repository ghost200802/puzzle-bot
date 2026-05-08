import os
import sys
import json
import heapq

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR
from common.board import Board, OPPOSITE, TOP, RIGHT, BOTTOM, LEFT
from common import output as board_output
from solve_display import generate_assembly_png

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)
SOLUTION_PATH = os.path.join(OUTPUT_DIR, SOLUTION_DIR)

ORI_MAP = {'^': 0, '>': 1, 'v': 2, '<': 3}
ORI_CHARS = ['^', '>', 'v', '<']


def load_ps():
    connectivity_file = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
    ncc_report_file = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')

    with open(connectivity_file, 'r') as f:
        connectivity_raw = json.load(f)

    ps_raw = {}
    for pid_str, fits_list in connectivity_raw.items():
        pid = int(pid_str)
        ps_raw[pid] = [[], [], [], []]
        for i in range(4):
            for m in fits_list[i]:
                ps_raw[pid][i].append((m['pid'], m['si'], m['error']))

    ncc_lookup = {}
    if os.path.exists(ncc_report_file):
        with open(ncc_report_file, 'r') as f:
            report = json.load(f)
        for pid_str, sides in report.items():
            pid = int(pid_str)
            for si, matches in enumerate(sides):
                for m in matches:
                    key = (pid, si, m['pid'], m['si'])
                    ncc_lookup[key] = {
                        'ncc': m['ncc'],
                        'reject': m.get('reject', False)
                    }

    ps_ncc = {}
    for pid, sides in ps_raw.items():
        ps_ncc[pid] = [[], [], [], []]
        for si in range(4):
            ncc_list = []
            fb_list = []
            for other_pid, other_si, error in sides[si]:
                key = (pid, si, other_pid, other_si)
                rev_key = (other_pid, other_si, pid, si)
                info = ncc_lookup.get(key) or ncc_lookup.get(rev_key)
                if info and not info['reject'] and info['ncc'] > 0:
                    composite = error / (info['ncc'] * 1000.0)
                    ncc_list.append((other_pid, other_si, composite))
                else:
                    fb_list.append((other_pid, other_si, error))
            ncc_list.sort(key=lambda x: x[2])
            fb_list.sort(key=lambda x: x[2])
            ps_ncc[pid][si] = ncc_list + fb_list

    return ps_raw, ps_ncc


def get_cost(ps, pid_a, ori_a, pid_b, ori_b, direction):
    a_side = (direction - ori_a) % 4
    b_side = (OPPOSITE[direction] - ori_b) % 4
    for n_pid, n_side, error in ps.get(pid_a, [[], [], [], []])[a_side]:
        if n_pid == pid_b and n_side == b_side:
            return error
    return None


def get_neighbors(ps, pid, ori, direction):
    a_side = (direction - ori) % 4
    results = []
    for n_pid, n_side, error in ps.get(pid, [[], [], [], []])[a_side]:
        results.append((n_pid, n_side, error))
    return results


def main():
    print("=" * 60)
    print("LEFT Edge Diagnostic: Force 4 -> 137 adjacency")
    print("=" * 60)

    ps_raw, ps_ncc = load_ps()

    left_pieces = [
        (0, 0, 134, ORI_MAP['v']),
        (0, 1, 78, ORI_MAP['<']),
        (0, 2, 4, ORI_MAP['<']),
        (0, 3, 79, ORI_MAP['v']),
        (0, 4, 136, ORI_MAP['<']),
        (0, 5, 144, ORI_MAP['v']),
        (0, 6, 137, ORI_MAP['>']),
        (0, 7, 141, ORI_MAP['>']),
        (0, 8, 112, ORI_MAP['>']),
        (0, 9, 127, ORI_MAP['>']),
    ]

    direction = BOTTOM

    print("\n--- Current LEFT edge ---")
    for i in range(len(left_pieces) - 1):
        _, _, pid_a, ori_a = left_pieces[i]
        _, _, pid_b, ori_b = left_pieces[i + 1]
        cost_ncc = get_cost(ps_ncc, pid_a, ori_a, pid_b, ori_b, direction)
        cost_raw = get_cost(ps_raw, pid_a, ori_a, pid_b, ori_b, direction)
        print(f"  {pid_a}{ORI_CHARS[ori_a]} -> {pid_b}{ORI_CHARS[ori_b]}: "
              f"NCC={cost_ncc}, RAW={cost_raw}")

    corner_top = left_pieces[0]
    corner_bot = left_pieces[-1]
    interior = left_pieces[1:-1]
    n = len(interior)

    print(f"\n--- Constraint: 4 must be immediately before 137 ---")

    idx_4 = None
    idx_137 = None
    for i, (_, _, pid, _) in enumerate(interior):
        if pid == 4:
            idx_4 = i
        if pid == 137:
            idx_137 = i
    print(f"  piece 4 at interior index {idx_4}, piece 137 at interior index {idx_137}")

    cost_4_to_137_ncc = get_cost(ps_ncc, 4, ORI_MAP['<'], 137, ORI_MAP['>'], direction)
    cost_4_to_137_raw = get_cost(ps_raw, 4, ORI_MAP['<'], 137, ORI_MAP['>'], direction)
    print(f"  4< -> 137> NCC cost: {cost_4_to_137_ncc}")
    print(f"  4< -> 137> RAW cost: {cost_4_to_137_raw}")

    if cost_4_to_137_raw is None and cost_4_to_137_ncc is None:
        print("  ERROR: 4 and 137 have NO connection at all with current orientations!")
        print("  Checking all possible orientations...")

        for ori_4 in range(4):
            for ori_137 in range(4):
                c = get_cost(ps_raw, 4, ori_4, 137, ori_137, direction)
                if c is not None:
                    print(f"    4{ORI_CHARS[ori_4]} -> 137{ORI_CHARS[ori_137]}: RAW={c}")
        return

    print(f"\n--- Full connectivity matrix (RAW, direction=BOTTOM) ---")
    print(f"  Each cell: what piece A (row) can connect to piece B (col) going DOWN")
    print(f"  Format: cost (or '-' if no connection)")

    all_pids = [corner_top[2]] + [p[2] for p in interior] + [corner_bot[2]]
    all_oris = [corner_top[3]] + [p[3] for p in interior] + [corner_bot[3]]

    header = f"  {'':>6}"
    for pid in all_pids:
        header += f" {pid:>6}"
    print(header)

    for i, (pid_a, ori_a) in enumerate(zip(all_pids, all_oris)):
        row = f"  {pid_a}{ORI_CHARS[ori_a]:>5}"
        for j, (pid_b, ori_b) in enumerate(zip(all_pids, all_oris)):
            if i == j:
                row += f" {'---':>6}"
                continue
            c = get_cost(ps_raw, pid_a, ori_a, pid_b, ori_b, direction)
            if c is not None:
                row += f" {c:>6.0f}"
            else:
                row += f" {'-':>6}"
        print(row)

    print(f"\n--- TSP with constraint: 4 must immediately precede 137 ---")
    INF = float('inf')

    cost_from_start = []
    for i, (_, _, pid, ori) in enumerate(interior):
        c = get_cost(ps_raw, corner_top[2], corner_top[3], pid, ori, direction)
        cost_from_start.append(c if c is not None else INF)

    cost_to_end = []
    for i, (_, _, pid, ori) in enumerate(interior):
        c = get_cost(ps_raw, pid, ori, corner_bot[2], corner_bot[3], direction)
        cost_to_end.append(c if c is not None else INF)

    cost_between = [[INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = get_cost(ps_raw, interior[i][2], interior[i][3],
                         interior[j][2], interior[j][3], direction)
            cost_between[i][j] = c if c is not None else INF

    full_mask = (1 << n) - 1
    dist = {}
    pq = []

    for i in range(n):
        if cost_from_start[i] >= INF:
            continue
        mask = full_mask ^ (1 << i)
        c = cost_from_start[i]
        state = (i, mask)
        if state not in dist or c < dist[state]:
            dist[state] = c
            heapq.heappush(pq, (c, i, mask, [i]))

    best_cost = INF
    best_path = None

    while pq:
        cost, last, mask, path = heapq.heappop(pq)

        state = (last, mask)
        if cost > dist.get(state, INF):
            continue

        if mask == 0:
            total = cost + cost_to_end[last]
            if total < best_cost:
                has_4_before_137 = False
                for k in range(len(path) - 1):
                    if interior[path[k]][2] == 4 and interior[path[k + 1]][2] == 137:
                        has_4_before_137 = True
                        break
                if has_4_before_137:
                    best_cost = total
                    best_path = path
            continue

        for j in range(n):
            if not (mask & (1 << j)):
                continue
            if cost_between[last][j] >= INF:
                continue

            new_mask = mask ^ (1 << j)
            new_cost = cost + cost_between[last][j]
            state = (j, new_mask)

            if state not in dist or new_cost < dist[state]:
                dist[state] = new_cost
                heapq.heappush(pq, (new_cost, j, new_mask, path + [j]))

    if best_path is None:
        print("  NO valid path with 4->137 constraint found!")
        print("  Trying without constraint to see what's possible...")

        dist2 = {}
        pq2 = []
        for i in range(n):
            if cost_from_start[i] >= INF:
                continue
            mask = full_mask ^ (1 << i)
            c = cost_from_start[i]
            state = (i, mask)
            if state not in dist2 or c < dist2[state]:
                dist2[state] = c
                heapq.heappush(pq2, (c, i, mask, [i]))

        best2 = INF
        path2 = None
        while pq2:
            cost, last, mask, path = heapq.heappop(pq2)
            state = (last, mask)
            if cost > dist2.get(state, INF):
                continue
            if mask == 0:
                total = cost + cost_to_end[last]
                if total < best2:
                    best2 = total
                    path2 = path
                continue
            for j in range(n):
                if not (mask & (1 << j)):
                    continue
                if cost_between[last][j] >= INF:
                    continue
                new_mask = mask ^ (1 << j)
                new_cost = cost + cost_between[last][j]
                state = (j, new_mask)
                if state not in dist2 or new_cost < dist2[state]:
                    dist2[state] = new_cost
                    heapq.heappush(pq2, (new_cost, j, new_mask, path + [j]))

        if path2:
            print(f"  Best without constraint: cost={best2}")
            names = [interior[i][2] for i in path2]
            print(f"  Path: {corner_top[2]} -> {' -> '.join(str(p) for p in names)} -> {corner_bot[2]}")

            print(f"\n  Checking each pair in unconstrained path:")
            prev_pid, prev_ori = corner_top[2], corner_top[3]
            for idx in path2:
                pid, ori = interior[idx][2], interior[idx][3]
                c = get_cost(ps_raw, prev_pid, prev_ori, pid, ori, direction)
                print(f"    {prev_pid}{ORI_CHARS[prev_ori]} -> {pid}{ORI_CHARS[ori]}: {c}")
                prev_pid, prev_ori = pid, ori
            c = get_cost(ps_raw, prev_pid, prev_ori, corner_bot[2], corner_bot[3], direction)
            print(f"    {prev_pid}{ORI_CHARS[prev_ori]} -> {corner_bot[2]}{ORI_CHARS[corner_bot[3]]}: {c}")

        print("\n  Diagnosing why 4->137 constraint fails...")
        print("  Checking reachability: can 137 be reached from 4?")
        c = get_cost(ps_raw, 4, ORI_MAP['<'], 137, ORI_MAP['>'], direction)
        print(f"    4< -> 137> (DOWN): {c}")

        print("  What can follow 4? (4< going DOWN)")
        for i, (_, _, pid, ori) in enumerate(interior):
            if pid == 4:
                continue
            c = get_cost(ps_raw, 4, ORI_MAP['<'], pid, ori, direction)
            if c is not None:
                print(f"    4< -> {pid}{ORI_CHARS[ori]}: RAW={c}")

        print("  What can precede 137? (going DOWN to 137>)")
        for i, (_, _, pid, ori) in enumerate(interior):
            if pid == 137:
                continue
            c = get_cost(ps_raw, pid, ori, 137, ORI_MAP['>'], direction)
            if c is not None:
                print(f"    {pid}{ORI_CHARS[ori]} -> 137>: RAW={c}")

        print("  What can follow 137? (137> going DOWN)")
        for i, (_, _, pid, ori) in enumerate(interior):
            if pid == 137:
                continue
            c = get_cost(ps_raw, 137, ORI_MAP['>'], pid, ori, direction)
            if c is not None:
                print(f"    137> -> {pid}{ORI_CHARS[ori]}: RAW={c}")

        print("  What can precede 4? (going DOWN to 4<)")
        for i, (_, _, pid, ori) in enumerate(interior):
            if pid == 4:
                continue
            c = get_cost(ps_raw, pid, ori, 4, ORI_MAP['<'], direction)
            if c is not None:
                print(f"    {pid}{ORI_CHARS[ori]} -> 4<: RAW={c}")
    else:
        names = [interior[i][2] for i in best_path]
        print(f"  FOUND path with 4->137 constraint! cost={best_cost}")
        print(f"  Path: {corner_top[2]} -> {' -> '.join(str(p) for p in names)} -> {corner_bot[2]}")

        prev_pid, prev_ori = corner_top[2], corner_top[3]
        for idx in best_path:
            pid, ori = interior[idx][2], interior[idx][3]
            c_raw = get_cost(ps_raw, prev_pid, prev_ori, pid, ori, direction)
            c_ncc = get_cost(ps_ncc, prev_pid, prev_ori, pid, ori, direction)
            marker = " <-- 4->137" if (prev_pid == 4 and pid == 137) else ""
            print(f"    {prev_pid}{ORI_CHARS[prev_ori]} -> {pid}{ORI_CHARS[ori]}: "
                  f"RAW={c_raw}, NCC={c_ncc}{marker}")
            prev_pid, prev_ori = pid, ori
        c_raw = get_cost(ps_raw, prev_pid, prev_ori, corner_bot[2], corner_bot[3], direction)
        c_ncc = get_cost(ps_ncc, prev_pid, prev_ori, corner_bot[2], corner_bot[3], direction)
        print(f"    {prev_pid}{ORI_CHARS[prev_ori]} -> {corner_bot[2]}{ORI_CHARS[corner_bot[3]]}: "
              f"RAW={c_raw}, NCC={c_ncc}")


if __name__ == '__main__':
    main()
