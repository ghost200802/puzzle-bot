import os
import sys
import json
import heapq

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))

from common.board import OPPOSITE, TOP, RIGHT, BOTTOM, LEFT

from pipline.config import get_connectivity_path
from pipline.solver_utils import load_connectivity_and_ncc, get_oriented_cost, ORI_MAP, ORI_CHARS

CONNECTIVITY_PATH = get_connectivity_path()


def solve_tsp(ps, corner_top, corner_bot, interior, direction, constraint_fn=None, label=""):
    n = len(interior)
    INF = float('inf')

    cost_from_start = []
    for i, (_, _, pid, ori) in enumerate(interior):
        c = get_oriented_cost(ps, corner_top[2], corner_top[3], pid, ori, direction)
        cost_from_start.append(c if c is not None else INF)

    cost_to_end = []
    for i, (_, _, pid, ori) in enumerate(interior):
        c = get_oriented_cost(ps, pid, ori, corner_bot[2], corner_bot[3], direction)
        cost_to_end.append(c if c is not None else INF)

    cost_between = [[INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = get_oriented_cost(ps, interior[i][2], interior[i][3],
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
                if constraint_fn is None or constraint_fn(path, interior):
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
        print(f"  [{label}] No valid path found!")
        return None, INF

    names = [interior[i][2] for i in best_path]
    print(f"  [{label}] cost={best_cost:.6f}")
    print(f"  Path: {corner_top[2]} -> {' -> '.join(str(p) for p in names)} -> {corner_bot[2]}")
    return best_path, best_cost


def print_path_detail(ps_raw, ps_ncc, corner_top, corner_bot, interior, path, direction):
    prev_pid, prev_ori = corner_top[2], corner_top[3]
    total_ncc = 0
    total_raw = 0
    for idx in path:
        pid, ori = interior[idx][2], interior[idx][3]
        c_raw = get_oriented_cost(ps_raw, prev_pid, prev_ori, pid, ori, direction)
        c_ncc = get_oriented_cost(ps_ncc, prev_pid, prev_ori, pid, ori, direction)
        print(f"    {prev_pid}{ORI_CHARS[prev_ori]} -> {pid}{ORI_CHARS[ori]}: "
              f"RAW={c_raw}, NCC={c_ncc}")
        total_ncc += c_ncc if c_ncc else 0
        total_raw += c_raw if c_raw else 0
        prev_pid, prev_ori = pid, ori
    c_raw = get_oriented_cost(ps_raw, prev_pid, prev_ori, corner_bot[2], corner_bot[3], direction)
    c_ncc = get_oriented_cost(ps_ncc, prev_pid, prev_ori, corner_bot[2], corner_bot[3], direction)
    print(f"    {prev_pid}{ORI_CHARS[prev_ori]} -> {corner_bot[2]}{ORI_CHARS[corner_bot[3]]}: "
          f"RAW={c_raw}, NCC={c_ncc}")
    total_ncc += c_ncc if c_ncc else 0
    total_raw += c_raw if c_raw else 0
    print(f"    TOTAL: RAW={total_raw}, NCC={total_ncc}")


def main():
    print("=" * 60)
    print("LEFT Edge: Find NCC-optimal arrangement")
    print("=" * 60)

    ps_raw, ps_ncc = load_connectivity_and_ncc(CONNECTIVITY_PATH)

    corner_top = (0, 0, 134, ORI_MAP['v'])
    corner_bot = (0, 9, 127, ORI_MAP['>'])
    interior = [
        (0, 1, 78, ORI_MAP['<']),
        (0, 2, 4, ORI_MAP['<']),
        (0, 3, 79, ORI_MAP['v']),
        (0, 4, 136, ORI_MAP['<']),
        (0, 5, 144, ORI_MAP['v']),
        (0, 6, 137, ORI_MAP['>']),
        (0, 7, 141, ORI_MAP['>']),
        (0, 8, 112, ORI_MAP['>']),
    ]
    direction = BOTTOM

    def constraint_4_137(path, interior):
        for k in range(len(path) - 1):
            if interior[path[k]][2] == 4 and interior[path[k + 1]][2] == 137:
                return True
        return False

    print("\n--- NCC cost matrix (direction=BOTTOM) ---")
    n = len(interior)
    all_pids = [corner_top[2]] + [p[2] for p in interior] + [corner_bot[2]]
    all_oris = [corner_top[3]] + [p[3] for p in interior] + [corner_bot[3]]

    header = f"  {'':>6}"
    for pid in all_pids:
        header += f" {pid:>8}"
    print(header)
    for i, (pid_a, ori_a) in enumerate(zip(all_pids, all_oris)):
        row = f"  {pid_a}{ORI_CHARS[ori_a]:>5}"
        for j, (pid_b, ori_b) in enumerate(zip(all_pids, all_oris)):
            if i == j:
                row += f" {'---':>8}"
                continue
            c = get_oriented_cost(ps_ncc, pid_a, ori_a, pid_b, ori_b, direction)
            if c is not None:
                row += f" {c:>8.4f}"
            else:
                row += f" {'-':>8}"
        print(row)

    print("\n" + "=" * 60)
    print("1. NCC-optimal WITHOUT constraint")
    print("=" * 60)
    path_free, cost_free = solve_tsp(ps_ncc, corner_top, corner_bot, interior, direction, label="NCC Free")
    if path_free:
        print_path_detail(ps_raw, ps_ncc, corner_top, corner_bot, interior, path_free, direction)

    print("\n" + "=" * 60)
    print("2. NCC-optimal WITH 4->137 constraint")
    print("=" * 60)
    path_con, cost_con = solve_tsp(ps_ncc, corner_top, corner_bot, interior, direction,
                                    constraint_fn=constraint_4_137, label="NCC 4->137")
    if path_con:
        print_path_detail(ps_raw, ps_ncc, corner_top, corner_bot, interior, path_con, direction)

    print("\n" + "=" * 60)
    print("3. RAW-optimal WITHOUT constraint")
    print("=" * 60)
    path_raw, cost_raw = solve_tsp(ps_raw, corner_top, corner_bot, interior, direction, label="RAW Free")
    if path_raw:
        print_path_detail(ps_raw, ps_ncc, corner_top, corner_bot, interior, path_raw, direction)

    print("\n" + "=" * 60)
    print("4. RAW-optimal WITH 4->137 constraint")
    print("=" * 60)
    path_raw_con, cost_raw_con = solve_tsp(ps_raw, corner_top, corner_bot, interior, direction,
                                            constraint_fn=constraint_4_137, label="RAW 4->137")
    if path_raw_con:
        print_path_detail(ps_raw, ps_ncc, corner_top, corner_bot, interior, path_raw_con, direction)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = []
    if path_free:
        names = [interior[i][2] for i in path_free]
        results.append(("NCC Free", cost_free, names))
    if path_con:
        names = [interior[i][2] for i in path_con]
        results.append(("NCC 4->137", cost_con, names))
    if path_raw:
        names = [interior[i][2] for i in path_raw]
        results.append(("RAW Free", cost_raw, names))
    if path_raw_con:
        names = [interior[i][2] for i in path_raw_con]
        results.append(("RAW 4->137", cost_raw_con, names))

    for label, cost, names in results:
        print(f"  {label:>15}: cost={cost:.4f} | 134 -> {' -> '.join(str(p) for p in names)} -> 127")


if __name__ == '__main__':
    main()
