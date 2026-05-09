import os
import sys
import json
import heapq

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.board import Board, OPPOSITE, TOP, RIGHT, BOTTOM, LEFT
from common import output as board_output
from solve_display import generate_assembly_png

from config import get_output_dir, get_deduped_path, get_connectivity_path, get_solution_path
from solver_utils import load_connectivity_and_ncc, get_oriented_cost, ORI_CHARS

OUTPUT_DIR = get_output_dir()
DEDUPED_PATH = get_deduped_path()
CONNECTIVITY_PATH = get_connectivity_path()
SOLUTION_PATH = get_solution_path()


def get_cost(ps, pid_a, ori_a, pid_b, ori_b, direction):
    return get_oriented_cost(ps, pid_a, ori_a, pid_b, ori_b, direction)


def solve_tsp(ps, start_pid, start_ori, end_pid, end_ori, interior, direction):
    n = len(interior)
    if n == 0:
        c = get_cost(ps, start_pid, start_ori, end_pid, end_ori, direction)
        return [], c if c is not None else float('inf')

    INF = float('inf')
    cost_from_start = []
    for i, (pid, ori) in enumerate(interior):
        c = get_cost(ps, start_pid, start_ori, pid, ori, direction)
        cost_from_start.append(c if c is not None else INF)

    cost_to_end = []
    for i, (pid, ori) in enumerate(interior):
        c = get_cost(ps, pid, ori, end_pid, end_ori, direction)
        cost_to_end.append(c if c is not None else INF)

    cost_between = [[INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = get_cost(ps, interior[i][0], interior[i][1],
                         interior[j][0], interior[j][1], direction)
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
        return None, INF

    result = [interior[i] for i in best_path]
    return result, best_cost


def print_edge(ps, start_pid, start_ori, pieces, end_pid, end_ori, direction):
    prev_pid, prev_ori = start_pid, start_ori
    total = 0
    for pid, ori in pieces:
        c = get_cost(ps, prev_pid, prev_ori, pid, ori, direction)
        print(f"    {prev_pid}{ORI_CHARS[prev_ori]} -> {pid}{ORI_CHARS[ori]}: {c}")
        if c is not None:
            total += c
        prev_pid, prev_ori = pid, ori
    c = get_cost(ps, prev_pid, prev_ori, end_pid, end_ori, direction)
    print(f"    {prev_pid}{ORI_CHARS[prev_ori]} -> {end_pid}{ORI_CHARS[end_ori]}: {c}")
    if c is not None:
        total += c
    print(f"    TOTAL: {total}")
    return total


def main():
    print("=" * 60)
    print("Full Border Re-optimization with swapped corners 70<->127")
    print("=" * 60)

    ps_raw, ps_ncc = load_connectivity_and_ncc(CONNECTIVITY_PATH)

    TL = (134, 2)
    TR = (127, 1)
    BR = (71, 0)
    BL = (70, 0)

    top_interior = [(95, 2), (34, 3), (24, 0), (12, 3), (41, 1), (92, 1), (88, 0), (6, 0)]
    right_interior = [(138, 2), (2, 3), (17, 1), (43, 2), (130, 2), (72, 0), (140, 3), (125, 1)]
    bottom_interior = [(20, 0), (126, 3), (35, 2), (124, 2), (118, 1), (26, 3), (1, 0), (117, 1)]
    left_interior = [(78, 3), (4, 3), (79, 2), (136, 3), (144, 2), (137, 1), (141, 1), (112, 1)]

    edges = {
        'top':    {'start': TL, 'end': TR, 'interior': top_interior, 'direction': RIGHT},
        'right':  {'start': TR, 'end': BR, 'interior': right_interior, 'direction': BOTTOM},
        'bottom': {'start': BL, 'end': BR, 'interior': bottom_interior, 'direction': RIGHT},
        'left':   {'start': TL, 'end': BL, 'interior': left_interior, 'direction': BOTTOM},
    }

    print(f"\nCorners: TL={TL}, TR={TR}, BR={BR}, BL={BL}")

    for phase_name, ps in [("NCC", ps_ncc), ("RAW", ps_raw)]:
        print(f"\n{'=' * 60}")
        print(f"Phase: {phase_name} optimization")
        print(f"{'=' * 60}")

        optimized = {}
        for name, info in edges.items():
            start_pid, start_ori = info['start']
            end_pid, end_ori = info['end']
            direction = info['direction']
            interior = info['interior']

            old_cost = 0
            prev_pid, prev_ori = start_pid, start_ori
            for pid, ori in interior:
                c = get_cost(ps, prev_pid, prev_ori, pid, ori, direction)
                old_cost += c if c is not None else float('inf')
                prev_pid, prev_ori = pid, ori
            c = get_cost(ps, prev_pid, prev_ori, end_pid, end_ori, direction)
            old_cost += c if c is not None else float('inf')

            result, new_cost = solve_tsp(ps, start_pid, start_ori, end_pid, end_ori,
                                         interior, direction)

            if result is not None and new_cost < old_cost:
                print(f"\n  [{name.upper()}] IMPROVED: {old_cost:.4f} -> {new_cost:.4f}")
                optimized[name] = result
            elif result is not None:
                print(f"\n  [{name.upper()}] SAME: {old_cost:.4f}")
                optimized[name] = result
            else:
                print(f"\n  [{name.upper()}] NO VALID PATH (original cost={old_cost:.4f})")
                optimized[name] = interior

            names = [f'{p}{ORI_CHARS[o]}' for p, o in optimized[name]]
            print(f"  Path: {start_pid}{ORI_CHARS[start_ori]} -> {' -> '.join(names)} -> {end_pid}{ORI_CHARS[end_ori]}")
            print_edge(ps, start_pid, start_ori, optimized[name], end_pid, end_ori, direction)

        print(f"\n--- Rebuilding board with {phase_name} optimized edges ---")
        board = Board(width=10, height=10)

        def place_edge(start, interior_pieces, direction_val):
            pieces = [start] + interior_pieces
            for i, (pid, ori) in enumerate(pieces):
                if pid not in board._placed_piece_ids:
                    if direction_val == RIGHT:
                        board.place(pid, ps[pid], i, 0, ori)
                    elif direction_val == BOTTOM:
                        board.place(pid, ps[pid], 9, i, ori)

        place_edge(TL, optimized['top'], RIGHT)
        place_edge(TR, optimized['right'], BOTTOM)

        bl_pieces = [BL] + list(reversed(optimized['bottom'][:-1])) + [optimized['bottom'][-1]]
        for i, (pid, ori) in enumerate(optimized['bottom']):
            if pid not in board._placed_piece_ids:
                board.place(pid, ps[pid], i, 9, ori)

        for i, (pid, ori) in enumerate(optimized['left']):
            if pid not in board._placed_piece_ids:
                board.place(pid, ps[pid], 0, i, ori)

        print(board)
        print(f"Placed: {board.placed_count} pieces")

        out_dir = os.path.join(SOLUTION_PATH, f'swap_opt_{phase_name.lower()}')
        os.makedirs(out_dir, exist_ok=True)
        board_output.generate_solution_grid(board, out_dir)
        board_output.generate_solution_svg(board, DEDUPED_PATH, out_dir)
        generate_assembly_png(board, DEDUPED_PATH, OUTPUT_DIR, os.path.join(out_dir, 'assembly.png'))
        print(f"Saved to {out_dir}/")


if __name__ == '__main__':
    main()
