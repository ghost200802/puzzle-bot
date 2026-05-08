import os
import sys
import json
import heapq
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR
from common.board import (Board, OPPOSITE, TOP, RIGHT, BOTTOM, LEFT,
                           _orient_start_corner_to_top_left, _get_combined_cost)

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)

connectivity_file = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
ncc_report_file = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')


def load_connectivity_raw(connectivity_file):
    with open(connectivity_file, 'r') as f:
        connectivity = json.load(f)
    ps = {}
    for pid_str, fits_list in connectivity.items():
        pid = int(pid_str)
        ps[pid] = [[], [], [], []]
        for i in range(4):
            for m in fits_list[i]:
                ps[pid][i].append((m['pid'], m['si'], m['error']))
    return ps


def load_ncc_lookup(report_path):
    if not os.path.exists(report_path):
        return {}
    with open(report_path, 'r') as f:
        report = json.load(f)
    lookup = {}
    for pid_str, sides in report.items():
        pid = int(pid_str)
        for si, matches in enumerate(sides):
            for m in matches:
                key = (pid, si, m['pid'], m['si'])
                lookup[key] = {'ncc': m['ncc'], 'reject': m.get('reject', False)}
    return lookup


def build_ncc_ps(ps_raw, ncc_lookup):
    ps = {}
    for pid, sides in ps_raw.items():
        ps[pid] = [[], [], [], []]
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
            ps[pid][si] = ncc_list + fb_list
    return ps


def parse_grid_file(filepath, ps):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    grid = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('---'):
            continue
        row = []
        for token in line.split():
            if token == '-':
                row.append(None)
            else:
                pid = int(token[:-1])
                ori_map = {'^': 0, '>': 1, 'v': 2, '<': 3}
                row.append((pid, ori_map[token[-1]]))
        if row:
            grid.append(row)
    ph = len(grid)
    pw = max(len(r) for r in grid)
    board = Board(width=pw, height=ph)
    for y in range(ph):
        for x in range(pw):
            cell = grid[y][x]
            if cell is not None:
                pid, ori = cell
                board.place(pid, ps[pid], x, y, ori)
    return board


def debug_continue_from_board(board, ps, ps_fallback, max_iter=5000):
    pw, ph = board.width, board.height
    total = pw * ph
    edge_length = 2 * (pw + ph) - 4

    corner_cell = board.get(0, 0)
    start_piece_id = corner_cell[0]
    start_orientation = corner_cell[2]

    direction = RIGHT
    x, y = 1, 0

    priority_q = []
    initial_push = (board, start_piece_id, start_orientation, x, y, direction)
    heapq.heappush(priority_q, (0.0, initial_push))

    iteration = 0
    longest = board.placed_count
    best_board = board
    t_start = time.time()

    while priority_q:
        priority, data = heapq.heappop(priority_q)
        cur_board, start_pid, start_ori, x, y, direction = data

        if iteration % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  iter {iteration:>6d} | cur {cur_board.placed_count:>3d} | best {longest:>3d}/{total} | cost {priority:.4f} | {elapsed:.1f}s")

        if cur_board.placed_count == total:
            print(f"FULL SOLUTION at iter {iteration}")
            break

        if cur_board.placed_count > longest:
            longest = cur_board.placed_count
            best_board = cur_board
            print(f"  *** NEW BEST: {longest}/{total} at iter {iteration}, pos=({x},{y}), dir={direction}")

        iteration += 1

        if iteration > max_iter:
            print(f"  Debug limit reached ({max_iter} iters)")
            break

        placed_neighbor_info = []
        for dx, dy, facing_us in [(-1, 0, RIGHT), (1, 0, LEFT), (0, -1, BOTTOM), (0, 1, TOP)]:
            nx, ny = x + dx, y + dy
            nb = cur_board.get(nx, ny)
            if nb is not None:
                placed_neighbor_info.append((nb[0], nb[2], facing_us))

        pool = []

        if len(placed_neighbor_info) >= 2:
            candidate_sets = []
            for nb_pid, nb_ori, facing_us in placed_neighbor_info:
                nb_side = (facing_us - nb_ori) % 4
                pids = {n_pid for n_pid, _, _ in ps[nb_pid][nb_side]}
                candidate_sets.append(pids)

            common_pids = set.intersection(*candidate_sets) - cur_board._placed_piece_ids

            if not common_pids and ps_fallback is not None:
                fb_sets = []
                for nb_pid, nb_ori, facing_us in placed_neighbor_info:
                    nb_side = (facing_us - nb_ori) % 4
                    pids = {n_pid for n_pid, _, _ in ps_fallback[nb_pid][nb_side]}
                    fb_sets.append(pids)
                fb_common = set.intersection(*fb_sets) - cur_board._placed_piece_ids
                if fb_common:
                    print(f"    ({x},{y}) NCC empty, raw fallback: {sorted(fb_common)}")
                else:
                    print(f"    ({x},{y}) BOTH EMPTY! neighbors: {[(p,f) for p,_,f in placed_neighbor_info]}")
                    for nb_pid, nb_ori, facing_us in placed_neighbor_info:
                        nb_side = (facing_us - nb_ori) % 4
                        ncc_pids = sorted({n for n, _, _ in ps[nb_pid][nb_side]})
                        raw_pids = sorted({n for n, _, _ in ps_fallback[nb_pid][nb_side]})
                        print(f"      nb {nb_pid} side[{nb_side}]: ncc({len(ncc_pids)})={ncc_pids[:15]}...")
                        print(f"      nb {nb_pid} side[{nb_side}]: raw({len(raw_pids)})={raw_pids[:15]}...")
                for pid in fb_common:
                    nb_pid0, nb_ori0, facing_us0 = placed_neighbor_info[0]
                    nb_side0 = (facing_us0 - nb_ori0) % 4
                    orientation = None
                    for n_pid, n_side, _ in ps_fallback[nb_pid0][nb_side0]:
                        if n_pid == pid:
                            orientation = (OPPOSITE[facing_us0] - n_side) % 4
                            break
                    if orientation is None:
                        continue
                    ok, _ = cur_board.can_place(pid, ps[pid], x, y, orientation)
                    if ok:
                        combined = _get_combined_cost(cur_board, pid, ps_fallback, x, y)
                        pool.append((combined + 50.0, pid, orientation))

            for pid in common_pids:
                nb_pid0, nb_ori0, facing_us0 = placed_neighbor_info[0]
                nb_side0 = (facing_us0 - nb_ori0) % 4
                orientation = None
                for n_pid, n_side, _ in ps[nb_pid0][nb_side0]:
                    if n_pid == pid:
                        orientation = (OPPOSITE[facing_us0] - n_side) % 4
                        break
                if orientation is None:
                    continue
                ok, _ = cur_board.can_place(pid, ps[pid], x, y, orientation)
                if ok:
                    combined = _get_combined_cost(cur_board, pid, ps, x, y)
                    pool.append((combined, pid, orientation))
        else:
            idx = (direction - start_ori) % 4
            for nb_pid, nb_si, _ in ps[start_pid][idx]:
                nb_ori = (OPPOSITE[direction] - nb_si) % 4
                ok, _ = cur_board.can_place(nb_pid, ps[nb_pid], x, y, nb_ori)
                if ok:
                    combined = _get_combined_cost(cur_board, nb_pid, ps, x, y)
                    pool.append((combined, nb_pid, nb_ori))

        if not pool and cur_board.placed_count >= 45:
            print(f"    ({x},{y}) NO CANDIDATES at count={cur_board.placed_count}, dir={direction}")
            nb_info = []
            for dx2, dy2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = cur_board.get(x+dx2, y+dy2)
                if nb:
                    nb_info.append(f"({x+dx2},{y+dy2})={nb[0]}")
            print(f"    neighbors: {nb_info}")

        pool.sort()
        for combined, pid, ori in pool:
            next_board = Board.copy(cur_board)
            next_board.place(pid, ps[pid], x, y, ori)
            next_direction = direction
            next_x = x + (1 if next_direction == RIGHT else -1 if next_direction == LEFT else 0)
            next_y = y + (1 if next_direction == BOTTOM else -1 if next_direction == TOP else 0)

            if not next_board.is_available(next_x, next_y):
                next_direction = (direction + 1) % 4
                next_x = x + (1 if next_direction == RIGHT else -1 if next_direction == LEFT else 0)
                next_y = y + (1 if next_direction == BOTTOM else -1 if next_direction == TOP else 0)

            data = [next_board, pid, ori, next_x, next_y, next_direction]
            heapq.heappush(priority_q, (combined, data))

    print(f"\nFinal best: {longest}/{total}")
    print(best_board)
    return best_board


def main():
    ps_raw = load_connectivity_raw(connectivity_file)
    ncc_lookup = load_ncc_lookup(ncc_report_file)
    ps_ncc = build_ncc_ps(ps_raw, ncc_lookup)

    grid_file = os.path.join(OUTPUT_DIR, '6_solution', 'milestone', 'pct45', 'solution_grid.txt')
    print(f"Loading board from {grid_file}")
    board = parse_grid_file(grid_file, ps_raw)
    print(f"Board: {board.width}x{board.height}, placed: {board.placed_count}")
    print(board)

    debug_continue_from_board(board, ps_ncc, ps_raw, max_iter=3000)


if __name__ == '__main__':
    main()
