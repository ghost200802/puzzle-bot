import os
import sys
import json
import heapq
import time
import shutil

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.board import Board, OPPOSITE, TOP, RIGHT, BOTTOM, LEFT
from common import output as board_output
from solve_display import generate_assembly_png

from config import get_deduped_path, get_connectivity_path, get_solution_path
from solver_utils import (parse_grid, load_connectivity_and_ncc,
                           get_oriented_cost, ORI_MAP, ORI_CHARS)

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = get_deduped_path()
CONNECTIVITY_PATH = get_connectivity_path()
SOLUTION_PATH = get_solution_path()


def score_piece(board, ps_ncc, ps_raw, x, y):
    cell = board.get(x, y)
    if cell is None:
        return None, []
    pid, _, ori = cell
    edges = []
    for dx, dy, direction in [(-1, 0, RIGHT), (1, 0, LEFT), (0, -1, BOTTOM), (0, 1, TOP)]:
        nx, ny = x + dx, y + dy
        nb = board.get(nx, ny)
        if nb is None:
            continue
        nb_pid, _, nb_ori = nb
        ncc_cost = get_oriented_cost(ps_ncc, pid, ori, nb_pid, nb_ori, direction)
        raw_cost = get_oriented_cost(ps_raw, pid, ori, nb_pid, nb_ori, direction)
        edges.append({
            'neighbor': nb_pid,
            'direction': direction,
            'ncc_cost': ncc_cost,
            'raw_cost': raw_cost,
        })
    if not edges:
        return None, edges
    ncc_costs = [e['ncc_cost'] for e in edges if e['ncc_cost'] is not None]
    avg_ncc = sum(ncc_costs) / len(ncc_costs) if ncc_costs else float('inf')
    max_ncc = max(ncc_costs) if ncc_costs else float('inf')
    return max_ncc, edges


def rebuild_board(w, h, grid, ps):
    board = Board(width=w, height=h)
    for (x, y), (pid, ori) in grid.items():
        board.place(pid, ps[pid], x, y, ori)
    return board


def find_candidates_for_pos(board, ps_ncc, ps_raw, x, y, available_pids):
    neighbors = []
    for dx, dy, facing_us in [(-1, 0, RIGHT), (1, 0, LEFT), (0, -1, BOTTOM), (0, 1, TOP)]:
        nx, ny = x + dx, y + dy
        nb = board.get(nx, ny)
        if nb is not None:
            neighbors.append((nb[0], nb[2], facing_us))

    if not neighbors:
        return []

    ncc_sets = []
    for nb_pid, nb_ori, facing_us in neighbors:
        nb_side = (facing_us - nb_ori) % 4
        pids = {n_pid for n_pid, _, _ in ps_ncc[nb_pid][nb_side]}
        if not pids:
            pids = {n_pid for n_pid, _, _ in ps_raw[nb_pid][nb_side]}
        ncc_sets.append(pids)

    common = set.intersection(*ncc_sets) & available_pids

    candidates = []
    for pid in common:
        nb_pid0, nb_ori0, facing_us0 = neighbors[0]
        nb_side0 = (facing_us0 - nb_ori0) % 4
        for n_pid, n_side, _ in ps_ncc[nb_pid0][nb_side0]:
            if n_pid == pid:
                orientation = (OPPOSITE[facing_us0] - n_side) % 4
                ok, _ = board.can_place(pid, ps_ncc[pid], x, y, orientation)
                if ok:
                    max_cost = 0
                    for nb_pid, nb_ori, facing_us in neighbors:
                        nb_side = (facing_us - nb_ori) % 4
                        for n2_pid, n2_side, c in ps_ncc[nb_pid][nb_side]:
                            if n2_pid == pid:
                                max_cost = max(max_cost, c)
                                break
                    candidates.append((max_cost, pid, orientation))
                break

    candidates.sort()
    return candidates


def main():
    print("=" * 60)
    print("Lock-and-Replace Search from pct75")
    print("=" * 60)

    ps_raw, ps_ncc = load_connectivity_and_ncc(CONNECTIVITY_PATH)
    w, h, grid = parse_grid(
        os.path.join(SOLUTION_PATH, 'milestone', 'pct75', 'solution_grid.txt'))
    print(f"Grid: {w}x{h}, {len(grid)} placed pieces")

    board = rebuild_board(w, h, grid, ps_ncc)
    print(f"Board rebuilt: {board.placed_count} pieces")

    print("\n--- Scoring all placed pieces ---")
    piece_scores = {}
    piece_edges_info = {}
    for y in range(h):
        for x in range(w):
            cell = board.get(x, y)
            if cell is None:
                continue
            pid = cell[0]
            max_ncc, edges = score_piece(board, ps_ncc, ps_raw, x, y)
            piece_scores[pid] = max_ncc
            piece_edges_info[pid] = {'pos': (x, y), 'edges': edges}

    scored_list = sorted(piece_scores.items(), key=lambda x: x[1] if x[1] is not None else float('inf'))
    print(f"\n  Score distribution (max NCC cost per piece):")
    bins = {'<0.5': 0, '0.5-1.0': 0, '1.0-2.0': 0, '2.0-5.0': 0, '>5.0': 0, 'no_ncc': 0}
    for pid, score in scored_list:
        if score is None:
            bins['no_ncc'] += 1
        elif score < 0.5:
            bins['<0.5'] += 1
        elif score < 1.0:
            bins['0.5-1.0'] += 1
        elif score < 2.0:
            bins['1.0-2.0'] += 1
        elif score < 5.0:
            bins['2.0-5.0'] += 1
        else:
            bins['>5.0'] += 1
    for k, v in bins.items():
        print(f"  {k:>10}: {v}")

    print(f"\n  Worst 10 pieces:")
    for pid, score in scored_list[-10:]:
        info = piece_edges_info[pid]
        x, y = info['pos']
        edge_strs = []
        for e in info['edges']:
            edge_strs.append(f"{e['neighbor']}({e['ncc_cost']:.3f})" if e['ncc_cost'] else f"{e['neighbor']}(raw={e['raw_cost']})")
        print(f"    pid {pid:>3} at ({x},{y}): max_ncc={score}, edges=[{', '.join(edge_strs)}]")

    LOCK_THRESHOLD = 1.0
    locked_pids = set()
    unlocked_pids = set()
    for pid, score in piece_scores.items():
        if score is not None and score <= LOCK_THRESHOLD:
            locked_pids.add(pid)
        else:
            unlocked_pids.add(pid)

    print(f"\n  Locked (ncc<={LOCK_THRESHOLD}): {len(locked_pids)}")
    print(f"  Unlockable (ncc>{LOCK_THRESHOLD} or no ncc): {len(unlocked_pids)}")
    print(f"  Unlockable pieces: {sorted(unlocked_pids)}")

    all_placed = set(piece_scores.keys())
    remaining_pids = set(range(100)) - all_placed
    print(f"  Remaining (not placed): {len(remaining_pids)}")

    print("\n--- Phase 1: Try filling empty positions without removing any piece ---")
    empty_positions = []
    for y in range(h):
        for x in range(w):
            if board.get(x, y) is None:
                has_nb = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if board.get(x + dx, y + dy) is not None:
                        has_nb = True
                        break
                if has_nb:
                    empty_positions.append((x, y))

    print(f"  Empty positions with neighbors: {len(empty_positions)}")

    filled = 0
    for x, y in empty_positions:
        cands = find_candidates_for_pos(board, ps_ncc, ps_raw, x, y, remaining_pids)
        if cands:
            _, best_pid, best_ori = cands[0]
            board.place(best_pid, ps_ncc[best_pid], x, y, best_ori)
            remaining_pids.discard(best_pid)
            filled += 1
            print(f"    ({x},{y}): placed {best_pid}{ORI_CHARS[best_ori]}")

    print(f"  Phase 1 filled: {filled}, total placed: {board.placed_count}")

    if board.placed_count >= w * h:
        print("\nFULL SOLUTION!")
        print(board)
        return

    print("\n--- Phase 2: Try swapping unlockable pieces ---")

    remaining_empty = []
    for y in range(h):
        for x in range(w):
            if board.get(x, y) is None:
                remaining_empty.append((x, y))

    print(f"  Still empty: {len(remaining_empty)}")

    for x, y in remaining_empty:
        neighbors = []
        for dx, dy, facing_us in [(-1, 0, RIGHT), (1, 0, LEFT), (0, -1, BOTTOM), (0, 1, TOP)]:
            nx, ny = x + dx, y + dy
            nb = board.get(nx, ny)
            if nb is not None:
                neighbors.append((nb[0], nb[2], facing_us))

        if not neighbors:
            continue

        ncc_sets = []
        for nb_pid, nb_ori, facing_us in neighbors:
            nb_side = (facing_us - nb_ori) % 4
            pids = {n_pid for n_pid, _, _ in ps_ncc[nb_pid][nb_side]}
            if not pids:
                pids = {n_pid for n_pid, _, _ in ps_raw[nb_pid][nb_side]}
            ncc_sets.append(pids)

        needed_pids = set.intersection(*ncc_sets)

        available = (needed_pids & remaining_pids)
        if available:
            best_pid = min(available,
                           key=lambda p: min(
                               (c for n_pid, n_side, c in ps_ncc[neighbors[0][0]][(neighbors[0][2] - neighbors[0][1]) % 4] if n_pid == p),
                               default=float('inf')))
            nb_pid0, nb_ori0, facing_us0 = neighbors[0]
            nb_side0 = (facing_us0 - nb_ori0) % 4
            for n_pid, n_side, _ in ps_ncc[nb_pid0][nb_side0]:
                if n_pid == best_pid:
                    orientation = (OPPOSITE[facing_us0] - n_side) % 4
                    ok, _ = board.can_place(best_pid, ps_ncc[best_pid], x, y, orientation)
                    if ok:
                        board.place(best_pid, ps_ncc[best_pid], x, y, orientation)
                        remaining_pids.discard(best_pid)
                        print(f"    ({x},{y}): placed {best_pid}{ORI_CHARS[best_ori]} from remaining")
                    break
            continue

        blocking_locked = needed_pids & locked_pids
        blocking_unlocked = needed_pids & unlocked_pids
        blocking_placed = needed_pids & all_placed - remaining_pids

        swappable = []
        for pid in needed_pids:
            if pid in locked_pids:
                continue
            if pid in all_placed:
                info = piece_edges_info.get(pid)
                if info:
                    swappable.append((piece_scores.get(pid, float('inf')), pid, info['pos']))

        swappable.sort(reverse=True)

        for _, swap_pid, (sx, sy) in swappable:
            swapped_cell = board.get(sx, sy)
            if swapped_cell is None:
                continue

            test_board = Board.copy(board)
            removed_pid = test_board._board[sy][sx][0]
            test_board._board[sy][sx] = None
            test_board._placed_piece_ids.discard(removed_pid)

            swap_remaining = remaining_pids | {removed_pid}

            cands = find_candidates_for_pos(test_board, ps_ncc, ps_raw, x, y, swap_remaining)
            if cands:
                _, new_pid, new_ori = cands[0]

                can_fill_hole = False
                cands_hole = find_candidates_for_pos(test_board, ps_ncc, ps_raw, sx, sy, swap_remaining - {new_pid})
                if cands_hole:
                    can_fill_hole = True

                test_board.place(new_pid, ps_ncc[new_pid], x, y, new_ori)
                swap_remaining.discard(new_pid)

                if can_fill_hole:
                    _, hole_pid, hole_ori = cands_hole[0]
                    test_board.place(hole_pid, ps_ncc[hole_pid], sx, sy, hole_ori)
                    swap_remaining.discard(hole_pid)

                old_score_new = piece_scores.get(new_pid, float('inf'))
                old_score_swap = piece_scores.get(swap_pid, float('inf'))

                new_score_pos = score_piece(test_board, ps_ncc, ps_raw, x, y)[0]
                new_score_hole = score_piece(test_board, ps_ncc, ps_raw, sx, sy)[0] if can_fill_hole else None

                print(f"\n    ({x},{y}): SWAP {swap_pid}@({sx},{sy})[score={old_score_swap:.3f}] -> "
                      f"place {new_pid}[old_score={old_score_new:.3f}]@({x},{y})[new={new_score_pos:.3f}]")

                if can_fill_hole:
                    print(f"      hole ({sx},{sy}): filled with {hole_pid}[new_score={new_score_hole}]")

                board = test_board
                remaining_pids = swap_remaining
                all_placed = set()
                for yy in range(h):
                    for xx in range(w):
                        c = board.get(xx, yy)
                        if c is not None:
                            all_placed.add(c[0])

                for pid_s in list(all_placed):
                    for yy in range(h):
                        for xx in range(w):
                            c = board.get(xx, yy)
                            if c and c[0] == pid_s:
                                piece_scores[pid_s] = score_piece(board, ps_ncc, ps_raw, xx, yy)[0]
                                piece_edges_info[pid_s] = {
                                    'pos': (xx, yy),
                                    'edges': score_piece(board, ps_ncc, ps_raw, xx, yy)[1]
                                }
                                break
                break

    print(f"\n  After Phase 2: {board.placed_count} pieces placed")

    print("\n--- Phase 3: Greedy fill remaining ---")
    improved = True
    passes = 0
    while improved:
        improved = False
        passes += 1
        for y in range(h):
            for x in range(w):
                if board.get(x, y) is not None:
                    continue
                has_nb = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if board.get(x + dx, y + dy) is not None:
                        has_nb = True
                        break
                if not has_nb:
                    continue
                cands = find_candidates_for_pos(board, ps_ncc, ps_raw, x, y, remaining_pids)
                if cands:
                    _, best_pid, best_ori = cands[0]
                    board.place(best_pid, ps_ncc[best_pid], x, y, best_ori)
                    remaining_pids.discard(best_pid)
                    improved = True

    print(f"  Phase 3 passes: {passes}, total placed: {board.placed_count}")
    print(board)

    out_dir = os.path.join(SOLUTION_PATH, 'lock_replace')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    board_output.generate_solution_grid(board, out_dir)
    board_output.generate_solution_svg(board, DEDUPED_PATH, out_dir)
    generate_assembly_png(board, DEDUPED_PATH, OUTPUT_DIR, os.path.join(out_dir, 'assembly.png'))
    print(f"\nOutputs saved to {out_dir}/")


if __name__ == '__main__':
    main()
