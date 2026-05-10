import os
import sys
import json

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))

from common.board import Board, OPPOSITE, TOP, RIGHT, BOTTOM, LEFT

from pipline.config import get_connectivity_path
from pipline.solver_utils import parse_grid, load_connectivity_and_ncc

CONNECTIVITY_PATH = get_connectivity_path()


def get_neighbor_candidates(ps, nb_pid, nb_ori, facing_us):
    nb_side = (facing_us - nb_ori) % 4
    return {n_pid for n_pid, _, _ in ps[nb_pid][nb_side]}


def main():
    _, _, grid = parse_grid(
        r'f:\work_Puzzle_github\puzzle-bot\output\puzzle_new\6_solution\milestone\pct75\solution_grid.txt')
    ps_raw, ps_ncc = load_connectivity_and_ncc(CONNECTIVITY_PATH)

    placed_pids = {v[0] for v in grid.values()}
    remaining = set(range(100)) - placed_pids if max(placed_pids) < 200 else set()

    print(f"Placed: {len(placed_pids)}, Remaining: {len(remaining)}")
    print(f"Remaining pieces: {sorted(remaining)}")

    print("\n--- Analyzing empty positions ---")
    for y in range(10):
        for x in range(10):
            if (x, y) in grid:
                continue

            neighbors = []
            for dx, dy, facing_us in [(-1, 0, RIGHT), (1, 0, LEFT), (0, -1, BOTTOM), (0, 1, TOP)]:
                nx, ny = x + dx, y + dy
                nb = grid.get((nx, ny))
                if nb is not None:
                    neighbors.append((nb[0], nb[1], facing_us))

            if not neighbors:
                continue

            ncc_sets = []
            raw_sets = []
            for nb_pid, nb_ori, facing_us in neighbors:
                ncc_s = get_neighbor_candidates(ps_ncc, nb_pid, nb_ori, facing_us)
                raw_s = get_neighbor_candidates(ps_raw, nb_pid, nb_ori, facing_us)
                ncc_sets.append(ncc_s)
                raw_sets.append(raw_s)

            ncc_common = set.intersection(*ncc_sets) - placed_pids if ncc_sets else set()
            raw_common = set.intersection(*raw_sets) - placed_pids if raw_sets else set()

            status = ""
            if ncc_common:
                status = f"NCC({len(ncc_common)})"
            elif raw_common:
                status = f"RAW_ONLY({len(raw_common)})"
            else:
                status = "DEAD"

            print(f"\n  ({x},{y}) {len(neighbors)} neighbors: {status}")
            for i, (nb_pid, nb_ori, facing_us) in enumerate(neighbors):
                ncc_cands = ncc_sets[i] - placed_pids
                raw_cands = raw_sets[i] - placed_pids
                ncc_str = f"NCC({len(ncc_cands)})"
                raw_str = f"RAW({len(raw_cands)})"
                print(f"    nb {nb_pid} facing={facing_us}: {ncc_str} {raw_str} raw_ids={sorted(raw_cands)[:10]}")


if __name__ == '__main__':
    main()
