import os
import sys
import json

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR
from common.board import Board, OPPOSITE, TOP, RIGHT, BOTTOM, LEFT
from common import output as board_output
from solve_display import generate_assembly_png

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)
SOLUTION_PATH = os.path.join(OUTPUT_DIR, SOLUTION_DIR)

ORI_MAP = {'^': 0, '>': 1, 'v': 2, '<': 3}


def load_ps():
    connectivity_file = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
    with open(connectivity_file, 'r') as f:
        connectivity_raw = json.load(f)
    ps = {}
    for pid_str, fits_list in connectivity_raw.items():
        pid = int(pid_str)
        ps[pid] = [[], [], [], []]
        for i in range(4):
            for m in fits_list[i]:
                ps[pid][i].append((m['pid'], m['si'], m['error']))
    return ps


def main():
    ps = load_ps()
    board = Board(width=10, height=10)

    top_edge = [
        (0, 0, 134, 2),
        (1, 0, 95, 2),
        (2, 0, 34, 3),
        (3, 0, 24, 0),
        (4, 0, 12, 3),
        (5, 0, 41, 1),
        (6, 0, 92, 1),
        (7, 0, 88, 0),
        (8, 0, 6, 0),
        (9, 0, 127, 1),
    ]

    right_edge = [
        (9, 0, 127, 1),
        (9, 1, 138, 2),
        (9, 2, 2, 3),
        (9, 3, 17, 1),
        (9, 4, 43, 2),
        (9, 5, 130, 2),
        (9, 6, 72, 0),
        (9, 7, 140, 3),
        (9, 8, 125, 1),
        (9, 9, 71, 0),
    ]

    bottom_edge = [
        (0, 9, 70, 0),
        (1, 9, 20, 0),
        (2, 9, 126, 3),
        (3, 9, 35, 2),
        (4, 9, 124, 2),
        (5, 9, 118, 1),
        (6, 9, 26, 3),
        (7, 9, 1, 0),
        (8, 9, 117, 1),
        (9, 9, 71, 0),
    ]

    left_edge = [
        (0, 0, 134, 2),
        (0, 1, 78, 3),
        (0, 2, 4, 3),
        (0, 3, 79, 2),
        (0, 4, 136, 3),
        (0, 5, 144, 2),
        (0, 6, 137, 1),
        (0, 7, 141, 1),
        (0, 8, 112, 1),
        (0, 9, 70, 0),
    ]

    all_edges = top_edge + right_edge[1:] + bottom_edge + left_edge[1:-1]

    placed = set()
    for x, y, pid, ori in all_edges:
        if pid not in placed:
            board.place(pid, ps[pid], x, y, ori)
            placed.add(pid)

    print(board)
    print(f"Placed: {board.placed_count} pieces")

    out_dir = os.path.join(SOLUTION_PATH, 'swap_70_127')
    os.makedirs(out_dir, exist_ok=True)

    board_output.generate_solution_grid(board, out_dir)
    board_output.generate_solution_svg(board, DEDUPED_PATH, out_dir)
    generate_assembly_png(board, DEDUPED_PATH, OUTPUT_DIR, os.path.join(out_dir, 'assembly.png'))

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == '__main__':
    main()
