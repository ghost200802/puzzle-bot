"""
Given a path to processed piece data, finds a solution
"""

import os
import time

from common import board, connect, util
from common.config import (
    MODE, DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR, TIGHTNESS_DIR,
)


def solve(path, start_at=3, puzzle_width=None, puzzle_height=None):
    """
    Given a path to processed piece data, finds a solution.
    """
    if start_at <= 5:
        connectivity = _find_connectivity(
            input_path=os.path.join(path, DEDUPED_DIR),
            output_path=os.path.join(path, CONNECTIVITY_DIR),
        )
    else:
        connectivity = None

    if start_at <= 6:
        puzzle = _build_board(
            connectivity=connectivity,
            input_path=os.path.join(path, CONNECTIVITY_DIR),
            output_path=os.path.join(path, SOLUTION_DIR),
            metadata_path=os.path.join(path, DEDUPED_DIR),
            puzzle_width=puzzle_width,
            puzzle_height=puzzle_height,
        )
        # Generate phone mode outputs
        if puzzle is not None and MODE == 'phone':
            from common import output
            output.generate_solution_grid(puzzle, os.path.join(path, SOLUTION_DIR))
            output.print_solution_summary(puzzle)

    if MODE == 'robot' and start_at <= 7:
        from common import move, spacing
        move.move_pieces_into_place(
            puzzle,
            metadata_path=os.path.join(path, DEDUPED_DIR),
            output_path=os.path.join(path, SOLUTION_DIR),
        )
        spacing.tighten_or_relax(
            solution_path=os.path.join(path, SOLUTION_DIR),
            output_path=os.path.join(path, TIGHTNESS_DIR),
        )


def _find_connectivity(input_path, output_path):
    """
    Opens each piece data and finds how each piece could connect to others
    """
    print(f"\n### 4 - Building connectivity ###\n")
    start_time = time.time()
    connectivity = connect.build(input_path, output_path)
    duration = time.time() - start_time
    print(f"Building the graph took {round(duration, 2)} seconds")
    return connectivity


def _build_board(connectivity, input_path, output_path, metadata_path,
                 puzzle_width=None, puzzle_height=None):
    """
    Searches connectivity to find the solution
    """
    print(f"\n### 5 - Finding where each piece goes ###\n")
    start_time = time.time()
    puzzle = board.build(
        connectivity=connectivity,
        input_path=input_path,
        output_path=output_path,
        puzzle_width=puzzle_width,
        puzzle_height=puzzle_height,
    )
    duration = time.time() - start_time
    print(f"Finding where each piece goes took {round(duration, 2)} seconds")
    return puzzle
