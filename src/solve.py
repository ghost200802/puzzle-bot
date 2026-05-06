"""
Given a path to processed piece data, finds a solution
"""

import os
import time
import math

from common import board, connect, util
from common.config import (
    MODE, DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR, TIGHTNESS_DIR,
)


def infer_grid_size(piece_count):
    """
    Infer puzzle grid dimensions from the number of unique pieces.
    Finds the most square-like w * h >= piece_count.
    """
    if piece_count <= 0:
        return 1, 1

    best = None
    best_diff = float('inf')
    for w in range(1, int(math.sqrt(piece_count)) + 3):
        h = math.ceil(piece_count / w)
        if w * h >= piece_count:
            diff = abs(w - h) + (w * h - piece_count) * 0.1
            if diff < best_diff:
                best_diff = diff
                best = (w, h)

    if best is None:
        s = int(math.ceil(math.sqrt(piece_count)))
        best = (s, s)

    return best


def solve(path, start_at=3, puzzle_width=None, puzzle_height=None):
    """
    Given a path to processed piece data, finds a solution.
    If puzzle_width/height are not provided, auto-detects from piece count.
    """
    if puzzle_width is None or puzzle_height is None:
        deduped_dir = os.path.join(path, DEDUPED_DIR)
        piece_files = [f for f in os.listdir(deduped_dir)
                       if f.startswith('side_') and f.endswith('_0.json')]
        piece_count = len(set(
            int(f.split('_')[1]) for f in piece_files
        ))

        if puzzle_width is None or puzzle_height is None:
            auto_w, auto_h = infer_grid_size(piece_count)
            print(f"Auto-detected {piece_count} pieces -> grid {auto_w}x{auto_h}")
            if puzzle_width is None:
                puzzle_width = auto_w
            if puzzle_height is None:
                puzzle_height = auto_h

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
            output.generate_assembly_guide(puzzle, os.path.join(path, SOLUTION_DIR))
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
    use_color = MODE == 'phone'
    connectivity = connect.build(input_path, output_path,
                                 use_image_matching=use_color)
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
