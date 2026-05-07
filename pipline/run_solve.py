import os
import sys
import json
from collections import Counter

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR
from common import board, output

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)
SOLUTION_PATH = os.path.join(OUTPUT_DIR, SOLUTION_DIR)


def trace_border_edge(start_pid, start_out_side, ps, piece_edge_info):
    """
    Trace along the border from a corner piece's non-flat side,
    only following edge pieces (pieces with >=1 flat side).
    Stops when reaching another corner (>=2 flat sides).
    Returns the number of pieces along this border edge (including start corner).
    """
    count = 1
    current_pid = start_pid
    out_side = start_out_side
    visited = {start_pid}
    for _ in range(200):
        fits = ps.get(current_pid, [[] for _ in range(4)])
        side_fits = fits[out_side] if out_side < len(fits) else []
        if not side_fits:
            break
        edge_candidates = []
        for other_pid, other_side, error in side_fits:
            if other_pid in visited:
                continue
            nf = piece_edge_info.get(other_pid, [False]*4)
            ec = sum(1 for f in nf if f)
            if ec >= 1:
                edge_candidates.append((other_pid, other_side, error, ec))
        if not edge_candidates:
            break
        edge_candidates.sort(key=lambda x: x[2])
        next_pid, in_side, _, ec = edge_candidates[0]
        visited.add(next_pid)
        count += 1
        if ec >= 2:
            break
        nf = piece_edge_info.get(next_pid, [False]*4)
        out_side = None
        for i in range(4):
            if i == in_side:
                continue
            if nf[i]:
                continue
            if len(ps.get(next_pid, [[] for _ in range(4)])[i]) > 0:
                out_side = i
                break
        if out_side is None:
            break
        current_pid = next_pid
    return count


def determine_dimensions(ps, corners, piece_edge_info):
    print("\nDetermining dimensions by tracing border edges from corners...")
    results = []
    for c in corners:
        ef = piece_edge_info.get(c, [False]*4)
        flat = [i for i, f in enumerate(ef) if f]
        non_flat = [i for i in range(4) if i not in flat]
        if len(non_flat) != 2:
            continue
        d1 = trace_border_edge(c, non_flat[0], ps, piece_edge_info)
        d2 = trace_border_edge(c, non_flat[1], ps, piece_edge_info)
        results.append((c, d1, d2))
        print(f"  Corner {c}: side[{non_flat[0]}]={d1} pcs, side[{non_flat[1]}]={d2} pcs -> {d1}x{d2}")

    if not results:
        return None, None

    dim_pairs = Counter()
    for c, d1, d2 in results:
        dim_pairs[(d1, d2)] += 1
        dim_pairs[(d2, d1)] += 1

    best_pair, count = dim_pairs.most_common(1)[0]
    w, h = best_pair
    print(f"\n  Most common dimension pair: {w} x {h} (from {count} observations)")
    return w, h


def main():
    print("=" * 60)
    print("Puzzle Solving Pipeline")
    print(f"Connectivity: {CONNECTIVITY_PATH}")
    print(f"Piece data:   {DEDUPED_PATH}")
    print(f"Output:       {SOLUTION_PATH}")
    print("=" * 60)

    connectivity_file = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
    edge_info_file = os.path.join(CONNECTIVITY_PATH, 'piece_edge_info.json')

    if not os.path.exists(connectivity_file):
        print(f"Error: connectivity.json not found: {connectivity_file}")
        print(f"Run run_connect.py first to build the connectivity graph.")
        return

    with open(connectivity_file, 'r') as f:
        connectivity = json.load(f)

    piece_edge_info = None
    if os.path.exists(edge_info_file):
        with open(edge_info_file, 'r') as f:
            raw = json.load(f)
        piece_edge_info = {int(k): v for k, v in raw.items()}
    else:
        print("Warning: piece_edge_info.json not found, will infer from connectivity")

    n_pieces = len(connectivity)

    ps = {}
    for pid_str, fits_list in connectivity.items():
        pid = int(pid_str)
        ps[pid] = [[], [], [], []]
        for i in range(4):
            for other_piece_id, other_side_id, error in fits_list[i]:
                ps[pid][i].append((other_piece_id, other_side_id, error))

    corners = []
    edges = []
    for pid in ps:
        if piece_edge_info and pid in piece_edge_info:
            edge_count = sum(1 for f in piece_edge_info[pid] if f)
        else:
            edge_count = sum(1 for f in ps[pid] if len(f) == 0)
        if edge_count > 0:
            edges.append(pid)
            if edge_count >= 2:
                corners.append(pid)

    print(f"\nPieces: {n_pieces}")
    print(f"Corners: {len(corners)} -> {corners}")
    print(f"Edge pieces (incl corners): {len(edges)}")

    os.makedirs(SOLUTION_PATH, exist_ok=True)

    w, h = determine_dimensions(ps, corners, piece_edge_info)

    if w is None or w < 2 or h < 2:
        print("Failed to determine dimensions.")
        return

    print(f"\n{'=' * 60}")
    print(f"Solving with traced dimensions {w} x {h}...")
    print(f"{'=' * 60}")

    solution = None
    try:
        solution = board.build(
            connectivity=connectivity,
            input_path=CONNECTIVITY_PATH,
            output_path=SOLUTION_PATH,
            puzzle_width=w,
            puzzle_height=h,
            piece_edge_info=piece_edge_info,
        )
    except Exception as e:
        print(f"Solve failed: {e}")

    if solution is None:
        print("\nFailed to solve the puzzle.")
        return

    print(f"\n{'=' * 60}")
    print(f"Solution: Board {w} x {h}")
    print(f"Pieces placed: {solution.placed_count}/{w * h}")
    print(f"{'=' * 60}")

    output.print_solution_summary(solution)
    output.generate_solution_grid(solution, SOLUTION_PATH)
    output.generate_solution_svg(solution, DEDUPED_PATH, SOLUTION_PATH)
    output.generate_assembly_guide(solution, SOLUTION_PATH)

    eval_result = board.evaluate_solution(solution)
    print(f"\nSolution evaluation:")
    print(f"  Coverage: {eval_result['coverage']:.1%}")
    print(f"  Matched edges: {eval_result['matched_edges']}/{eval_result['total_possible_edges']}")
    print(f"  Match quality: {eval_result['match_quality']:.1%}")

    eval_path = os.path.join(SOLUTION_PATH, 'evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_result, f, indent=2)
    print(f"  Saved to {eval_path}")

    solution_data = {
        'width': w,
        'height': h,
        'placed_count': solution.placed_count,
        'total': w * h,
    }
    sol_meta_path = os.path.join(SOLUTION_PATH, 'solution_meta.json')
    with open(sol_meta_path, 'w') as f:
        json.dump(solution_data, f, indent=2)

    print(f"\nAll outputs saved to {SOLUTION_PATH}/")


if __name__ == '__main__':
    main()
