import os
import sys
import json
from collections import Counter

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR
from common import board, output
from common.board import build_from_corner

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)
SOLUTION_PATH = os.path.join(OUTPUT_DIR, SOLUTION_DIR)

NCC_PRIORITY_WEIGHT = 1000.0


def trace_border_edge(start_pid, start_out_side, ps, piece_edge_info):
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


def load_ncc_report(report_path):
    if not os.path.exists(report_path):
        return None

    with open(report_path, 'r') as f:
        report = json.load(f)

    ncc_lookup = {}
    for pid_str, sides in report.items():
        pid = int(pid_str)
        for si, matches in enumerate(sides):
            for m in matches:
                if m.get('reject', False):
                    continue
                key = (pid, si, m['pid'], m['si'])
                ncc_lookup[key] = m['ncc']

    return ncc_lookup


def build_phase1_connectivity(ps_raw, ncc_lookup):
    print("\n  Phase 1: Building NCC-priority connectivity (only NCC-verified pairs)...")
    ps = {}
    kept_count = 0
    dropped_count = 0

    for pid, sides in ps_raw.items():
        ps[pid] = [[], [], [], []]
        for si in range(4):
            for other_pid, other_si, error in sides[si]:
                ncc_key = (pid, si, other_pid, other_si)
                rev_key = (other_pid, other_si, pid, si)
                ncc = ncc_lookup.get(ncc_key) or ncc_lookup.get(rev_key)
                if ncc is not None and ncc > 0:
                    composite = error / (ncc * NCC_PRIORITY_WEIGHT)
                    ps[pid][si].append((other_pid, other_si, composite))
                    kept_count += 1
                else:
                    dropped_count += 1

    print(f"    Kept {kept_count} NCC-verified pairs, dropped {dropped_count} unverified")
    return ps


def build_phase2_connectivity(ps_raw, ncc_lookup):
    print("\n  Phase 2: Building NCC-enhanced connectivity (all pairs, NCC composite score)...")
    ps = {}
    ncc_count = 0
    fallback_count = 0

    for pid, sides in ps_raw.items():
        ps[pid] = [[], [], [], []]
        for si in range(4):
            for other_pid, other_si, error in sides[si]:
                ncc_key = (pid, si, other_pid, other_si)
                rev_key = (other_pid, other_si, pid, si)
                ncc = ncc_lookup.get(ncc_key) or ncc_lookup.get(rev_key)
                if ncc is not None and ncc > 0:
                    composite = error / (ncc * NCC_PRIORITY_WEIGHT)
                    ps[pid][si].append((other_pid, other_si, composite))
                    ncc_count += 1
                else:
                    ps[pid][si].append((other_pid, other_si, error))
                    fallback_count += 1

    print(f"    {ncc_count} pairs with NCC composite, {fallback_count} pairs with original error")
    return ps


def try_solve_from_corners(ps, corners, pw, ph, edge_length, phase_name):
    corners_sorted = sorted(
        corners,
        key=lambda c: sum(len(fits) for fits in ps[c]),
        reverse=True
    )

    for i, corner_id in enumerate(corners_sorted):
        print(f"\n  [{phase_name}] Trying corner {i}: piece {corner_id}...")
        try:
            solution = build_from_corner(
                ps, start_piece_id=corner_id,
                edge_length=edge_length,
                puzzle_width=pw, puzzle_height=ph
            )
            if solution.placed_count == pw * ph:
                print(f"  [{phase_name}] SUCCESS with corner {corner_id}!")
                return solution
            else:
                print(f"  [{phase_name}] Corner {corner_id}: placed {solution.placed_count}/{pw * ph}")
        except Exception as e:
            print(f"  [{phase_name}] Corner {corner_id} failed: {e}")
            continue

    return None


def main():
    print("=" * 60)
    print("Puzzle Solving Pipeline (NCC Priority)")
    print(f"Connectivity: {CONNECTIVITY_PATH}")
    print(f"Piece data:   {DEDUPED_PATH}")
    print(f"Output:       {SOLUTION_PATH}")
    print("=" * 60)

    connectivity_file = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
    edge_info_file = os.path.join(CONNECTIVITY_PATH, 'piece_edge_info.json')
    ncc_report_file = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')

    if not os.path.exists(connectivity_file):
        print(f"Error: connectivity.json not found: {connectivity_file}")
        print(f"Run run_connect.py first to build the connectivity graph.")
        return

    ps_raw = load_connectivity_raw(connectivity_file)

    piece_edge_info = None
    if os.path.exists(edge_info_file):
        with open(edge_info_file, 'r') as f:
            raw = json.load(f)
        piece_edge_info = {int(k): v for k, v in raw.items()}
    else:
        print("Warning: piece_edge_info.json not found, will infer from connectivity")

    ncc_lookup = load_ncc_report(ncc_report_file)
    has_ncc = ncc_lookup is not None and len(ncc_lookup) > 0
    if has_ncc:
        print(f"Loaded NCC report: {len(ncc_lookup)} verified pairs")
    else:
        print("No NCC report found, will use original connectivity only")

    n_pieces = len(ps_raw)

    corners = []
    edges = []
    for pid in ps_raw:
        if piece_edge_info and pid in piece_edge_info:
            edge_count = sum(1 for f in piece_edge_info[pid] if f)
        else:
            edge_count = sum(1 for f in ps_raw[pid] if len(f) == 0)
        if edge_count > 0:
            edges.append(pid)
            if edge_count >= 2:
                corners.append(pid)

    print(f"\nPieces: {n_pieces}")
    print(f"Corners: {len(corners)} -> {corners}")
    print(f"Edge pieces (incl corners): {len(edges)}")

    os.makedirs(SOLUTION_PATH, exist_ok=True)

    w, h = determine_dimensions(ps_raw, corners, piece_edge_info)

    if w is None or w < 2 or h < 2:
        print("Failed to determine dimensions.")
        return

    edge_length = 2 * (w + h) - 4

    print(f"\n{'=' * 60}")
    print(f"Solving with traced dimensions {w} x {h} (NCC priority)")
    print(f"{'=' * 60}")

    solution = None

    if has_ncc:
        # Phase 1: Only NCC-verified pairs, sorted by NCC composite score
        print(f"\n{'=' * 40}")
        print(f"Phase 1: NCC-only connectivity")
        print(f"{'=' * 40}")
        ps_phase1 = build_phase1_connectivity(ps_raw, ncc_lookup)
        solution = try_solve_from_corners(ps_phase1, corners, w, h, edge_length, "Phase1-NCC")

        if solution is None:
            # Phase 2: All pairs, NCC-enhanced scoring
            print(f"\n{'=' * 40}")
            print(f"Phase 2: NCC-enhanced connectivity (with fallback)")
            print(f"{'=' * 40}")
            ps_phase2 = build_phase2_connectivity(ps_raw, ncc_lookup)
            solution = try_solve_from_corners(ps_phase2, corners, w, h, edge_length, "Phase2-Enhanced")

    if solution is None:
        # Phase 3 (or Phase 1 if no NCC): Original connectivity
        print(f"\n{'=' * 40}")
        print(f"{'Phase 3' if has_ncc else 'Phase 1'}: Original connectivity")
        print(f"{'=' * 40}")
        solution = try_solve_from_corners(ps_raw, corners, w, h, edge_length, "Original")

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
