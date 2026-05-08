import os
import sys
import json
import math
from collections import Counter

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR
from common import board, output as board_output
from common.board import build_from_corner, Board
from solve_display import generate_assembly_png

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)
SOLUTION_PATH = os.path.join(OUTPUT_DIR, SOLUTION_DIR)

NCC_PRIORITY_WEIGHT = 1000.0


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
                lookup[key] = {
                    'ncc': m['ncc'],
                    'reject': m.get('reject', False),
                }
    return lookup


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


def build_ncc_ps(ps_raw, ncc_lookup):
    ps = {}
    ncc_count = 0
    fb_count = 0
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
                    composite = error / (info['ncc'] * NCC_PRIORITY_WEIGHT)
                    ncc_list.append((other_pid, other_si, composite))
                    ncc_count += 1
                else:
                    fb_list.append((other_pid, other_si, error))
                    fb_count += 1
            ncc_list.sort(key=lambda x: x[2])
            fb_list.sort(key=lambda x: x[2])
            ps[pid][si] = ncc_list + fb_list
    print(f"  NCC composite: {ncc_count}, Fallback: {fb_count}")
    return ps


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
            nf = piece_edge_info.get(other_pid, [False] * 4)
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
        nf = piece_edge_info.get(next_pid, [False] * 4)
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


def _infer_from_piece_count(n_pieces, hint=None):
    candidates = []
    for w in range(2, int(n_pieces ** 0.5) + 1):
        if n_pieces % w == 0:
            h = n_pieces // w
            candidates.append((w, h))
    if not candidates:
        return None
    if hint:
        hw, hh = hint
        candidates.sort(key=lambda wh: abs(wh[0] - hw) + abs(wh[1] - hh))
    return candidates[0]


def determine_dimensions(ps, corners, piece_edge_info, n_pieces):
    print("\nDetermining dimensions by tracing border edges from corners...")
    results = []
    for c in corners:
        ef = piece_edge_info.get(c, [False] * 4)
        flat = [i for i, f in enumerate(ef) if f]
        non_flat = [i for i in range(4) if i not in flat]
        if len(non_flat) != 2:
            continue
        d1 = trace_border_edge(c, non_flat[0], ps, piece_edge_info)
        d2 = trace_border_edge(c, non_flat[1], ps, piece_edge_info)
        results.append((c, d1, d2))
        print(f"  Corner {c}: side[{non_flat[0]}]={d1} pcs, side[{non_flat[1]}]={d2} pcs -> {d1}x{d2}")

    if not results:
        inferred = _infer_from_piece_count(n_pieces)
        if inferred:
            return inferred
        return None, None

    dim_pairs = Counter()
    for c, d1, d2 in results:
        dim_pairs[(d1, d2)] += 1
        dim_pairs[(d2, d1)] += 1

    best_pair, count = dim_pairs.most_common(1)[0]
    w, h = best_pair
    print(f"  Traced dimension pair: {w} x {h} = {w * h} (from {count} observations, {n_pieces} pieces)")

    if w * h == n_pieces:
        return w, h

    print(f"  WARNING: {w}x{h}={w*h} != {n_pieces} pieces, inferring from piece count")
    inferred = _infer_from_piece_count(n_pieces, hint=(w, h))
    if inferred:
        iw, ih = inferred
        print(f"  Using inferred: {iw} x {ih} = {iw * ih}")
        return iw, ih

    print(f"  Cannot factor {n_pieces}, keeping traced {w}x{h}")
    return w, h


def main():
    print("=" * 60)
    print("Puzzle Solve (NCC Priority + Spiral Assembly)")
    print("=" * 60)

    connectivity_file = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
    edge_info_file = os.path.join(CONNECTIVITY_PATH, 'piece_edge_info.json')
    ncc_report_file = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')

    with open(connectivity_file, 'r') as f:
        connectivity_raw = json.load(f)
    with open(edge_info_file, 'r') as f:
        piece_edge_info = {int(k): v for k, v in json.load(f).items()}

    ncc_lookup = load_ncc_lookup(ncc_report_file)
    print(f"NCC lookup: {len(ncc_lookup)} entries")

    ps_raw = load_connectivity_raw(connectivity_file)
    n_pieces = len(ps_raw)

    corners = []
    for pid in ps_raw:
        if pid in piece_edge_info:
            edge_count = sum(1 for f in piece_edge_info[pid] if f)
        else:
            edge_count = sum(1 for f in ps_raw[pid] if len(f) == 0)
        if edge_count >= 2:
            corners.append(pid)

    print(f"Pieces: {n_pieces}")
    print(f"Corners: {corners}")

    os.makedirs(SOLUTION_PATH, exist_ok=True)

    print("\n--- Building NCC-enhanced connectivity ---")
    ps_ncc = build_ncc_ps(ps_raw, ncc_lookup)

    w, h = determine_dimensions(ps_ncc, corners, piece_edge_info, n_pieces)
    if w is None or w < 2 or h < 2:
        print("Failed to determine dimensions.")
        return

    edge_length = 2 * (w + h) - 4

    import common.board as board_mod
    board_mod.MAX_ITERATIONS_TO_FIND_BORDER = 50000
    board_mod.MAX_ITERATIONS = 300000000
    print(f"MAX_ITERATIONS_TO_FIND_BORDER: {board_mod.MAX_ITERATIONS_TO_FIND_BORDER}")

    print(f"\n{'=' * 60}")
    print(f"Solving {w}x{h} ({w * h} pieces, {n_pieces} available)")
    print(f"{'=' * 60}")

    corners_sorted = sorted(
        corners,
        key=lambda c: sum(len(fits) for fits in ps_ncc[c]),
        reverse=True
    )

    best_solution = None
    best_count = 0

    for i, corner_id in enumerate(corners_sorted):
        print(f"\n  Trying corner {i}: piece {corner_id}...")
        solution = build_from_corner(
            ps_ncc, start_piece_id=corner_id,
            edge_length=edge_length,
            puzzle_width=pw if (pw := w) else None,
            puzzle_height=h
        )
        if solution.placed_count > best_count:
            best_solution = solution
            best_count = solution.placed_count
        if solution.placed_count == w * h:
            break

    if best_solution is None:
        print("\nNo solution at all.")
        return

    is_full = best_solution.placed_count == w * h
    print(f"\n{'=' * 60}")
    print(f"{'FULL SOLUTION' if is_full else 'PARTIAL SOLUTION'}: "
          f"{best_solution.placed_count}/{w * h}")
    print(f"{'=' * 60}")
    print(best_solution)

    print("\n--- Generating outputs ---")
    board_output.generate_solution_grid(best_solution, SOLUTION_PATH)
    board_output.generate_solution_svg(best_solution, DEDUPED_PATH, SOLUTION_PATH)
    generate_assembly_png(best_solution, DEDUPED_PATH, OUTPUT_DIR,
                          os.path.join(SOLUTION_PATH, 'assembly.png'))

    eval_result = board.evaluate_solution(best_solution)
    print(f"\nSolution evaluation:")
    print(f"  Coverage: {eval_result['coverage']:.1%}")
    print(f"  Matched edges: {eval_result['matched_edges']}/{eval_result['total_possible_edges']}")
    print(f"  Match quality: {eval_result['match_quality']:.1%}")

    print(f"\nAll outputs saved to {SOLUTION_PATH}/")


if __name__ == '__main__':
    main()
