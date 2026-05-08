import os
import sys
import json
import heapq

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
BORDER_DIR = os.path.join(SOLUTION_PATH, 'milestone', 'border')

NCC_PRIORITY_WEIGHT = 1000.0
ORI_MAP = {'^': 0, '>': 1, 'v': 2, '<': 3}
ORI_CHARS = ['^', '>', 'v', '<']


def parse_grid(grid_path):
    with open(grid_path, 'r') as f:
        content = f.read()

    grid = {}
    y = 0
    for line in content.strip().split('\n'):
        line = line.strip()
        if line.startswith('---') or not line:
            continue
        tokens = line.split()
        x = 0
        for token in tokens:
            if token == '-':
                x += 1
                continue
            if len(token) >= 2:
                ori_char = token[-1]
                pid_str = token[:-1]
                if ori_char in ORI_MAP:
                    try:
                        grid[(x, y)] = (int(pid_str), ORI_MAP[ori_char])
                    except ValueError:
                        pass
            x += 1
        y += 1

    width = max(k[0] for k in grid) + 1 if grid else 0
    height = max(k[1] for k in grid) + 1 if grid else 0
    return width, height, grid


def extract_edges(grid, width, height):
    edges = {}

    top = []
    for x in range(width):
        if (x, 0) in grid:
            top.append((x, 0, grid[(x, 0)][0], grid[(x, 0)][1]))
    edges['top'] = {'pieces': top, 'direction': RIGHT, 'flat_dir': TOP}

    right = []
    for y in range(height):
        if (width - 1, y) in grid:
            right.append((width - 1, y, grid[(width - 1, y)][0], grid[(width - 1, y)][1]))
    edges['right'] = {'pieces': right, 'direction': BOTTOM, 'flat_dir': RIGHT}

    bottom = []
    for x in range(width):
        if (x, height - 1) in grid:
            bottom.append((x, height - 1, grid[(x, height - 1)][0], grid[(x, height - 1)][1]))
    edges['bottom'] = {'pieces': bottom, 'direction': RIGHT, 'flat_dir': BOTTOM}

    left = []
    for y in range(height):
        if (0, y) in grid:
            left.append((0, y, grid[(0, y)][0], grid[(0, y)][1]))
    edges['left'] = {'pieces': left, 'direction': BOTTOM, 'flat_dir': LEFT}

    return edges


def get_connection_cost(ps, pid_a, ori_a, pid_b, ori_b, direction):
    a_side = (direction - ori_a) % 4
    b_side = (OPPOSITE[direction] - ori_b) % 4

    for n_pid, n_side, error in ps.get(pid_a, [[], [], [], []])[a_side]:
        if n_pid == pid_b and n_side == b_side:
            return error
    return None


def compute_edge_cost(ps, pieces, direction):
    total = 0.0
    for i in range(len(pieces) - 1):
        _, _, pid_a, ori_a = pieces[i]
        _, _, pid_b, ori_b = pieces[i + 1]
        cost = get_connection_cost(ps, pid_a, ori_a, pid_b, ori_b, direction)
        if cost is None:
            return float('inf')
        total += cost
    return total


def optimize_edge(ps, edge_info):
    pieces = edge_info['pieces']
    direction = edge_info['direction']

    if len(pieces) <= 2:
        return pieces, 0.0

    start = pieces[0]
    end = pieces[-1]
    interior = pieces[1:-1]
    n = len(interior)

    print(f"    Interior pieces ({n}): {[p[2] for p in interior]}")

    INF = float('inf')

    cost_from_start = []
    for i, (_, _, pid, ori) in enumerate(interior):
        c = get_connection_cost(ps, start[2], start[3], pid, ori, direction)
        cost_from_start.append(c if c is not None else INF)

    cost_to_end = []
    for i, (_, _, pid, ori) in enumerate(interior):
        c = get_connection_cost(ps, pid, ori, end[2], end[3], direction)
        cost_to_end.append(c if c is not None else INF)

    cost_between = [[INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = get_connection_cost(ps, interior[i][2], interior[i][3],
                                    interior[j][2], interior[j][3], direction)
            cost_between[i][j] = c if c is not None else INF

    reachable_from_start = sum(1 for c in cost_from_start if c < INF)
    reachable_to_end = sum(1 for c in cost_to_end if c < INF)
    print(f"    Reachable from start: {reachable_from_start}/{n}, to end: {reachable_to_end}/{n}")

    if reachable_from_start == 0 or reachable_to_end == 0:
        print(f"    Cannot optimize: no valid starting or ending piece")
        return None, INF

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
        print(f"    No valid complete path found!")
        return None, INF

    old_positions = [(p[0], p[1]) for p in pieces]
    result = [start]
    for idx in best_path:
        old_pos = old_positions[1 + idx]
        result.append((old_pos[0], old_pos[1], interior[idx][2], interior[idx][3]))
    result.append(end)

    return result, best_cost


def load_ps(connectivity_file, ncc_report_file):
    with open(connectivity_file, 'r') as f:
        connectivity_raw = json.load(f)

    ps_raw = {}
    for pid_str, fits_list in connectivity_raw.items():
        pid = int(pid_str)
        ps_raw[pid] = [[], [], [], []]
        for i in range(4):
            for m in fits_list[i]:
                ps_raw[pid][i].append((m['pid'], m['si'], m['error']))

    ncc_lookup = {}
    if os.path.exists(ncc_report_file):
        with open(ncc_report_file, 'r') as f:
            report = json.load(f)
        for pid_str, sides in report.items():
            pid = int(pid_str)
            for si, matches in enumerate(sides):
                for m in matches:
                    key = (pid, si, m['pid'], m['si'])
                    ncc_lookup[key] = {'ncc': m['ncc'], 'reject': m.get('reject', False)}

    ps_ncc = {}
    for pid, sides in ps_raw.items():
        ps_ncc[pid] = [[], [], [], []]
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
                else:
                    fb_list.append((other_pid, other_si, error))
            ncc_list.sort(key=lambda x: x[2])
            fb_list.sort(key=lambda x: x[2])
            ps_ncc[pid][si] = ncc_list + fb_list

    return ps_raw, ps_ncc


def print_edge_detail(ps, pieces, direction, label):
    print(f"  {label}: {[f'{p[2]}{ORI_CHARS[p[3]]}' for p in pieces]}")
    total = 0.0
    for i in range(len(pieces) - 1):
        _, _, pid_a, ori_a = pieces[i]
        _, _, pid_b, ori_b = pieces[i + 1]
        cost = get_connection_cost(ps, pid_a, ori_a, pid_b, ori_b, direction)
        cost_str = f'{cost:.6f}' if cost is not None else 'NONE'
        print(f"    {pid_a}{ORI_CHARS[ori_a]} -> {pid_b}{ORI_CHARS[ori_b]}: {cost_str}")
        if cost is not None:
            total += cost
    print(f"    Total: {total:.6f}")


def main():
    print("=" * 60)
    print("Border Edge Optimization")
    print("=" * 60)

    grid_path = os.path.join(BORDER_DIR, 'solution_grid.txt')
    connectivity_file = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
    ncc_report_file = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')

    width, height, grid = parse_grid(grid_path)
    print(f"Grid: {width}x{height}, {len(grid)} pieces")

    ps_raw, ps_ncc = load_ps(connectivity_file, ncc_report_file)

    edges = extract_edges(grid, width, height)

    print("\n--- Original edges (NCC cost) ---")
    for name, info in edges.items():
        pieces = info['pieces']
        direction = info['direction']
        cost = compute_edge_cost(ps_ncc, pieces, direction)
        print(f"\n  {name.upper()} edge (dir={direction}): total_cost={cost:.6f}")
        print_edge_detail(ps_ncc, pieces, direction, "Original")

    print("\n" + "=" * 60)
    print("Phase 1: Optimizing with NCC connectivity")
    print("=" * 60)

    for name in ['top', 'right', 'bottom', 'left']:
        info = edges[name]
        direction = info['direction']
        pieces = info['pieces']
        old_cost = compute_edge_cost(ps_ncc, pieces, direction)

        print(f"\n{'=' * 40}")
        print(f"  [{name.upper()}] {len(pieces)} pieces, {len(pieces)-2} interior")
        print(f"{'=' * 40}")

        new_pieces, new_cost = optimize_edge(ps_ncc, info)

        if new_pieces is not None and new_cost < old_cost:
            print(f"\n  NCC IMPROVED: {old_cost:.6f} -> {new_cost:.6f}")
            print_edge_detail(ps_ncc, new_pieces, direction, "NCC Optimized")
            info['pieces'] = new_pieces
        elif new_pieces is not None and new_cost == old_cost:
            print(f"\n  NCC SAME: {old_cost:.6f}")
            info['pieces'] = new_pieces
        else:
            print(f"\n  NCC NO CHANGE (cost={old_cost:.6f})")

    print("\n" + "=" * 60)
    print("Phase 2: Re-optimizing with RAW connectivity (all edges)")
    print("=" * 60)

    for name in ['top', 'right', 'bottom', 'left']:
        info = edges[name]
        direction = info['direction']
        pieces = info['pieces']
        old_cost_raw = compute_edge_cost(ps_raw, pieces, direction)
        old_cost_ncc = compute_edge_cost(ps_ncc, pieces, direction)

        print(f"\n{'=' * 40}")
        print(f"  [{name.upper()}] RAW re-optimize (current raw cost={old_cost_raw:.6f})")
        print(f"{'=' * 40}")

        new_pieces, new_cost_raw = optimize_edge(ps_raw, info)

        if new_pieces is not None and new_cost_raw < old_cost_raw:
            new_cost_ncc = compute_edge_cost(ps_ncc, new_pieces, direction)
            print(f"\n  RAW IMPROVED: {old_cost_raw:.6f} -> {new_cost_raw:.6f}")
            print(f"  NCC cost: {old_cost_ncc:.6f} -> {new_cost_ncc:.6f}")
            print_edge_detail(ps_raw, new_pieces, direction, "RAW Optimized")
            info['pieces'] = new_pieces
        elif new_pieces is not None and new_cost_raw == old_cost_raw:
            print(f"\n  RAW SAME: {old_cost_raw:.6f}")
        else:
            print(f"\n  RAW NO CHANGE (cost={old_cost_raw:.6f})")

    print("\n" + "=" * 60)
    print("Rebuilding board with optimized edges...")
    print("=" * 60)

    board = Board(width=width, height=height)
    for name, info in edges.items():
        for x, y, pid, ori in info['pieces']:
            if pid not in board._placed_piece_ids:
                board.place(pid, ps_ncc[pid], x, y, ori)

    print(board)

    out_dir = os.path.join(SOLUTION_PATH, 'border_optimized')
    os.makedirs(out_dir, exist_ok=True)

    board_output.generate_solution_grid(board, out_dir)
    board_output.generate_solution_svg(board, DEDUPED_PATH, out_dir)
    generate_assembly_png(board, DEDUPED_PATH, OUTPUT_DIR, os.path.join(out_dir, 'assembly.png'))

    print(f"\nOutputs saved to {out_dir}/")

    print("\n--- Final edge summary ---")
    for name, info in edges.items():
        pieces = info['pieces']
        direction = info['direction']
        cost = compute_edge_cost(ps_ncc, pieces, direction)
        pids = [f'{p[2]}{ORI_CHARS[p[3]]}' for p in pieces]
        print(f"  {name:>6}: cost={cost:.6f} | {' '.join(pids)}")


if __name__ == '__main__':
    main()
