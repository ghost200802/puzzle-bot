import os
import sys
import json
import heapq

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from config import get_output_dir
from common.board import Board, OPPOSITE, TOP, RIGHT, BOTTOM, LEFT
from common import output as board_output

ORI_MAP = {'^': 0, '>': 1, 'v': 2, '<': 3}
ORI_CHARS = ['^', '>', 'v', '<']
NCC_PRIORITY_WEIGHT = 1000.0


def parse_grid(grid_file):
    grid = {}
    y = 0
    with open(grid_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('---') or not line:
                continue
            x = 0
            for token in line.split():
                if token == '-':
                    x += 1
                    continue
                if len(token) >= 2 and token[-1] in ORI_MAP:
                    try:
                        grid[(x, y)] = (int(token[:-1]), ORI_MAP[token[-1]])
                    except ValueError:
                        pass
                x += 1
            y += 1
    w = max(k[0] for k in grid) + 1 if grid else 0
    h = max(k[1] for k in grid) + 1 if grid else 0
    return w, h, grid


def load_connectivity_and_ncc(connectivity_path):
    connectivity_file = os.path.join(connectivity_path, 'connectivity.json')
    ncc_report_file = os.path.join(connectivity_path, 'texture_verify_report.json')

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
                    ncc_lookup[key] = {
                        'ncc': m['ncc'],
                        'reject': m.get('reject', False),
                    }

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


def get_cost(ps, pid_from, side_from, pid_to, side_to):
    for n_pid, n_side, error in ps.get(pid_from, [[], [], [], []])[side_from]:
        if n_pid == pid_to and n_side == side_to:
            return error
    return None


def get_oriented_cost(ps, pid_a, ori_a, pid_b, ori_b, direction):
    a_side = (direction - ori_a) % 4
    b_side = (OPPOSITE[direction] - ori_b) % 4
    return get_cost(ps, pid_a, a_side, pid_b, b_side)


def solve_tsp(nodes, cost_fn, fixed_first=None, fixed_last=None, constraint_fn=None):
    n = len(nodes)
    if n == 0:
        if fixed_first is not None and fixed_last is not None:
            c = cost_fn(fixed_first, fixed_last)
            return [], c if c is not None else float('inf')
        return [], 0.0

    INF = float('inf')

    cost_from_first = []
    for i in range(n):
        if fixed_first is not None:
            c = cost_fn(fixed_first, nodes[i])
        else:
            c = 0.0
        cost_from_first.append(c if c is not None else INF)

    cost_to_last = []
    for i in range(n):
        if fixed_last is not None:
            c = cost_fn(nodes[i], fixed_last)
        else:
            c = 0.0
        cost_to_last.append(c if c is not None else INF)

    cost_between = [[INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = cost_fn(nodes[i], nodes[j])
            cost_between[i][j] = c if c is not None else INF

    full_mask = (1 << n) - 1
    dist = {}
    pq = []

    for i in range(n):
        if cost_from_first[i] >= INF:
            continue
        mask = full_mask ^ (1 << i)
        c = cost_from_first[i]
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
            total = cost + cost_to_last[last]
            if total < best_cost:
                if constraint_fn is None or constraint_fn(path, nodes):
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
        return None, INF

    result = [nodes[i] for i in best_path]
    return result, best_cost


def save_solution(board, deduped_path, output_dir, output_name='puzzle_new'):
    from solve_display import generate_assembly_png

    os.makedirs(output_dir, exist_ok=True)
    board_output.generate_solution_grid(board, output_dir)
    board_output.generate_solution_svg(board, deduped_path, output_dir)

    output_root = get_output_dir(output_name)
    generate_assembly_png(
        board, deduped_path, output_root,
        os.path.join(output_dir, 'assembly.png'),
    )
