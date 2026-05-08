import os
import sys
import json

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import CONNECTIVITY_DIR
from common.board import Board, OPPOSITE, TOP, RIGHT, BOTTOM, LEFT

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)

ORI_MAP = {'^': 0, '>': 1, 'v': 2, '<': 3}


def parse_grid(path):
    grid = {}
    y = 0
    with open(path) as f:
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
    return grid


def load_ps():
    conn_file = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
    with open(conn_file) as f:
        raw = json.load(f)
    ps_raw = {}
    for pid_str, fits in raw.items():
        pid = int(pid_str)
        ps_raw[pid] = [[], [], [], []]
        for i in range(4):
            for m in fits[i]:
                ps_raw[pid][i].append((m['pid'], m['si'], m['error']))

    ncc_file = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')
    ncc_lookup = {}
    if os.path.exists(ncc_file):
        with open(ncc_file) as f:
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
            ncc_list, fb_list = [], []
            for opid, osi, err in sides[si]:
                key = (pid, si, opid, osi)
                rev = (opid, osi, pid, si)
                info = ncc_lookup.get(key) or ncc_lookup.get(rev)
                if info and not info['reject'] and info['ncc'] > 0:
                    ncc_list.append((opid, osi, err / (info['ncc'] * 1000.0)))
                else:
                    fb_list.append((opid, osi, err))
            ncc_list.sort(key=lambda x: x[2])
            fb_list.sort(key=lambda x: x[2])
            ps_ncc[pid][si] = ncc_list + fb_list

    return ps_raw, ps_ncc


def get_neighbor_candidates(ps, nb_pid, nb_ori, facing_us):
    nb_side = (facing_us - nb_ori) % 4
    return {n_pid for n_pid, _, _ in ps[nb_pid][nb_side]}


def main():
    grid = parse_grid(r'f:\work_Puzzle_github\puzzle-bot\output\puzzle_new\6_solution\milestone\pct75\solution_grid.txt')
    ps_raw, ps_ncc = load_ps()

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
