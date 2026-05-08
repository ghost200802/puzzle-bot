import os
import sys
import json
import time
import shutil
import random

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
ORI_CHARS = ['^', '>', 'v', '<']


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
    w = max(k[0] for k in grid) + 1 if grid else 0
    h = max(k[1] for k in grid) + 1 if grid else 0
    return w, h, grid


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
                    ncc_lookup[key] = {
                        'ncc': m['ncc'],
                        'reject': m.get('reject', False)
                    }

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


def get_ncc_cost(ps_ncc, pid_a, ori_a, pid_b, ori_b, direction):
    a_side = (direction - ori_a) % 4
    b_side = (OPPOSITE[direction] - ori_b) % 4
    for n_pid, n_side, cost in ps_ncc.get(pid_a, [[], [], [], []])[a_side]:
        if n_pid == pid_b and n_side == b_side:
            return cost
    return None


def get_raw_cost(ps_raw, pid_a, ori_a, pid_b, ori_b, direction):
    a_side = (direction - ori_a) % 4
    b_side = (OPPOSITE[direction] - ori_b) % 4
    for n_pid, n_side, cost in ps_raw.get(pid_a, [[], [], [], []])[a_side]:
        if n_pid == pid_b and n_side == b_side:
            return cost
    return None


class IterativeSolver:
    def __init__(self, w, h, grid, ps_ncc, ps_raw):
        self.w = w
        self.h = h
        self.ps_ncc = ps_ncc
        self.ps_raw = ps_raw
        self.total = w * h

        self.board = Board(width=w, height=h)
        self.pos_map = {}
        self.pid_pos = {}

        for (x, y), (pid, ori) in grid.items():
            self.board.place(pid, ps_ncc[pid], x, y, ori)
            self.pos_map[(x, y)] = (pid, ori)
            self.pid_pos[pid] = (x, y)

        all_pids = set(range(100))
        placed = set(self.pid_pos.keys())
        self.available = all_pids - placed

        self.costs = {}
        self._update_all_costs()

        self.recently_freed = set()
        self.changed_positions = set()
        for y in range(h):
            for x in range(w):
                self.changed_positions.add((x, y))

        self.rounds = 0
        self.best_count = len(placed)
        self.best_board = Board.copy(self.board)
        self.best_pos_map = dict(self.pos_map)
        self.best_pid_pos = dict(self.pid_pos)
        self.best_available = set(self.available)

        self.t_start = time.time()
        self.last_milestone_pct = 0
        self.milestone_dir = os.path.join(SOLUTION_PATH, 'milestone_iter')
        if os.path.exists(self.milestone_dir):
            shutil.rmtree(self.milestone_dir)
        os.makedirs(self.milestone_dir, exist_ok=True)

    def _compute_piece_cost(self, pid):
        pos = self.pid_pos.get(pid)
        if pos is None:
            return 0.0
        x, y = pos
        cell = self.board.get(x, y)
        if cell is None:
            return 0.0
        _, _, ori = cell

        edge_costs = []
        for dx, dy, direction in [(-1, 0, RIGHT), (1, 0, LEFT), (0, -1, BOTTOM), (0, 1, TOP)]:
            nx, ny = x + dx, y + dy
            nb = self.board.get(nx, ny)
            if nb is None:
                continue
            nb_pid, _, nb_ori = nb
            ncc = get_ncc_cost(self.ps_ncc, pid, ori, nb_pid, nb_ori, direction)
            if ncc is not None:
                edge_costs.append(ncc)
            else:
                raw = get_raw_cost(self.ps_raw, pid, ori, nb_pid, nb_ori, direction)
                if raw is not None:
                    edge_costs.append(raw / 100.0)
                else:
                    edge_costs.append(100.0)

        if not edge_costs:
            return 50.0
        avg = sum(edge_costs) / len(edge_costs)
        worst = max(edge_costs)
        return worst + avg * 0.5

    def _update_all_costs(self):
        for pid in self.pid_pos:
            self.costs[pid] = self._compute_piece_cost(pid)

    def _update_cost_around(self, x, y):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = self.board.get(x + dx, y + dy)
            if nb is not None:
                self.costs[nb[0]] = self._compute_piece_cost(nb[0])

    def _get_empty_with_neighbors(self):
        positions = []
        for y in range(self.h):
            for x in range(self.w):
                if self.board.get(x, y) is not None:
                    continue
                nb_count = 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if self.board.get(x + dx, y + dy) is not None:
                        nb_count += 1
                if nb_count >= 1:
                    positions.append((x, y))
        return positions

    def _find_candidates(self, x, y, prefer_pids=None):
        neighbors = []
        for dx, dy, facing_us in [(-1, 0, RIGHT), (1, 0, LEFT), (0, -1, BOTTOM), (0, 1, TOP)]:
            nx, ny = x + dx, y + dy
            nb = self.board.get(nx, ny)
            if nb is not None:
                neighbors.append((nb[0], nb[2], facing_us))

        if not neighbors:
            return []

        candidate_sets = []
        for nb_pid, nb_ori, facing_us in neighbors:
            nb_side = (facing_us - nb_ori) % 4
            ncc_pids = {n_pid for n_pid, _, _ in self.ps_ncc[nb_pid][nb_side]}
            if not ncc_pids:
                ncc_pids = {n_pid for n_pid, _, _ in self.ps_raw[nb_pid][nb_side]}
            candidate_sets.append(ncc_pids)

        common = set.intersection(*candidate_sets) & self.available
        if not common:
            all_candidate_sets = []
            for nb_pid, nb_ori, facing_us in neighbors:
                nb_side = (facing_us - nb_ori) % 4
                raw_pids = {n_pid for n_pid, _, _ in self.ps_raw[nb_pid][nb_side]}
                all_candidate_sets.append(raw_pids)
            common = set.intersection(*all_candidate_sets) & self.available

        results = []
        for pid in common:
            for nb_pid, nb_ori, facing_us in neighbors:
                nb_side = (facing_us - nb_ori) % 4
                for n_pid, n_side, _ in self.ps_ncc[nb_pid][nb_side]:
                    if n_pid == pid:
                        orientation = (OPPOSITE[facing_us] - n_side) % 4
                        ok, _ = self.board.can_place(pid, self.ps_ncc[pid], x, y, orientation)
                        if ok:
                            cost = self._compute_placement_cost(pid, orientation, x, y)
                            if prefer_pids and pid in prefer_pids:
                                cost -= 5.0
                            results.append((cost, pid, orientation))
                        break
                else:
                    for n_pid, n_side, _ in self.ps_raw[nb_pid][nb_side]:
                        if n_pid == pid:
                            orientation = (OPPOSITE[facing_us] - n_side) % 4
                            ok, _ = self.board.can_place(pid, self.ps_ncc[pid], x, y, orientation)
                            if ok:
                                cost = self._compute_placement_cost(pid, orientation, x, y) + 10.0
                                if prefer_pids and pid in prefer_pids:
                                    cost -= 5.0
                                results.append((cost, pid, orientation))
                            break
                break

        results.sort()
        return results

    def _compute_placement_cost(self, pid, ori, x, y):
        edge_costs = []
        for dx, dy, direction in [(-1, 0, RIGHT), (1, 0, LEFT), (0, -1, BOTTOM), (0, 1, TOP)]:
            nx, ny = x + dx, y + dy
            nb = self.board.get(nx, ny)
            if nb is None:
                continue
            nb_pid, _, nb_ori = nb
            ncc = get_ncc_cost(self.ps_ncc, pid, ori, nb_pid, nb_ori, direction)
            if ncc is not None:
                edge_costs.append(ncc)
            else:
                raw = get_raw_cost(self.ps_raw, pid, ori, nb_pid, nb_ori, direction)
                if raw is not None:
                    edge_costs.append(raw / 100.0)
                else:
                    edge_costs.append(100.0)
        if not edge_costs:
            return 50.0
        return max(edge_costs) + sum(edge_costs) / len(edge_costs) * 0.5

    def _place_piece(self, pid, ori, x, y):
        self.board.place(pid, self.ps_ncc[pid], x, y, ori)
        self.pos_map[(x, y)] = (pid, ori)
        self.pid_pos[pid] = (x, y)
        self.available.discard(pid)
        self.recently_freed.discard(pid)
        self.costs[pid] = self._compute_piece_cost(pid)
        self._update_cost_around(x, y)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self.changed_positions.add((x + dx, y + dy))

    def _remove_piece(self, pid):
        pos = self.pid_pos.get(pid)
        if pos is None:
            return
        x, y = pos
        self.board._board[y][x] = None
        self.board._placed_piece_ids.discard(pid)
        del self.pos_map[(x, y)]
        del self.pid_pos[pid]
        self.available.add(pid)
        self.recently_freed.add(pid)
        if pid in self.costs:
            del self.costs[pid]
        self._update_cost_around(x, y)
        self.changed_positions.add((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self.changed_positions.add((x + dx, y + dy))

    def _random_remove_interior(self):
        interior = []
        for pid in list(self.pid_pos.keys()):
            pos = self.pid_pos[pid]
            x, y = pos
            if x > 0 and x < self.w - 1 and y > 0 and y < self.h - 1:
                interior.append(pid)
        if not interior:
            return None
        pid = random.choice(interior)
        x, y = self.pid_pos[pid]
        cost = self.costs.get(pid, 0)
        self._remove_piece(pid)
        return pid, x, y, cost

    def _save_milestone(self, tag):
        ms_dir = os.path.join(self.milestone_dir, tag)
        os.makedirs(ms_dir, exist_ok=True)
        try:
            board_output.generate_solution_grid(self.board, ms_dir)
        except Exception:
            pass
        try:
            generate_assembly_png(self.board, DEDUPED_PATH, OUTPUT_DIR,
                                  os.path.join(ms_dir, 'assembly.png'))
        except Exception:
            pass

    def _save_best(self):
        out_dir = os.path.join(SOLUTION_PATH, 'iter_best')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        board_output.generate_solution_grid(self.best_board, out_dir)
        board_output.generate_solution_svg(self.best_board, DEDUPED_PATH, out_dir)
        generate_assembly_png(self.best_board, DEDUPED_PATH, OUTPUT_DIR,
                              os.path.join(out_dir, 'assembly.png'))

    def run(self):
        print(f"\nStarting: {self.board.placed_count}/{self.total} placed, "
              f"{len(self.available)} available")

        last_output = time.time()
        no_progress_rounds = 0

        while True:
            self.rounds += 1
            now = time.time()

            if self.rounds % 1000 == 0 or now - last_output > 10:
                elapsed = now - self.t_start
                cost_vals = list(self.costs.values()) if self.costs else [0]
                print(f"  round {self.rounds:>8d} | placed {self.board.placed_count:>3d}/{self.total} "
                      f"| best {self.best_count:>3d} | avail {len(self.available):>3d} "
                      f"| cost avg={sum(cost_vals)/len(cost_vals):.2f} | {elapsed:.1f}s")
                last_output = now

                pct = int(self.best_count * 100 / self.total)
                threshold = (pct // 5) * 5
                if threshold > self.last_milestone_pct and threshold > 0:
                    self.last_milestone_pct = threshold
                    print(f"  *** Milestone {threshold}%: {self.best_count}/{self.total} ***")
                    self._save_milestone(f"pct{threshold}")

            if self.board.placed_count == self.total:
                print(f"\nFULL SOLUTION at round {self.rounds}!")
                print(self.board)
                self._save_best()
                return True

            empty_positions = self._get_empty_with_neighbors()
            random.shuffle(empty_positions)

            filled_this_round = 0
            for x, y in empty_positions:
                if self.board.get(x, y) is not None:
                    continue
                if (x, y) in self.changed_positions:
                    cands = self._find_candidates(x, y, prefer_pids=self.recently_freed)
                else:
                    if not self.recently_freed:
                        continue
                    cands = self._find_candidates(x, y, prefer_pids=self.recently_freed)
                    cands = [(c, p, o) for c, p, o in cands if p in self.recently_freed]
                if cands:
                    _, best_pid, best_ori = cands[0]
                    self._place_piece(best_pid, best_ori, x, y)
                    filled_this_round += 1

                    if self.board.placed_count > self.best_count:
                        self.best_count = self.board.placed_count
                        self.best_board = Board.copy(self.board)
                        self.best_pos_map = dict(self.pos_map)
                        self.best_pid_pos = dict(self.pid_pos)
                        self.best_available = set(self.available)
                        print(f"  *** NEW BEST: {self.best_count}/{self.total} at round {self.rounds} ***")

                    if self.board.placed_count == self.total:
                        break

            if self.board.placed_count == self.total:
                continue

            if filled_this_round == 0:
                no_progress_rounds += 1
                self.changed_positions.clear()
                result = self._random_remove_interior()
                if result is None:
                    print("  No interior pieces to remove, stopping.")
                    break
                removed_pid, rx, ry, rcost = result
                print(f"    [{no_progress_rounds}] REMOVED {removed_pid} at ({rx},{ry}) cost={rcost:.1f} "
                      f"-> {self.board.placed_count} placed, {len(self.available)} avail")
            else:
                no_progress_rounds = 0

            if no_progress_rounds > 50:
                print(f"  No progress for {no_progress_rounds} rounds, resetting to best ({self.best_count}).")
                self.board = Board.copy(self.best_board)
                self.pos_map = dict(self.best_pos_map)
                self.pid_pos = dict(self.best_pid_pos)
                self.available = set(self.best_available)
                self.costs.clear()
                self._update_all_costs()
                no_progress_rounds = 0

            if self.rounds > 1000000:
                print("  Reached max rounds.")
                break

        print(f"\nFinal: {self.best_count}/{self.total} pieces")
        print(self.best_board)
        self._save_best()
        return False


def main():
    print("=" * 60)
    print("Iterative Solver: Random-Order Fill + Random Remove")
    print("=" * 60)

    ps_raw, ps_ncc = load_ps()

    pct75_path = os.path.join(SOLUTION_PATH, 'milestone', 'pct75', 'solution_grid.txt')
    if not os.path.exists(pct75_path):
        print(f"ERROR: {pct75_path} not found")
        return

    w, h, grid = parse_grid(pct75_path)
    print(f"Grid: {w}x{h}, {len(grid)} placed pieces")

    solver = IterativeSolver(w, h, grid, ps_ncc, ps_raw)
    solver.run()


if __name__ == '__main__':
    main()
