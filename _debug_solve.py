import sys, os, json, math
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipline'))

from common.config import DEDUPED_DIR
from common.board import Board
from solve_display import compute_piece_transforms
from run_matchtarget import load_solution, _resize_to_max

solution_dir = 'output/puzzle_new/6_solution/milestone/pct75'
output_root = 'output/puzzle_new'
deduped_dir = os.path.join(output_root, DEDUPED_DIR)
color_dir = os.path.join(output_root, '2_piece_colors')

pw, ph, placed = load_solution(solution_dir)

report_path = os.path.join(os.path.dirname(solution_dir), 'target_match_report.json')
with open(report_path, 'r') as f:
    report = json.load(f)

ps_raw = {}
for pid in placed:
    sides = []
    for si in range(4):
        json_path = os.path.join(deduped_dir, f'side_{pid}_{si}.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                sides.append(json.load(f))
        else:
            sides.append(None)
    ps_raw[pid] = sides

threshold = 0.8
board = Board(pw, ph)
fixed_pids = set()
removed_pids = {}
for pid, info in placed.items():
    score = report.get(str(pid), {}).get('score', 0)
    if score >= threshold:
        fits = ps_raw.get(pid, [[], [], [], []])
        board.place(pid, fits, info['gx'], info['gy'], info['orientation'])
        fixed_pids.add(pid)
    else:
        removed_pids[pid] = {'score': score, 'info': info}

print(f"Board: {pw}x{ph}, fixed={len(fixed_pids)}, removed={len(removed_pids)}")
print(f"\nRemoved pieces (<{threshold}):")
for pid, d in sorted(removed_pids.items()):
    info = d['info']
    print(f"  pid={pid}: grid({info['gx']},{info['gy']}) ori={info['orientation']} score={d['score']:.4f}")

# Print board state
print(f"\nBoard state:")
for gy in range(ph):
    row = ""
    for gx in range(pw):
        cell = board.get(gx, gy)
        if cell is None:
            row += "  .  "
        else:
            pid = cell[0]
            ori = cell[2]
            row += f"{pid:>3d}/{ori} "
    print(f"  y={gy}: {row}")

# Draw on target image
target_path = os.path.join(os.path.dirname(solution_dir), 'target_aligned.png')
target = cv2.imread(target_path)
th, tw = target.shape[:2]

raw_ass = cv2.imread(os.path.join(os.path.dirname(solution_dir), '_match_assembly.png'))
_, resize_scale = _resize_to_max(raw_ass, 2000)

grid_ox = 90.3
grid_oy = 104.4
cell_w = 124.7
cell_h = 180.5

canvas = target.copy()

# Draw grid
for gy in range(ph):
    for gx in range(pw):
        x1 = int(grid_ox + gx * cell_w)
        y1 = int(grid_oy + gy * cell_h)
        x2 = int(grid_ox + (gx + 1) * cell_w)
        y2 = int(grid_oy + (gy + 1) * cell_h)

        cell = board.get(gx, gy)
        if cell is None:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(canvas, f"?", (x1 + 2, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        else:
            pid = cell[0]
            ori = cell[2]
            score = report.get(str(pid), {}).get('score', 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(canvas, f"{pid}/{ori}", (x1 + 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            cv2.putText(canvas, f"{score:.2f}", (x1 + 2, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

out_dir = os.path.join(os.path.dirname(solution_dir), 'targeted_solve')
os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, 'debug_board_grid.png'), canvas)
print(f"\nSaved debug_board_grid.png")

# Now show candidates for each empty position
connectivity_file = os.path.join(output_root, '5_connectivity', 'connectivity.json')
with open(connectivity_file, 'r') as f:
    connectivity_raw = json.load(f)
connectivity = {}
for pid_str, sides in connectivity_raw.items():
    pid = int(pid_str)
    connectivity[pid] = [[], [], [], []]
    for si in range(4):
        for m in sides[si]:
            connectivity[pid][si].append((m['pid'], m['si'], m['error']))

used_pids = fixed_pids.copy()
print(f"\n--- Empty positions and candidates ---")
for gy in range(ph):
    for gx in range(pw):
        if board.get(gx, gy) is not None:
            continue
        has_neighbor = False
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            if board.get(gx + dx, gy + dy) is not None:
                has_neighbor = True
                break
        if not has_neighbor:
            continue

        constraints = []
        for d, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            cell = board.get(gx + dx, gy + dy)
            if cell is None:
                continue
            adj_pid, _, adj_ori = cell
            s_adj = (d - adj_ori) % 4
            back_d = (d + 2) % 4
            if adj_pid not in connectivity:
                continue
            constraint = []
            for match_pid, match_si, error in connectivity[adj_pid][s_adj]:
                required_ori = (back_d - match_si) % 4
                constraint.append((match_pid, required_ori, error))
            constraints.append(constraint)

        candidates = {}
        if len(constraints) == 1:
            for pid, ori, err in constraints[0]:
                candidates[pid] = ori
        elif len(constraints) > 1:
            pid_oris = {}
            for constraint in constraints:
                for pid, ori, _ in constraint:
                    pid_oris.setdefault(pid, set()).add(ori)
            for pid, oris in pid_oris.items():
                if len(oris) == 1:
                    candidates[pid] = oris.pop()

        available = [(pid, ori) for pid, ori in candidates.items()
                     if pid not in used_pids and pid in ps_raw]

        print(f"\n  ({gx},{gy}): {len(constraints)} constraints, {len(available)} available candidates")
        for pid, ori in available[:5]:
            info = placed.get(pid, {})
            old_gx = info.get('gx', '?')
            old_gy = info.get('gy', '?')
            old_ori = info.get('orientation', '?')
            score = report.get(str(pid), {}).get('score', 0)
            print(f"    pid={pid} ori={ori} (was at grid({old_gx},{old_gy}) ori={old_ori} score={score:.4f})")

print(f"\nDone!")
