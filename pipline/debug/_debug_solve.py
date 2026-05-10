import sys, os, json, math, time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from common.config import DEDUPED_DIR
from common.board import Board
from pipline.run_matchtarget import load_solution, TargetMatcher, _resize_to_max
from pipline.run_targetedsolve import _prepare_piece_at_cell, _ncc_search, _load_piece_images

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
for pid, info in placed.items():
    score = report.get(str(pid), {}).get('score', 0)
    if score >= threshold:
        fits = ps_raw.get(pid, [[], [], [], []])
        board.place(pid, fits, info['gx'], info['gy'], info['orientation'])
        fixed_pids.add(pid)

with open(os.path.join(output_root, '5_connectivity', 'connectivity.json'), 'r') as f:
    connectivity_raw = json.load(f)
connectivity = {}
for pid_str, sides in connectivity_raw.items():
    pid = int(pid_str)
    connectivity[pid] = [[], [], [], []]
    for si in range(4):
        for m in sides[si]:
            connectivity[pid][si].append((m['pid'], m['si'], m['error']))

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = ['TOP', 'RIGHT', 'BOTTOM', 'LEFT']

gx, gy = 5, 9
print(f"=== Candidates for ({gx},{gy}) ===\n")

constraint_sets = []
for d, (dx, dy) in enumerate(DIRS):
    nx, ny = gx + dx, gy + dy
    cell = board.get(nx, ny)
    if cell is None:
        print(f"  {DIR_NAMES[d]} ({nx},{ny}): EMPTY")
        continue
    adj_pid, _, adj_ori = cell
    back_d = (d + 2) % 4
    s_adj = (back_d - adj_ori) % 4
    print(f"  {DIR_NAMES[d]} ({nx},{ny}): pid={adj_pid} ori={adj_ori} "
          f"-> neighbor's physical_side={s_adj}, candidate needs board_side={back_d}")

    if adj_pid not in connectivity:
        print(f"    No connectivity data")
        continue
    matches = connectivity[adj_pid][s_adj]
    s = set()
    for mp, ms, err in matches:
        req_ori = (d - ms) % 4
        s.add((mp, req_ori))
        tag = "FIXED" if mp in fixed_pids else "ok"
        print(f"    -> pid={ms}of{mp} ori={req_ori} err={err:.0f} [{tag}]")
    print(f"    {len(s)} unique (pid,ori) pairs")
    constraint_sets.append(s)

if not constraint_sets:
    candidates = []
else:
    result = constraint_sets[0]
    for s in constraint_sets[1:]:
        result = result & s
    candidates = sorted([(pid, ori) for pid, ori in result if pid not in fixed_pids],
                        key=lambda x: x[0])

print(f"\nCandidates (intersection): {len(candidates)}")
for pid, ori in candidates:
    old = placed.get(pid, {})
    score = report.get(str(pid), {}).get('score', 0)
    print(f"  #{pid} ori={ori} (was grid({old.get('gx','?')},{old.get('gy','?')}) "
          f"ori={old.get('orientation','?')}) score={score:.3f}")

ORI_TO_ANGLE = {0: 0, 1: 90, 2: 180, 3: 270}
cell_sz = 200
cols = max(len(candidates), 1)
margin = 10
label_h = 30
canvas_w = cols * (cell_sz + margin) + margin
canvas_h = cell_sz + label_h + 2 * margin
canvas = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)

for i, (pid, ori) in enumerate(candidates):
    x0 = margin + i * (cell_sz + margin)
    y0 = margin
    cv2.putText(canvas, f"#{pid} ori={ori}", (x0 + 5, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

    color_path = os.path.join(color_dir, f'piece_{pid}.png')
    if not os.path.exists(color_path):
        continue
    img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
    bgr = img[:, :, :3].copy()
    alpha = img[:, :, 3]
    bgr[alpha < 128] = 240

    angle = ORI_TO_ANGLE.get(ori, 0)
    if angle != 0:
        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos_a = abs(math.cos(math.radians(angle)))
        sin_a = abs(math.sin(math.radians(angle)))
        nw = int(h * sin_a + w * cos_a)
        nh = int(h * cos_a + w * sin_a)
        M[0, 2] += (nw - w) / 2
        M[1, 2] += (nh - h) / 2
        bgr = cv2.warpAffine(bgr, M, (nw, nh), borderValue=(240, 240, 240))

    h, w = bgr.shape[:2]
    scale = min((cell_sz - 4) / w, (cell_sz - 4) / h)
    rw, rh = int(w * scale), int(h * scale)
    bgr = cv2.resize(bgr, (rw, rh))
    px = x0 + (cell_sz - rw) // 2
    py = y0 + label_h + (cell_sz - rh) // 2
    canvas[py:py + rh, px:px + rw] = bgr

out_dir = os.path.join(os.path.dirname(solution_dir), 'targeted_solve')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'debug_5_9_candidates.png')
cv2.imwrite(out_path, canvas)
print(f"\nSaved: {out_path}")

print(f"\n=== NCC Verification for candidates at ({gx},{gy}) ===")

target_path = os.path.join(os.path.dirname(solution_dir), 'target_aligned.png')
target_aligned = cv2.imread(target_path)
if target_aligned is None:
    print("ERROR: target_aligned.png not found, trying TargetMatcher...")
    target_img_path = os.path.join(output_root, 'target.jpg')
    if not os.path.exists(target_img_path):
        for ext in ['jpg', 'png', 'jpeg', 'JPG', 'PNG']:
            p = os.path.join(output_root, f'target.{ext}')
            if os.path.exists(p):
                target_img_path = p
                break
    matcher = TargetMatcher(target_img_path, pw, ph, placed,
                            os.path.dirname(solution_dir), output_root)
    matcher.rectify_target()
    target_aligned = matcher.target_aligned
    grid_ox = matcher._grid_offset_x
    grid_oy = matcher._grid_offset_y
    cell_w = matcher._cell_w
    cell_h = matcher._cell_h
else:
    raw_ass_path = os.path.join(os.path.dirname(solution_dir), '_match_assembly.png')
    raw_ass = cv2.imread(raw_ass_path)
    _, rs = _resize_to_max(raw_ass, 2000) if raw_ass is not None else (None, 1.0)

    from pipline.solve_display import compute_piece_transforms
    full_board = Board(pw, ph)
    for pid, info in placed.items():
        fits = ps_raw.get(pid, [[], [], [], []])
        full_board.place(pid, fits, info['gx'], info['gy'], info['orientation'])
    _, _, ci = compute_piece_transforms(full_board, deduped_dir)

    min_x, min_y = ci['min_x'], ci['min_y']
    max_x, max_y = ci['max_x'], ci['max_y']
    data_w = max_x - min_x
    data_h = max_y - min_y
    mg = max(data_w, data_h) * 0.05
    header_h = 60
    gen_scale = 1.0
    grid_ox = (mg * gen_scale) * rs
    grid_oy = (mg * gen_scale + header_h) * rs
    cell_w = (data_w * gen_scale) * rs / pw
    cell_h = (data_h * gen_scale) * rs / ph

th, tw = target_aligned.shape[:2]
print(f"target_aligned: {tw}x{th}")
print(f"grid: ox={grid_ox:.1f} oy={grid_oy:.1f} cell_w={cell_w:.1f} cell_h={cell_h:.1f}")

all_pids = set(connectivity.keys())
piece_images, piece_alphas = _load_piece_images(color_dir, all_pids)

from common import util
from pipline.solve_display import compute_piece_transforms

full_board = Board(pw, ph)
for p_id, info in placed.items():
    fits = ps_raw.get(p_id, [[], [], [], []])
    full_board.place(p_id, fits, info['gx'], info['gy'], info['orientation'])
orig_transforms, _, ci = compute_piece_transforms(full_board, deduped_dir)

min_x = ci['min_x']
min_y = ci['min_y']
margin = max(ci['max_x'] - min_x, ci['max_y'] - min_y) * 0.05
header_h = 60
gen_scale = 1.0

ERODE_PX = 5

def prepare_from_transform(pid, rotation, translation, ic):
    piece_bgr = piece_images[pid]
    alpha_raw = piece_alphas[pid]
    h_img, w_img = piece_bgr.shape[:2]
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
    img_pts = []
    for cx, cy in corners:
        ddx = cx - ic[0]
        ddy = cy - ic[1]
        ox = ddx * cos_r - ddy * sin_r + ic[0] + translation[0]
        oy = ddx * sin_r + ddy * cos_r + ic[1] + translation[1]
        img_pts.append((ox, oy))
    img_min_x = min(p[0] for p in img_pts)
    img_min_y = min(p[1] for p in img_pts)
    img_max_x = max(p[0] for p in img_pts)
    img_max_y = max(p[1] for p in img_pts)

    out_w_gen = int(math.ceil((img_max_x - img_min_x) * gen_scale)) + 2
    out_h_gen = int(math.ceil((img_max_y - img_min_y) * gen_scale)) + 2

    cos_neg = math.cos(-rotation)
    sin_neg = math.sin(-rotation)
    sm_x = (img_min_x - ic[0] - translation[0]) * gen_scale
    sm_y = (img_min_y - ic[1] - translation[1]) * gen_scale

    a_v = cos_neg * gen_scale
    b_v = -sin_neg * gen_scale
    c_v = cos_neg * sm_x - sin_neg * sm_y + ic[0]
    d_v = sin_neg * gen_scale
    e_v = cos_neg * gen_scale
    f_v = sin_neg * sm_x + cos_neg * sm_y + ic[1]

    M_pil = np.array([[a_v, b_v, c_v], [d_v, e_v, f_v], [0, 0, 1]], dtype=np.float64)
    M_aff = np.linalg.inv(M_pil)[:2, :]

    piece_gen = cv2.warpAffine(piece_bgr, M_aff, (out_w_gen, out_h_gen),
                                flags=cv2.INTER_AREA,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    alpha_gen = cv2.warpAffine(alpha_raw, M_aff, (out_w_gen, out_h_gen),
                                flags=cv2.INTER_AREA,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    out_w = max(1, int(out_w_gen * rs))
    out_h = max(1, int(out_h_gen * rs))
    piece_final = cv2.resize(piece_gen, (out_w, out_h), interpolation=cv2.INTER_AREA)
    alpha_final = cv2.resize(alpha_gen, (out_w, out_h), interpolation=cv2.INTER_AREA)

    mask_raw = alpha_final > 128
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX * 2 + 1, ERODE_PX * 2 + 1))
    mask_eroded = cv2.erode(mask_raw.astype(np.uint8), erode_kernel) > 0
    piece_gray = cv2.cvtColor(piece_final, cv2.COLOR_BGR2GRAY)

    paste_x = (img_min_x - min_x + margin) * gen_scale * rs
    paste_y = ((img_min_y - min_y + margin) * gen_scale + header_h) * rs

    return piece_gray, mask_eroded, mask_raw, piece_final, (paste_x, paste_y), (out_w, out_h)

for pid, ori in candidates:
    print(f"\n--- #{pid} ori={ori} ---")
    if pid not in piece_images:
        print(f"  No color image")
        continue

    cand_sides = ps_raw.get(pid)
    if not cand_sides or cand_sides[0] is None:
        print(f"  No side data")
        continue
    cand_ic = tuple(cand_sides[0]['incenter'])
    new_sides = util.rotate_list([0, 1, 2, 3], -ori)

    rotations = []
    translation_samples = []
    for d, (ddx, ddy) in enumerate(DIRS):
        nx, ny = gx + ddx, gy + ddy
        cell_n = board.get(nx, ny)
        if cell_n is None:
            continue
        adj_pid, _, adj_ori = cell_n
        back_d = (d + 2) % 4
        s_adj = (back_d - adj_ori) % 4

        if adj_pid not in orig_transforms:
            continue
        rot_n, trans_n, ic_n = orig_transforms[adj_pid]

        adj_side = ps_raw.get(adj_pid, [None]*4)[s_adj]
        if adj_side is None:
            continue
        adj_verts = adj_side['vertices']
        adj_rotated = [util.rotate(v, ic_n, rot_n) for v in adj_verts]
        adj_translated = [(r[0] + trans_n[0], r[1] + trans_n[1]) for r in adj_rotated]

        adj_angle = math.atan2(
            adj_translated[-1][1] - adj_translated[0][1],
            adj_translated[-1][0] - adj_translated[0][0]
        ) % (2 * math.pi)

        cand_phys_side = new_sides[d]
        cand_side = cand_sides[cand_phys_side]
        if cand_side is None:
            continue
        cand_verts = cand_side['vertices']
        cand_angle = math.atan2(
            cand_verts[-1][1] - cand_verts[0][1],
            cand_verts[-1][0] - cand_verts[0][0]
        ) % (2 * math.pi)

        rot = adj_angle - cand_angle - math.pi
        rotations.append(rot)

        cand_rotated = [util.rotate(v, cand_ic, rot) for v in cand_verts]
        sample = util.subtract(adj_translated[-1], cand_rotated[0])
        translation_samples.append(sample)

    if not rotations:
        print(f"  No rotation computed, skipping")
        continue

    rotation = util.average_angles(rotations)
    translation = util.multimidpoint(translation_samples)

    print(f"  side-based: rotation={math.degrees(rotation):.2f} ORI_TO_ANGLE={ORI_TO_ANGLE.get(ori,0)}")
    print(f"  translation=({translation[0]:.1f},{translation[1]:.1f})")

    result = prepare_from_transform(pid, rotation, translation, cand_ic)
    if result[0] is None:
        print(f"  prepare failed")
        continue
    piece_gray, mask_eroded, mask_raw, piece_rot, paste_pos, size = result
    print(f"  size={size}, paste=({paste_pos[0]:.1f},{paste_pos[1]:.1f})")

    t0 = time.time()
    score, dx, dy, angle = _ncc_search(
        piece_gray, mask_eroded, paste_pos, size, target_aligned)
    elapsed = time.time() - t0
    print(f"  NCC: score={score:.4f} dx={dx} dy={dy} angle={angle:.1f} ({elapsed:.3f}s)")

    out_w, out_h = size
    paste_x, paste_y = paste_pos
    fx = int(paste_x) + dx
    fy = int(paste_y) + dy
    px1 = max(0, fx)
    py1 = max(0, fy)
    px2 = min(tw, fx + out_w)
    py2 = min(th, fy + out_h)
    sx1 = px1 - fx
    sy1 = py1 - fy
    sx2 = sx1 + (px2 - px1)
    sy2 = sy1 + (py2 - py1)
    if sx2 > sx1 and sy2 > sy1:
        vis = target_aligned.copy()
        m = mask_raw[sy1:sy2, sx1:sx2]
        if m.sum() > 50:
            if abs(angle) > 0.01:
                M_rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), angle, 1.0)
                pr = cv2.warpAffine(piece_rot, M_rot, (out_w, out_h),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                ar = cv2.warpAffine(mask_raw.astype(np.uint8) * 255, M_rot, (out_w, out_h),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                m = ar[sy1:sy2, sx1:sx2] > 128
                pr = pr[sy1:sy2, sx1:sx2]
            else:
                pr = piece_rot[sy1:sy2, sx1:sx2]
            bl = m.astype(np.float32) * 0.7
            bl3 = np.stack([bl] * 3, axis=2)
            region = vis[py1:py2, px1:px2]
            vis[py1:py2, px1:px2] = (
                region.astype(np.float32) * (1 - bl3) + pr.astype(np.float32) * bl3
            ).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f'debug_{pid}_at_{gx}_{gy}.png'), vis)
        print(f"  Saved debug_{pid}_at_{gx}_{gy}.png")
