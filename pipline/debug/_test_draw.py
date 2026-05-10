import sys, os, json, math
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from common.config import DEDUPED_DIR
from common.board import Board
from pipline.solve_display import compute_piece_transforms
from pipline.run_matchtarget import load_solution, _resize_to_max

solution_dir = 'output/puzzle_new/6_solution/milestone/pct75'
output_root = 'output/puzzle_new'
deduped_dir = os.path.join(output_root, DEDUPED_DIR)
color_dir = os.path.join(output_root, '2_piece_colors')

pw, ph, placed = load_solution(solution_dir)

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

board = Board(pw, ph)
for pid, info in placed.items():
    fits = ps_raw.get(pid, [[], [], [], []])
    board.place(pid, fits, info['gx'], info['gy'], info['orientation'])

print(f"Board: {pw}x{ph}, placed={board.placed_count}")

piece_transforms, _, canvas_info = compute_piece_transforms(board, deduped_dir)
print(f"Transforms: {len(piece_transforms)} pieces")
print(f"Canvas info: {canvas_info}")

if not piece_transforms or not canvas_info:
    print("ERROR: no transforms!")
    sys.exit(1)

min_x = canvas_info['min_x']
min_y = canvas_info['min_y']
margin = max(canvas_info['max_x'] - min_x, canvas_info['max_y'] - min_y) * 0.05
header_h = 60
gen_scale = 1.0

raw_ass_path = os.path.join(os.path.dirname(solution_dir), '_match_assembly.png')
raw_ass = cv2.imread(raw_ass_path)
_, resize_scale = _resize_to_max(raw_ass, 2000)

target_path = os.path.join(os.path.dirname(solution_dir), 'target_aligned.png')
target = cv2.imread(target_path)
th, tw = target.shape[:2]

print(f"resize_scale={resize_scale}")
print(f"target: {tw}x{th}")

canvas = target.copy()
drawn_count = 0
error_count = 0

report_path = os.path.join(os.path.dirname(solution_dir), 'target_match_report.json')
report = {}
if os.path.exists(report_path):
    with open(report_path, 'r') as f:
        report = json.load(f)

for gy in range(ph):
    for gx in range(pw):
        cell = board.get(gx, gy)
        if cell is None:
            continue
        pid = cell[0]

        if pid not in piece_transforms:
            print(f"  SKIP {pid}: no transform")
            error_count += 1
            continue

        color_path = os.path.join(color_dir, f'piece_{pid}.png')
        if not os.path.exists(color_path):
            print(f"  SKIP {pid}: no color image")
            error_count += 1
            continue

        piece_img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        piece_bgr = piece_img[:, :, :3]
        alpha_raw = piece_img[:, :, 3]
        h_img, w_img = piece_bgr.shape[:2]

        rotation, translation, ic = piece_transforms[pid]
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
        img_pts = []
        for cx, cy in corners:
            dx = cx - ic[0]
            dy = cy - ic[1]
            ox = dx * cos_r - dy * sin_r + ic[0] + translation[0]
            oy = dx * sin_r + dy * cos_r + ic[1] + translation[1]
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

        out_w = max(1, int(out_w_gen * resize_scale))
        out_h = max(1, int(out_h_gen * resize_scale))
        piece_final = cv2.resize(piece_gen, (out_w, out_h), interpolation=cv2.INTER_AREA)
        alpha_final = cv2.resize(alpha_gen, (out_w, out_h), interpolation=cv2.INTER_AREA)

        mask_raw = alpha_final > 128

        nd = report.get(str(pid), {})
        dx = nd.get('dx', 0)
        dy = nd.get('dy', 0)
        angle = nd.get('angle', 0)

        if abs(angle) > 0.01:
            M_rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), angle, 1.0)
            piece_final = cv2.warpAffine(piece_final, M_rot, (out_w, out_h),
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            alpha_u8 = (mask_raw.astype(np.uint8)) * 255
            alpha_rot = cv2.warpAffine(alpha_u8, M_rot, (out_w, out_h),
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask_raw = alpha_rot > 128

        paste_x = (img_min_x - min_x + margin) * gen_scale * resize_scale
        paste_y = ((img_min_y - min_y + margin) * gen_scale + header_h) * resize_scale

        if drawn_count < 3:
            print(f"\n  Piece {pid}: grid({gx},{gy})")
            print(f"    rotation={math.degrees(rotation):.1f}°, ic=({ic[0]:.0f},{ic[1]:.0f})")
            print(f"    gen_size={out_w_gen}x{out_h_gen}, final={out_w}x{out_h}")
            print(f"    paste=({paste_x:.1f},{paste_y:.1f}) dx={dx} dy={dy} angle={angle}")
            print(f"    mask pixels: {mask_raw.sum()}/{mask_raw.size}")

        fx = int(paste_x) + int(dx)
        fy = int(paste_y) + int(dy)
        px1 = max(0, fx)
        py1 = max(0, fy)
        px2 = min(tw, fx + out_w)
        py2 = min(th, fy + out_h)
        sx1 = px1 - fx
        sy1 = py1 - fy
        sx2 = sx1 + (px2 - px1)
        sy2 = sy1 + (py2 - py1)

        if sx2 <= sx1 or sy2 <= sy1:
            print(f"  SKIP {pid}: out of bounds paste=({fx},{fy}) size=({out_w},{out_h})")
            error_count += 1
            continue

        m = mask_raw[sy1:sy2, sx1:sx2]
        if m.sum() < 50:
            print(f"  SKIP {pid}: mask too small ({m.sum()} px)")
            error_count += 1
            continue

        blend = m.astype(np.float32) * 0.95
        blend3 = np.stack([blend] * 3, axis=2)
        region = canvas[py1:py2, px1:px2]
        piece_region = piece_final[sy1:sy2, sx1:sx2]
        canvas[py1:py2, px1:px2] = (
            region.astype(np.float32) * (1 - blend3) +
            piece_region.astype(np.float32) * blend3
        ).astype(np.uint8)
        drawn_count += 1

out = os.path.join(os.path.dirname(solution_dir), 'test_draw.png')
cv2.imwrite(out, canvas)
print(f"\nDrawn: {drawn_count}/{board.placed_count}, errors: {error_count}")
print(f"Saved: {out}")
