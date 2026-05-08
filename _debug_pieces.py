import sys, os, json, math, time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipline'))

from pipline.run_matchtarget import load_solution, _build_board_from_placed, _resize_to_max
from pipline.solve_display import compute_piece_transforms
from common.config import DEDUPED_DIR

solution_dir = 'output/puzzle_new/6_solution/milestone/pct75'
output_root = 'output/puzzle_new'
deduped_dir = os.path.join(output_root, DEDUPED_DIR)
color_dir = os.path.join(output_root, '2_piece_colors')

out = os.path.join(output_root, '6_solution', 'milestone', 'target_match')
os.makedirs(out, exist_ok=True)

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

sol_board = _build_board_from_placed(pw, ph, placed, ps_raw)
piece_transforms, _, canvas_info = compute_piece_transforms(sol_board, deduped_dir)

min_x = canvas_info['min_x']
min_y = canvas_info['min_y']
data_w = canvas_info['max_x'] - min_x
data_h = canvas_info['max_y'] - min_y
margin = max(data_w, data_h) * 0.05
header_h = 60
gen_scale = 1.0

assembly_path = os.path.join(output_root, '6_solution', 'milestone', '_match_assembly.png')
target_path = os.path.join(output_root, '6_solution', 'milestone', 'target_aligned.png')
raw_assembly = cv2.imread(assembly_path)
_, resize_scale = _resize_to_max(raw_assembly, 2000)
target = cv2.imread(target_path)
th, tw = target.shape[:2]

pid = 109
rotation, translation, ic = piece_transforms[pid]
color_path = os.path.join(color_dir, f'piece_{pid}.png')
piece_raw = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
piece_bgr = piece_raw[:, :, :3]
alpha_raw = piece_raw[:, :, 3]
h_img, w_img = piece_bgr.shape[:2]

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
                            flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
alpha_gen = cv2.warpAffine(alpha_raw, M_aff, (out_w_gen, out_h_gen),
                            flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

out_w = max(1, int(out_w_gen * resize_scale))
out_h = max(1, int(out_h_gen * resize_scale))
piece_final = cv2.resize(piece_gen, (out_w, out_h), interpolation=cv2.INTER_AREA)
alpha_final = cv2.resize(alpha_gen, (out_w, out_h), interpolation=cv2.INTER_AREA)

paste_x = (img_min_x - min_x + margin) * gen_scale * resize_scale
paste_y = ((img_min_y - min_y + margin) * gen_scale + header_h) * resize_scale

piece_gray = cv2.cvtColor(piece_final, cv2.COLOR_BGR2GRAY)

# Erode mask
erode_px = 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
mask_eroded = cv2.erode((alpha_final > 128).astype(np.uint8), kernel) > 0

print(f"Piece {pid}: {out_w}x{out_h}, mask eroded: {mask_eroded.sum()}/{mask_eroded.size} ({100*mask_eroded.sum()/mask_eroded.size:.1f}%)")

# ===== Phase 1: Translation search =====
search_margin = 40
tx1 = max(0, int(paste_x) - search_margin)
ty1 = max(0, int(paste_y) - search_margin)
tx2 = min(tw, int(paste_x + out_w) + search_margin)
ty2 = min(th, int(paste_y + out_h) + search_margin)
target_region = target[ty1:ty2, tx1:tx2]
target_region_gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY)
base_x = int(paste_x) - tx1
base_y = int(paste_y) - ty1

t0 = time.time()
best_score = -999
best_dx = 0
best_dy = 0

coarse_step = 4
for dy in range(-search_margin, search_margin + 1, coarse_step):
    for dx in range(-search_margin, search_margin + 1, coarse_step):
        rx = base_x + dx
        ry = base_y + dy
        x1t = max(0, rx)
        y1t = max(0, ry)
        x2t = min(target_region_gray.shape[1], rx + out_w)
        y2t = min(target_region_gray.shape[0], ry + out_h)
        x1p = x1t - rx
        y1p = y1t - ry
        pw_ = x2t - x1t
        ph_ = y2t - y1t
        if pw_ <= 0 or ph_ <= 0:
            continue
        m = mask_eroded[y1p:y1p+ph_, x1p:x1p+pw_]
        if m.sum() < 100:
            continue
        p = piece_gray[y1p:y1p+ph_, x1p:x1p+pw_][m].astype(np.float64)
        t = target_region_gray[y1t:y1t+ph_, x1t:x1t+pw_][m].astype(np.float64)
        p_n = p - p.mean()
        t_n = t - t.mean()
        denom = np.sqrt(np.sum(p_n**2) * np.sum(t_n**2))
        score = float(np.sum(p_n * t_n) / denom) if denom > 1e-6 else 0.0
        if score > best_score:
            best_score = score
            best_dx = dx
            best_dy = dy

t1 = time.time()

fine_range = coarse_step + 1
for dy in range(best_dy - fine_range, best_dy + fine_range + 1):
    for dx in range(best_dx - fine_range, best_dx + fine_range + 1):
        rx = base_x + dx
        ry = base_y + dy
        x1t = max(0, rx)
        y1t = max(0, ry)
        x2t = min(target_region_gray.shape[1], rx + out_w)
        y2t = min(target_region_gray.shape[0], ry + out_h)
        x1p = x1t - rx
        y1p = y1t - ry
        pw_ = x2t - x1t
        ph_ = y2t - y1t
        if pw_ <= 0 or ph_ <= 0:
            continue
        m = mask_eroded[y1p:y1p+ph_, x1p:x1p+pw_]
        if m.sum() < 100:
            continue
        p = piece_gray[y1p:y1p+ph_, x1p:x1p+pw_][m].astype(np.float64)
        t = target_region_gray[y1t:y1t+ph_, x1t:x1t+pw_][m].astype(np.float64)
        p_n = p - p.mean()
        t_n = t - t.mean()
        denom = np.sqrt(np.sum(p_n**2) * np.sum(t_n**2))
        score = float(np.sum(p_n * t_n) / denom) if denom > 1e-6 else 0.0
        if score > best_score:
            best_score = score
            best_dx = dx
            best_dy = dy

t2 = time.time()
print(f"\nPhase 1 (translation): NCC={best_score:.4f} dx={best_dx} dy={best_dy}")
print(f"  Coarse: {t1-t0:.2f}s, Fine: {t2-t1:.2f}s, Total: {t2-t0:.2f}s")

# ===== Phase 2: Rotation search at best translation =====
fx = int(paste_x) + best_dx
fy = int(paste_y) + best_dy
fx1 = max(0, fx)
fy1 = max(0, fy)
fx2 = min(tw, fx + out_w)
fy2 = min(th, fy + out_h)
px1 = fx1 - fx
py1 = fy1 - fy
pw_ = fx2 - fx1
ph_ = fy2 - fy1

target_at_best = target[fy1:fy2, fx1:fx2]
target_at_best_gray = cv2.cvtColor(target_at_best, cv2.COLOR_BGR2GRAY)

angles = np.arange(-5, 5.5, 0.5)
best_angle = 0.0

for angle in angles:
    if abs(angle) < 0.01:
        rotated_gray = piece_gray[py1:py1+ph_, px1:px1+pw_]
        rotated_mask = mask_eroded[py1:py1+ph_, px1:px1+pw_]
    else:
        M_rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), angle, 1.0)
        rotated_bgr = cv2.warpAffine(piece_final, M_rot, (out_w, out_h),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        rotated_gray = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2GRAY)
        rotated_alpha = cv2.warpAffine(alpha_final, M_rot, (out_w, out_h),
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_mask = cv2.erode((rotated_alpha > 128).astype(np.uint8), kernel) > 0
        rotated_gray = rotated_gray[py1:py1+ph_, px1:px1+pw_]
        rotated_mask = rotated_mask[py1:py1+ph_, px1:px1+pw_]

    m = rotated_mask
    if m.sum() < 100:
        continue
    p = rotated_gray[m].astype(np.float64)
    t = target_at_best_gray[m].astype(np.float64)
    p_n = p - p.mean()
    t_n = t - t.mean()
    denom = np.sqrt(np.sum(p_n**2) * np.sum(t_n**2))
    score = float(np.sum(p_n * t_n) / denom) if denom > 1e-6 else 0.0
    if score > best_score:
        best_score = score
        best_angle = angle

t3 = time.time()
print(f"\nPhase 2 (rotation): NCC={best_score:.4f} angle={best_angle:.1f}°")
print(f"  Rotation search: {t3-t2:.2f}s ({len(angles)} angles)")

# ===== Phase 3: Histmatch NCC at final best =====
if abs(best_angle) < 0.01:
    final_gray = piece_gray[py1:py1+ph_, px1:px1+pw_]
    final_mask = mask_eroded[py1:py1+ph_, px1:px1+pw_]
else:
    M_rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), best_angle, 1.0)
    rot_bgr = cv2.warpAffine(piece_final, M_rot, (out_w, out_h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    final_gray = cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2GRAY)
    rot_alpha = cv2.warpAffine(alpha_final, M_rot, (out_w, out_h),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    final_mask = cv2.erode((rot_alpha > 128).astype(np.uint8), kernel) > 0
    final_gray = final_gray[py1:py1+ph_, px1:px1+pw_]
    final_mask = final_mask[py1:py1+ph_, px1:px1+pw_]

p = final_gray[final_mask].astype(np.float64)
t = target_at_best_gray[final_mask].astype(np.float64)
src_hist, _ = np.histogram(p.astype(np.uint8), bins=256, range=(0, 256))
ref_hist, _ = np.histogram(t.astype(np.uint8), bins=256, range=(0, 256))
src_cdf = np.cumsum(src_hist).astype(np.float64)
ref_cdf = np.cumsum(ref_hist).astype(np.float64)
src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1
ref_cdf /= ref_cdf[-1] if ref_cdf[-1] > 0 else 1
hm_map = np.zeros(256, dtype=np.uint8)
for i in range(256):
    hm_map[i] = int(np.argmin(np.abs(ref_cdf - src_cdf[i])))
pm = hm_map[p.astype(np.uint8)].astype(np.float64)
pm_n = pm - pm.mean()
t_n = t - t.mean()
denom = np.sqrt(np.sum(pm_n**2) * np.sum(t_n**2))
ncc_hm = float(np.sum(pm_n * t_n) / denom) if denom > 1e-6 else 0.0

print(f"\n===== FINAL =====")
print(f"  dx={best_dx} dy={best_dy} angle={best_angle:.1f}°")
print(f"  Raw NCC:    {best_score:.4f}")
print(f"  Histmatch:  {ncc_hm:.4f}")
print(f"  Total time: {t3-t0:.2f}s")

# ===== Visualize =====
final_piece_bgr = piece_final[py1:py1+ph_, px1:px1+pw_]
if abs(best_angle) > 0.01:
    M_rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), best_angle, 1.0)
    final_piece_bgr = cv2.warpAffine(piece_final, M_rot, (out_w, out_h),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    final_piece_bgr = final_piece_bgr[py1:py1+ph_, px1:px1+pw_]

blend = final_mask.astype(np.float32) / 255.0
blend3 = np.stack([blend]*3, axis=2)
overlay = (target_at_best.astype(np.float32) * (1 - blend3 * 0.5) + final_piece_bgr.astype(np.float32) * (blend3 * 0.5)).astype(np.uint8)

cv2.imwrite(os.path.join(out, 'R1_final_overlay_eroded5.png'), overlay)

vis = target.copy()
cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
cv2.putText(vis, f"P{pid} dx={best_dx} dy={best_dy} a={best_angle:.1f} NCC={best_score:.3f}",
            (fx1, fy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
cv2.imwrite(os.path.join(out, 'R1_position_on_target.png'), vis)

comparison = np.hstack([final_piece_bgr, target_at_best, overlay])
cv2.imwrite(os.path.join(out, 'R1_piece_target_overlay.png'), comparison)

print(f"\nSaved R1 images to {out}")
