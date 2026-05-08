import sys, os, json, math
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipline'))

from pipline.run_matchtarget import load_solution, _build_board_from_placed, _resize_to_max, TargetMatcher
from common.config import DEDUPED_DIR

solution_dir = 'output/puzzle_new/6_solution/milestone/pct75'
output_root = 'output/puzzle_new'

pw, ph, placed = load_solution(solution_dir)
output_dir = os.path.dirname(solution_dir)

report_path = os.path.join(output_dir, 'target_match_report.json')
with open(report_path, 'r') as f:
    report = json.load(f)

matcher = TargetMatcher(
    target_image_path='input/puzzles/TargetImage.png',
    pw=pw, ph=ph,
    placed=placed,
    output_dir=output_dir,
    output_root=output_root,
)
matcher.rectify_target()

target_h, target_w = matcher.target_aligned.shape[:2]
canvas = np.full((target_h, target_w, 4), 255, dtype=np.uint8)
canvas[:, :, 3] = 0

done = 0
total = len(report)
for pid_str, r in report.items():
    pid = int(pid_str)
    if pid not in placed:
        continue
    done += 1

    piece_gray, mask_eroded, paste_pos, size, piece_bgr, mask_raw = matcher._prepare_piece(pid)
    if piece_bgr is None:
        print(f"  {pid}: no piece image")
        continue

    paste_x, paste_y = paste_pos
    out_w, out_h = size
    dx = r.get('dx', 0)
    dy = r.get('dy', 0)
    angle = r.get('angle', 0)
    score = r.get('score', 0)

    if abs(angle) > 0.01:
        M_rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), angle, 1.0)
        piece_bgr = cv2.warpAffine(piece_bgr, M_rot, (out_w, out_h),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        alpha_uint8 = (mask_raw.astype(np.uint8)) * 255
        alpha_rot = cv2.warpAffine(alpha_uint8, M_rot, (out_w, out_h),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_raw = alpha_rot > 128

    fx = int(paste_x) + int(dx)
    fy = int(paste_y) + int(dy)

    px1 = max(0, fx)
    py1 = max(0, fy)
    px2 = min(target_w, fx + out_w)
    py2 = min(target_h, fy + out_h)

    sx1 = px1 - fx
    sy1 = py1 - fy
    sx2 = sx1 + (px2 - px1)
    sy2 = sy1 + (py2 - py1)

    if sx2 <= sx1 or sy2 <= sy1:
        continue

    m = mask_raw[sy1:sy2, sx1:sx2]
    if m.sum() < 50:
        continue

    piece_region = piece_bgr[sy1:sy2, sx1:sx2]
    alpha_region = m.astype(np.uint8) * 255

    existing_alpha = canvas[py1:py2, px1:px2, 3].astype(np.float32) / 255.0
    new_alpha = alpha_region.astype(np.float32) / 255.0

    for c in range(3):
        existing = canvas[py1:py2, px1:px2, c].astype(np.float32) * existing_alpha
        new_val = piece_region[:, :, c].astype(np.float32) * new_alpha
        total_alpha = existing_alpha + new_alpha
        total_alpha = np.where(total_alpha > 0, total_alpha, 1.0)
        canvas[py1:py2, px1:px2, c] = np.clip(
            (existing + new_val) / total_alpha, 0, 255).astype(np.uint8)
    canvas[py1:py2, px1:px2, 3] = np.clip(
        (existing_alpha + new_alpha) * 255, 0, 255).astype(np.uint8)

    if done % 10 == 0:
        print(f"  Placed {done}/{total}")

out_path = os.path.join(output_dir, 'pieces_corrected.png')
cv2.imwrite(out_path, canvas)
print(f"\nSaved: {out_path}")

# Also create side-by-side with target
target = matcher.target_aligned
canvas_bgr = canvas[:, :, :3]
canvas_mask = canvas[:, :, 3]
target_mask = canvas_mask > 0
blend = target_mask.astype(np.float32) / 255.0 * 0.7
blend3 = np.stack([blend] * 3, axis=2)
blended = target.astype(np.float32) * (1 - blend3) + canvas_bgr.astype(np.float32) * blend3
blended = np.clip(blended, 0, 255).astype(np.uint8)

blend_path = os.path.join(output_dir, 'pieces_on_target.png')
cv2.imwrite(blend_path, blended)
print(f"Saved: {blend_path}")
