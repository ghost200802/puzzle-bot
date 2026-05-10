#!/usr/bin/env python3
import os
import sys
import json
import re
import argparse
import math
import time

import numpy as np
import cv2

_here = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_here, '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'pipline'))

from common.config import SOLUTION_DIR, DEDUPED_DIR
from common.board import Board
from solve_display import compute_piece_transforms

ORI_MAP = {'^': 0, '>': 1, 'v': 2, '<': 3}
ROTATION_DESC = {
    0: "0\u00b0 (no rotation)",
    1: "90\u00b0 clockwise",
    2: "180\u00b0",
    3: "90\u00b0 counter-clockwise",
}

PANEL_W = 700
PANEL_H = 800
OUTLINE_PX = 8


def load_solution(solution_dir):
    with open(os.path.join(solution_dir, 'solution_grid.txt'), 'r') as f:
        grid_text = f.read()
    placed = {}
    lines = [l.strip() for l in grid_text.split('\n')
             if l.strip() and not l.strip().startswith('--')]
    row = 0
    for line in lines:
        tokens = line.split()
        col = 0
        for tok in tokens:
            m = re.match(r'^(\d+)([\^v<>])$', tok)
            if m:
                pid = int(m.group(1))
                ori = ORI_MAP[m.group(2)]
                placed[pid] = {'gx': col, 'gy': row, 'orientation': ori}
            col += 1
        row += 1
    pw = max(p['gx'] for p in placed.values()) + 1 if placed else 0
    ph = max(p['gy'] for p in placed.values()) + 1 if placed else 0
    return pw, ph, placed


def ensure_piece_origins(input_dir, output_dir):
    origins_path = os.path.join(output_dir, 'piece_origins.json')
    if os.path.exists(origins_path):
        return
    print("piece_origins.json not found, extracting from photos...")
    sys.path.insert(0, os.path.join(ROOT_DIR, 'src', 'check'))
    from check_segmentation import segment_image
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    origins = {}
    global_id = 1
    for img_file in image_files:
        print(f"  Processing: {img_file}")
        pieces = segment_image(os.path.join(input_dir, img_file))
        for p in pieces:
            origins[str(global_id)] = {
                'source_file': img_file,
                'tight_bbox': list(p['tight_bbox']),
                'origin': list(p['origin']),
                'crop_size': list(p['crop_size']),
                'scale_factor': p['scale_factor'],
            }
            global_id += 1
    with open(origins_path, 'w') as f:
        json.dump(origins, f, indent=2)
    print(f"  Saved {origins_path} ({len(origins)} pieces)")


def _rotate_point(x, y, W, H, rotation):
    if rotation == 1:
        return y, W - 1 - x
    elif rotation == 2:
        return W - 1 - x, H - 1 - y
    elif rotation == 3:
        return H - 1 - y, x
    return x, y


def _rotate_bbox(bbox, W, H, rotation):
    x0, y0, x1, y1 = bbox
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    rotated = [_rotate_point(x, y, W, H, rotation) for x, y in corners]
    rx0 = min(p[0] for p in rotated)
    ry0 = min(p[1] for p in rotated)
    rx1 = max(p[0] for p in rotated)
    ry1 = max(p[1] for p in rotated)
    return [rx0, ry0, rx1, ry1]


def ensure_assembly_positions(output_dir, targeted_dir):
    asm_pos_path = os.path.join(targeted_dir, 'piece_assembly_positions.json')
    if os.path.exists(asm_pos_path):
        return

    print("piece_assembly_positions.json not found, computing from solution data...")
    pw, ph, placed = load_solution(targeted_dir)
    if not placed:
        return

    board = Board(pw, ph)
    for pid, info in placed.items():
        board.place(pid, [[], [], [], []], info['gx'], info['gy'], info['orientation'])

    deduped_dir = os.path.join(output_dir, DEDUPED_DIR)
    transforms, _, canvas_info = compute_piece_transforms(board, deduped_dir)
    if not transforms or not canvas_info:
        return

    min_x, min_y = canvas_info['min_x'], canvas_info['min_y']
    max_x, max_y = canvas_info['max_x'], canvas_info['max_y']
    data_w = max_x - min_x
    data_h = max_y - min_y
    if data_w == 0 or data_h == 0:
        return

    margin = max(data_w, data_h) * 0.05
    header_h = 60
    canvas_w = int(data_w + 2 * margin)
    canvas_h = int(data_h + 2 * margin) + header_h
    max_size = 12000
    if max(canvas_w, canvas_h) > max_size:
        img_scale = max_size / max(canvas_w, canvas_h)
    else:
        img_scale = 1.0

    color_dir = os.path.join(output_dir, '2_piece_colors')

    positions = {}
    for pid, (rotation, translation, ic) in transforms.items():
        img_path = os.path.join(color_dir, f'piece_{pid}.png')
        if not os.path.exists(img_path):
            continue
        from PIL import Image as PILImage
        piece_img = PILImage.open(img_path)
        w_img, h_img = piece_img.size

        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
        all_pts = []
        for cx, cy in corners:
            dx = cx - ic[0]
            dy = cy - ic[1]
            ox = dx * cos_r - dy * sin_r + ic[0] + translation[0]
            oy = dx * sin_r + dy * cos_r + ic[1] + translation[1]
            all_pts.append((ox, oy))

        bx0 = min(p[0] for p in all_pts)
        by0 = min(p[1] for p in all_pts)
        bx1 = max(p[0] for p in all_pts)
        by1 = max(p[1] for p in all_pts)

        px0 = (bx0 - min_x + margin) * img_scale
        py0 = (by0 - min_y + margin) * img_scale + header_h
        px1 = (bx1 - min_x + margin) * img_scale
        py1 = (by1 - min_y + margin) * img_scale + header_h

        positions[str(pid)] = {
            'bbox': [px0, py0, px1, py1],
            'rotation': rotation,
            'translation': list(translation),
            'incenter': list(ic),
            'img_scale': img_scale,
            'raw_bbox': [px0, py0, px1, py1],
        }

    rotation_back = 0
    report_path = os.path.join(output_dir, 'target_match_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
        rotation_back = report.get('_meta', {}).get('orientation', 0)

    cw = int(canvas_w * img_scale)
    ch = int(canvas_h * img_scale)

    if rotation_back != 0:
        for pid_str, pos in positions.items():
            pos['bbox'] = _rotate_bbox(pos['raw_bbox'], cw, ch, rotation_back)
        if rotation_back in [1, 3]:
            cw, ch = ch, cw

    canvas_info_out = {
        'min_x': min_x, 'min_y': min_y,
        'max_x': max_x, 'max_y': max_y,
        'margin': margin, 'img_scale': img_scale,
        'header_h': header_h,
        'canvas_w': cw, 'canvas_h': ch,
        'rotation_back': rotation_back,
    }

    with open(asm_pos_path, 'w') as f:
        json.dump({'positions': positions, 'canvas_info': canvas_info_out}, f, indent=2)
    print(f"  Saved {asm_pos_path} ({len(positions)} pieces)")


def get_ordered_pieces(placed, pw, ph):
    ordered = []
    for gy in range(ph):
        for gx in range(pw):
            for pid, info in placed.items():
                if info['gx'] == gx and info['gy'] == gy:
                    ordered.append((pid, info))
                    break
    return ordered


def make_photo_overlays(origins, input_dir, bmp_dir):
    photos = {}
    overlays = {}
    for pid_str, orig in origins.items():
        pid = int(pid_str)
        src = orig['source_file']
        if src not in photos:
            p = os.path.join(input_dir, src)
            if os.path.exists(p):
                photos[src] = cv2.imread(p)
        if src not in photos:
            continue
        photo = photos[src]
        ph, pw = photo.shape[:2]

        bmp_path = os.path.join(bmp_dir, f'piece_{pid}.bmp')
        if not os.path.exists(bmp_path):
            continue
        bmp = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
        crop_w, crop_h = orig['crop_size']
        mask = cv2.resize(bmp, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask > 127

        ox, oy = orig['origin']
        ey = min(oy + crop_h, ph)
        ex = min(ox + crop_w, pw)
        ch_actual = ey - oy
        cw_actual = ex - ox
        if ch_actual <= 0 or cw_actual <= 0:
            continue

        local_mask = mask_bool[:ch_actual, :cw_actual]

        overlays[pid] = {
            'src': src,
            'bbox': (ox, oy, ex, ey),
            'mask': local_mask,
        }

    return photos, overlays


def _pil_affine_transform_mask(mask_pil, rotation, translation, ic, img_scale,
                               min_x, min_y, margin, header_h, raw_cw, raw_ch,
                               rotation_back, asm_w, asm_h, bbox):
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    w_bmp, h_bmp = mask_pil.size

    corners = [(0, 0), (w_bmp, 0), (w_bmp, h_bmp), (0, h_bmp)]
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

    out_w = int(math.ceil((img_max_x - img_min_x) * img_scale)) + 2
    out_h = int(math.ceil((img_max_y - img_min_y) * img_scale)) + 2
    if out_w <= 0 or out_h <= 0:
        return None

    cos_neg = math.cos(-rotation)
    sin_neg = math.sin(-rotation)
    sm_x = (img_min_x - ic[0] - translation[0]) * img_scale
    sm_y = (img_min_y - ic[1] - translation[1]) * img_scale

    a = cos_neg * img_scale
    b = -sin_neg * img_scale
    c = cos_neg * sm_x - sin_neg * sm_y + ic[0]
    d = sin_neg * img_scale
    e = cos_neg * img_scale
    f = sin_neg * sm_x + cos_neg * sm_y + ic[1]

    from PIL import Image as PILImage
    transformed = mask_pil.transform(
        (out_w, out_h), PILImage.AFFINE,
        (a, b, c, d, e, f),
        resample=PILImage.NEAREST,
    )
    paste_x = int((img_min_x - min_x + margin) * img_scale)
    paste_y = int((img_min_y - min_y + margin) * img_scale + header_h)

    piece_canvas = np.zeros((raw_ch, raw_cw), dtype=np.uint8)
    mask_arr = np.array(transformed)
    mh, mw = mask_arr.shape
    y1 = min(paste_y + mh, raw_ch)
    x1 = min(paste_x + mw, raw_cw)
    sy = max(0, -paste_y)
    sx = max(0, -paste_x)
    if y1 > paste_y + sy and x1 > paste_x + sx:
        piece_canvas[paste_y + sy:y1, paste_x + sx:x1] = \
            mask_arr[sy:y1 - paste_y, sx:x1 - paste_x]

    if rotation_back != 0:
        rot_codes = {
            1: cv2.ROTATE_90_COUNTERCLOCKWISE,
            2: cv2.ROTATE_180,
            3: cv2.ROTATE_90_CLOCKWISE,
        }
        piece_canvas = cv2.rotate(piece_canvas, rot_codes[rotation_back])

    x0 = max(0, int(round(bbox[0])))
    y0 = max(0, int(round(bbox[1])))
    x1b = min(asm_w, int(round(bbox[2])))
    y1b = min(asm_h, int(round(bbox[3])))
    if x1b <= x0 or y1b <= y0:
        return None
    return piece_canvas[y0:y1b, x0:x1b] > 127


def make_assembly_overlays(assembly_img, asm_positions, bmp_dir, canvas_info):
    from PIL import Image as PILImage

    asm_h, asm_w = assembly_img.shape[:2]
    min_x = canvas_info.get('min_x', 0)
    min_y = canvas_info.get('min_y', 0)
    margin = canvas_info.get('margin', 0)
    header_h = canvas_info.get('header_h', 60)
    img_scale = canvas_info.get('img_scale', 1.0)
    rotation_back = canvas_info.get('rotation_back', 0)

    raw_cw = int((canvas_info.get('max_x', 0) - min_x + 2 * margin) * img_scale)
    raw_ch = int((canvas_info.get('max_y', 0) - min_y + 2 * margin) * img_scale + header_h)

    overlays = {}
    total = len(asm_positions)
    for idx, (pid_str, pos) in enumerate(asm_positions.items()):
        pid = int(pid_str)
        if (idx + 1) % 20 == 0 or idx + 1 == total:
            print(f"    assembly mask [{idx+1}/{total}]")

        bmp_path = os.path.join(bmp_dir, f'piece_{pid}.bmp')
        if not os.path.exists(bmp_path):
            continue

        bmp = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
        mask_pil = PILImage.fromarray(bmp)

        local_mask = _pil_affine_transform_mask(
            mask_pil, pos['rotation'], tuple(pos['translation']),
            tuple(pos['incenter']), img_scale,
            min_x, min_y, margin, header_h,
            raw_cw, raw_ch, rotation_back,
            asm_w, asm_h, pos['bbox'],
        )
        if local_mask is None:
            continue

        bbox = pos['bbox']
        x0 = max(0, int(round(bbox[0])))
        y0 = max(0, int(round(bbox[1])))
        x1 = min(asm_w, int(round(bbox[2])))
        y1 = min(asm_h, int(round(bbox[3])))

        overlays[pid] = {
            'bbox': (x0, y0, x1, y1),
            'mask': local_mask,
        }

    return overlays


def panel_scale(img_h, img_w):
    return min(PANEL_W / img_w, PANEL_H / img_h)


def draw_blue_outline(base, bbox, mask, thickness=8):
    x0, y0, x1, y1 = bbox
    h, w = base.shape[:2]
    pad = thickness + 2
    x0c = max(0, x0 - pad)
    y0c = max(0, y0 - pad)
    x1c = min(w, x1 + pad)
    y1c = min(h, y1 + pad)
    if x1c <= x0c or y1c <= y0c:
        return
    bw = x1 - x0
    bh = y1 - y0
    padded = np.zeros((y1c - y0c, x1c - x0c), dtype=np.uint8)
    dy = y0 - y0c
    dx = x0 - x0c
    src_h = min(bh, y1c - y0)
    src_w = min(bw, x1c - x0)
    if src_h <= 0 or src_w <= 0:
        return
    padded[dy:dy + src_h, dx:dx + src_w] = (mask[:src_h, :src_w] * 255).astype(np.uint8)
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(padded)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness)
    outer_only = (contour_mask > 0) & (padded == 0)
    base[y0c:y1c, x0c:x1c][outer_only] = (255, 100, 0)


def apply_mask_overlay(base, bbox, mask, color_bgr, alpha):
    x0, y0, x1, y1 = bbox
    h, w = base.shape[:2]
    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(w, x1)
    y1c = min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return

    mh = y1 - y0
    mw = x1 - x0
    sx = x0c - x0
    sy = y0c - y0
    ex = sx + (x1c - x0c)
    ey = sy + (y1c - y0c)

    local_mask = mask[sy:ey, sx:ex]
    roi = base[y0c:y1c, x0c:x1c]

    color_arr = np.array(color_bgr, dtype=np.uint8)
    blended = roi.copy()
    blended[local_mask] = (
        roi[local_mask].astype(np.float32) * (1 - alpha) +
        color_arr.astype(np.float32) * alpha
    ).astype(np.uint8)
    base[y0c:y1c, x0c:x1c] = blended


def generate_step_image(step_num, total_steps, pid, info,
                        photo_img, assembly_img, piece_thumb, output_dir,
                        source_name=None):
    panel_w = 700
    panel_h = 800
    header_h = 50
    footer_h = 80
    total_w = panel_w * 2 + 30
    total_h = panel_h + header_h + footer_h + 20

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240

    cv2.rectangle(canvas, (0, 0), (total_w, header_h), (60, 60, 60), -1)
    src_part = f"  [{source_name}]" if source_name else ""
    header_text = f"Step {step_num}/{total_steps}   Piece #{pid}{src_part}   " \
                  f"Row {info['gy']+1}, Col {info['gx']+1}"
    cv2.putText(canvas, header_text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    left_x = 10
    right_x = panel_w + 20
    panel_y = header_h

    if photo_img is not None:
        panel = fit_to_panel(photo_img, panel_w, panel_h)
        canvas[panel_y:panel_y + panel_h, left_x:left_x + panel_w] = panel
    cv2.putText(canvas, "Source Photo" if source_name is None else source_name,
                (left_x + 5, panel_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    if assembly_img is not None:
        panel = fit_to_panel(assembly_img, panel_w, panel_h)
        canvas[panel_y:panel_y + panel_h, right_x:right_x + panel_w] = panel
        cv2.putText(canvas, "Target Position", (right_x + 5, panel_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    footer_y = panel_y + panel_h + 5
    cv2.rectangle(canvas, (0, footer_y), (total_w, total_h), (255, 255, 255), -1)

    if piece_thumb is not None:
        thumb_size = 60
        thumb = cv2.resize(piece_thumb, (thumb_size, thumb_size))
        if thumb.ndim == 2:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        elif thumb.shape[2] == 4:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGRA2BGR)
        tx, ty = 15, footer_y + 10
        canvas[ty:ty + thumb_size, tx:tx + thumb_size] = thumb
        cv2.rectangle(canvas, (tx, ty), (tx + thumb_size, ty + thumb_size), (0, 0, 0), 1)

    info_text = f"Piece #{pid}   Position: Row {info['gy']+1}, Col {info['gx']+1}"
    cv2.putText(canvas, info_text, (90, footer_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

    out_dir = os.path.join(output_dir, 'howtosolve')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'step_{step_num:04d}_piece_{pid}.png')
    cv2.imwrite(out_path, canvas)


def fit_to_panel(img, panel_w, panel_h):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 240
    scale = min(panel_w / w, panel_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    result = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 240
    y_off = (panel_h - new_h) // 2
    x_off = (panel_w - new_w) // 2
    result[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate step-by-step puzzle assembly guide images')
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing original piece photos')
    parser.add_argument('-o', '--output', required=True,
                        help='Pipeline output root directory')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    solution_dir = os.path.join(output_dir, SOLUTION_DIR)
    targeted_dir = os.path.join(output_dir, 'targeted_solve')
    bmp_dir = os.path.join(output_dir, '2_piece_bmps')
    color_dir = os.path.join(output_dir, '2_piece_colors')

    if os.path.isdir(targeted_dir) and os.path.isfile(
            os.path.join(targeted_dir, 'solution_grid.txt')):
        effective_solution = targeted_dir
        print("Using targeted solve results")
    else:
        effective_solution = solution_dir
        print("Using original solve results")

    pw, ph, placed = load_solution(effective_solution)
    print(f"Puzzle: {pw}x{ph}, {len(placed)} pieces placed")

    ensure_piece_origins(input_dir, output_dir)
    with open(os.path.join(output_dir, 'piece_origins.json'), 'r') as f:
        origins = json.load(f)

    ensure_assembly_positions(output_dir, targeted_dir)

    print("Loading photo overlays...")
    photos, photo_overlays = make_photo_overlays(origins, input_dir, bmp_dir)
    print(f"  {len(photo_overlays)} piece masks ready")

    assembly_path = os.path.join(targeted_dir, 'assembly.png')
    if not os.path.exists(assembly_path):
        assembly_path = os.path.join(solution_dir, 'assembly.png')
    assembly_img = cv2.imread(assembly_path)
    print(f"Assembly: {assembly_path}")

    asm_positions = {}
    asm_pos_path = os.path.join(targeted_dir, 'piece_assembly_positions.json')
    if not os.path.exists(asm_pos_path):
        asm_pos_path = os.path.join(solution_dir, 'piece_assembly_positions.json')
    if os.path.exists(asm_pos_path):
        with open(asm_pos_path, 'r') as f:
            asm_data = json.load(f)
        asm_positions = asm_data.get('positions', {})
        canvas_info = asm_data.get('canvas_info', {})
        print(f"Loaded assembly positions: {len(asm_positions)} pieces")

    print("Computing assembly overlays...")
    asm_overlays = {}
    if assembly_img is not None and asm_positions:
        asm_overlays = make_assembly_overlays(assembly_img, asm_positions, bmp_dir, canvas_info)
        print(f"  {len(asm_overlays)} assembly masks ready")

    ordered = get_ordered_pieces(placed, pw, ph)
    total = len(ordered)
    print(f"\nGenerating {total} step images...")

    t0 = time.time()

    accumulated_photo = {}
    for src, base in photos.items():
        accumulated_photo[src] = base.copy()

    accumulated_asm = assembly_img.copy() if assembly_img is not None else None

    photo_thickness_cache = {}
    asm_thickness = max(2, int(OUTLINE_PX / panel_scale(assembly_img.shape[0], assembly_img.shape[1]))) if accumulated_asm is not None else 8

    for i, (pid, info) in enumerate(ordered):
        step_num = i + 1
        if step_num % 10 == 0 or step_num == total:
            elapsed = time.time() - t0
            eta = elapsed / step_num * (total - step_num) if step_num > 0 else 0
            print(f"  [{step_num}/{total}] {elapsed:.1f}s elapsed, ETA {eta:.1f}s")

        photo_img = None
        current_src = None
        if pid in photo_overlays:
            ov = photo_overlays[pid]
            current_src = ov['src']
            base = accumulated_photo.get(current_src)
            if base is not None:
                photo_img = base.copy()
                if current_src not in photo_thickness_cache:
                    ph, pw = base.shape[:2]
                    photo_thickness_cache[current_src] = max(2, int(OUTLINE_PX / panel_scale(ph, pw)))
                draw_blue_outline(photo_img, ov['bbox'], ov['mask'], photo_thickness_cache[current_src])

        assembly_step = None
        if accumulated_asm is not None and pid in asm_overlays:
            assembly_step = accumulated_asm.copy()
            ov = asm_overlays[pid]
            draw_blue_outline(assembly_step, ov['bbox'], ov['mask'], asm_thickness)

        piece_thumb = None
        p = os.path.join(color_dir, f'piece_{pid}.png')
        if os.path.exists(p):
            piece_thumb = cv2.imread(p, cv2.IMREAD_UNCHANGED)

        generate_step_image(step_num, total, pid, info,
                            photo_img, assembly_step, piece_thumb, output_dir,
                            source_name=current_src)

        if pid in photo_overlays:
            ov = photo_overlays[pid]
            base = accumulated_photo.get(ov['src'])
            if base is not None:
                m = ov['mask'].astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                dilated = cv2.dilate(m, kernel, iterations=1) > 0
                apply_mask_overlay(base, ov['bbox'], dilated, (0, 0, 0), 1.0)

        if accumulated_asm is not None and pid in asm_overlays:
            ov = asm_overlays[pid]
            apply_mask_overlay(accumulated_asm, ov['bbox'], ov['mask'], (0, 200, 0), 0.3)

    elapsed = time.time() - t0
    howtosolve_dir = os.path.join(output_dir, 'howtosolve')
    print(f"\n{'=' * 60}")
    print(f"How-to-solve guide complete! ({elapsed:.1f}s)")
    print(f"  Steps: {total}")
    print(f"  Output: {howtosolve_dir}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
