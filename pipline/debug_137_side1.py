import os
import sys
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common import pieces, sides, util

DEDUPED_DIR = os.path.join(_here, '..', 'output', 'puzzle_new', '4_deduped')
OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new', 'check', 'debug_137')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PID = 137
TARGET_SI = 1
RELAXED_LEN_DISC = 0.30
RELAXED_ERROR = 5.0
MAX_SHOW = 15


def _side_angle_v(vertices):
    p1, p2 = vertices[0], vertices[-1]
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def _side_midpoint_v(vertices):
    p1, p2 = vertices[0], vertices[-1]
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _render_pair_cell_v2(pid_a, si_a, pid_b, si_b, piece_data, piece_imgs, cell_w, cell_h):
    sides_a = piece_data.get(pid_a)
    sides_b = piece_data.get(pid_b)
    img_a = piece_imgs.get(pid_a)
    img_b = piece_imgs.get(pid_b)

    if not sides_a or not sides_b:
        cell = Image.new('RGBA', (cell_w, cell_h), (240, 240, 240, 255))
        d = ImageDraw.Draw(cell)
        d.text((10, 10), "Missing data", fill=(200, 0, 0, 255))
        return cell

    target_side_verts = sides_a[si_a]['vertices']
    tgt_mid = _side_midpoint_v(target_side_verts)
    tgt_theta = _side_angle_v(target_side_verts)
    src_verts = sides_b[si_b]['vertices']
    src_mid = _side_midpoint_v(src_verts)
    src_theta = _side_angle_v(src_verts)
    rot_angle = tgt_theta + math.pi - src_theta
    cos_r = math.cos(rot_angle)
    sin_r = math.sin(rot_angle)

    all_pts = []
    if img_a:
        all_pts.extend([(0, 0), (img_a.size[0], 0), (img_a.size[0], img_a.size[1]), (0, img_a.size[1])])
    else:
        for side in sides_a:
            all_pts.extend(side['vertices'])

    if img_b:
        w_b, h_b = img_b.size
        for x, y in [(0, 0), (w_b, 0), (w_b, h_b), (0, h_b)]:
            dx = x - src_mid[0]
            dy = y - src_mid[1]
            ox = dx * cos_r - dy * sin_r + tgt_mid[0]
            oy = dx * sin_r + dy * cos_r + tgt_mid[1]
            all_pts.append((ox, oy))
    else:
        for side in sides_b:
            new_verts = []
            for v in side['vertices']:
                dx = v[0] - src_mid[0]
                dy = v[1] - src_mid[1]
                fx = dx * cos_r - dy * sin_r
                fy = dx * sin_r + dy * cos_r
                new_verts.append((fx + tgt_mid[0], fy + tgt_mid[1]))
            all_pts.extend(new_verts)

    pts_x = [p[0] for p in all_pts]
    pts_y = [p[1] for p in all_pts]
    data_min_x = min(pts_x)
    data_min_y = min(pts_y)
    data_max_x = max(pts_x)
    data_max_y = max(pts_y)
    data_w = data_max_x - data_min_x
    data_h = data_max_y - data_min_y
    if data_w == 0:
        data_w = 1
    if data_h == 0:
        data_h = 1

    pad = 10
    scale = min((cell_w - 2 * pad) / data_w, (cell_h - 2 * pad) / data_h)

    cell_img = Image.new('RGBA', (cell_w, cell_h), (255, 255, 255, 255))

    def to_cell(x, y):
        cx = (x - data_min_x) * scale + pad + ((cell_w - 2 * pad) - data_w * scale) / 2
        cy = (y - data_min_y) * scale + pad + ((cell_h - 2 * pad) - data_h * scale) / 2
        return (cx, cy)

    if img_b:
        w_b, h_b = img_b.size
        corners_b = [(0, 0), (w_b, 0), (w_b, h_b), (0, h_b)]
        out_b = []
        for x, y in corners_b:
            dx = x - src_mid[0]
            dy = y - src_mid[1]
            ox = dx * cos_r - dy * sin_r + tgt_mid[0]
            oy = dx * sin_r + dy * cos_r + tgt_mid[1]
            out_b.append((ox, oy))
        bx = [c[0] for c in out_b]
        by = [c[1] for c in out_b]
        b_min_x, b_min_y = min(bx), min(by)
        b_max_x, b_max_y = max(bx), max(by)
        out_w = int(math.ceil(b_max_x - b_min_x)) + 1
        out_h = int(math.ceil(b_max_y - b_min_y)) + 1
        cos_neg = math.cos(-rot_angle)
        sin_neg = math.sin(-rot_angle)
        a = cos_neg
        b_coef = -sin_neg
        c_coef = cos_neg * (b_min_x - tgt_mid[0]) - sin_neg * (b_min_y - tgt_mid[1]) + src_mid[0]
        d = sin_neg
        e = cos_neg
        f_coef = sin_neg * (b_min_x - tgt_mid[0]) + cos_neg * (b_min_y - tgt_mid[1]) + src_mid[1]
        try:
            transformed_b_img = img_b.transform(
                (out_w, out_h), Image.AFFINE,
                (a, b_coef, c_coef, d, e, f_coef),
                resample=Image.BICUBIC,
            )
            b_off = (b_min_x, b_min_y)
            px = int(to_cell(b_off[0], b_off[1])[0])
            py = int(to_cell(b_off[0], b_off[1])[1])
            tw = int(transformed_b_img.size[0] * scale)
            th = int(transformed_b_img.size[1] * scale)
            if tw > 0 and th > 0:
                resized_b = transformed_b_img.resize((tw, th), Image.LANCZOS)
                cell_img.paste(resized_b, (px, py), resized_b)
        except Exception:
            pass

    if img_a:
        px = int(to_cell(0, 0)[0])
        py = int(to_cell(0, 0)[1])
        tw = int(img_a.size[0] * scale)
        th = int(img_a.size[1] * scale)
        if tw > 0 and th > 0:
            resized_a = img_a.resize((tw, th), Image.LANCZOS)
            cell_img.paste(resized_a, (px, py), resized_a)

    return cell_img


def main():
    print("=" * 60)
    print(f"Debug: Piece {TARGET_PID} Side {TARGET_SI} - Relaxed matching")
    print("=" * 60)

    ps_raw = pieces.Piece.load_all(DEDUPED_DIR, resample=False)
    ps_res = pieces.Piece.load_all(DEDUPED_DIR, resample=True)

    piece_137_raw = ps_raw[TARGET_PID]
    piece_137_res = ps_res[TARGET_PID]

    print("\n--- Piece 137 side info ---")
    for si in range(4):
        s = piece_137_raw.sides[si]
        print(f"  Side {si}: angle={math.degrees(s.original_angle):.1f} deg, "
              f"len={s.original_length:.1f}, is_edge={s.is_edge}, "
              f"is_convex={s.is_convex}, center_sign={s.center_side_sign():.3f}")

    target_side_raw = piece_137_raw.sides[TARGET_SI]
    target_side_res = piece_137_res.sides[TARGET_SI]

    print(f"\n--- Target: Side {TARGET_SI} ---")
    print(f"  original_angle = {math.degrees(target_side_raw.original_angle):.2f} deg")
    print(f"  original_length = {target_side_raw.original_length:.2f}")
    print(f"  is_convex = {target_side_raw.is_convex}")
    print(f"  center_side_sign = {target_side_raw.center_side_sign():.4f}")

    print(f"\n--- Relaxed thresholds ---")
    print(f"  SIDE_MAX_LENGTH_DISCREPANCY: {sides.SIDE_MAX_LENGTH_DISCREPANCY} -> {RELAXED_LEN_DISC}")
    print(f"  SIDE_MAX_ERROR_TO_MATCH: {sides.SIDE_MAX_ERROR_TO_MATCH} -> {RELAXED_ERROR}")

    candidates = []
    reject_reasons = {}

    for other_pid in sorted(ps_raw.keys()):
        if other_pid == TARGET_PID:
            continue
        other_raw = ps_raw[other_pid]
        other_res = ps_res[other_pid]

        for sj in range(4):
            other_side_raw = other_raw.sides[sj]
            other_side_res = other_res.sides[sj]

            if other_side_raw.is_edge:
                reason = "other_is_edge"
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue

            len_a = target_side_raw.original_length
            len_b = other_side_raw.original_length
            if len_a < 1 or len_b < 1:
                reason = "zero_length"
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue
            d_scale = abs(1.0 - (len_a / len_b))
            if d_scale > RELAXED_LEN_DISC:
                reason = "len_diff_too_large"
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue

            if target_side_raw.is_convex is not None and other_side_raw.is_convex is not None:
                if target_side_raw.is_convex == other_side_raw.is_convex:
                    reason = "same_convexity"
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                    continue

            sd_a = target_side_raw.center_side_sign()
            sd_b = other_side_raw.center_side_sign()
            if sd_a != 0 and sd_b != 0 and sd_a * sd_b < 0:
                reason = "center_sign_mismatch"
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue

            rot_for_b = target_side_raw.original_angle + math.pi - other_side_raw.original_angle
            adj_map = {
                (TARGET_SI - 1) % 4: (sj + 1) % 4,
                (TARGET_SI + 1) % 4: (sj - 1) % 4,
            }
            adj_ok = True
            adj_reason = ""
            for adj_a_si, adj_b_si in adj_map.items():
                adj_a = piece_137_raw.sides[adj_a_si]
                adj_b = other_raw.sides[adj_b_si]

                if adj_a.is_edge != adj_b.is_edge:
                    adj_reason = f"adj_edge_mismatch(a{adj_a_si}={adj_a.is_edge},b{adj_b_si}={adj_b.is_edge})"
                    adj_ok = False
                    break

                if adj_a.is_edge and adj_b.is_edge:
                    adj_b_rotated = adj_b.original_angle + rot_for_b
                    angle_diff = util.compare_angles(adj_a.original_angle, adj_b_rotated)
                    if angle_diff > sides.EDGE_PARALLEL_THRESHOLD_RAD:
                        adj_reason = f"adj_edge_not_parallel(a{adj_a_si},b{adj_b_si},diff={math.degrees(angle_diff):.1f}deg)"
                        adj_ok = False
                        break

            if not adj_ok:
                reject_reasons[adj_reason] = reject_reasons.get(adj_reason, 0) + 1
                continue

            error, shift = target_side_res.error_when_fit_with(
                other_side_res, flip=True, skip_edges=False
            )

            len_diff = abs(1.0 - (target_side_raw.original_length / other_side_raw.original_length))

            candidates.append({
                'pid': other_pid,
                'si': sj,
                'error': error,
                'len_diff': len_diff,
                'shift_x': float(shift[0]),
                'shift_y': float(shift[1]),
                'convex_a': target_side_raw.is_convex,
                'convex_b': other_side_raw.is_convex,
                'passed_old_len': d_scale <= sides.SIDE_MAX_LENGTH_DISCREPANCY,
                'passed_old_error': error <= sides.SIDE_MAX_ERROR_TO_MATCH,
            })

    candidates.sort(key=lambda x: x['error'])

    print(f"\n--- Reject reasons summary ---")
    for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    print(f"\n--- Candidates that passed all relaxed filters: {len(candidates)} ---")

    old_pass = [c for c in candidates if c['passed_old_len'] and c['passed_old_error']]
    new_only = [c for c in candidates if not (c['passed_old_len'] and c['passed_old_error'])]

    print(f"  Would pass OLD thresholds: {len(old_pass)}")
    print(f"  Only pass RELAXED thresholds: {len(new_only)}")

    print(f"\n--- Top {min(30, len(candidates))} candidates (by error) ---")
    for i, c in enumerate(candidates[:30]):
        old_tag = "OLD_PASS" if (c['passed_old_len'] and c['passed_old_error']) else "RELAX"
        print(f"  #{i+1}: pid={c['pid']}, si={c['si']}, error={c['error']:.4f}, "
              f"len_diff={c['len_diff']:.4f}, convex_a={c['convex_a']}, convex_b={c['convex_b']}, "
              f"shift=({c['shift_x']:.1f},{c['shift_y']:.1f}), [{old_tag}]")

    print("\n--- Generating visualization ---")

    pids_needed = set([TARGET_PID] + [c['pid'] for c in candidates[:MAX_SHOW]])
    piece_data = {}
    for pid in pids_needed:
        if pid in ps_raw:
            sides_list = []
            for si2 in range(4):
                s = ps_raw[pid].sides[si2]
                sides_list.append({
                    'vertices': [tuple(v) for v in s.vertices],
                    'is_edge': s.is_edge,
                })
            piece_data[pid] = sides_list

    piece_imgs = {}
    parent = os.path.dirname(DEDUPED_DIR)
    color_dir = os.path.join(parent, '2_piece_colors')
    if os.path.exists(color_dir):
        from pathlib import Path
        for path in sorted(Path(color_dir).glob("piece_*.png")):
            pid = int(path.stem.split('_')[1])
            if pid in pids_needed:
                piece_imgs[pid] = Image.open(str(path)).convert('RGBA')

    show_candidates = candidates[:MAX_SHOW]

    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        label_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = title_font

    cell_w = 500
    cell_h = 500
    gap = 12
    header_h = 80
    cols = 3
    rows = (MAX_SHOW + cols - 1) // cols

    canvas_w = cols * (cell_w + gap) + gap
    canvas_h = header_h + rows * (cell_h + 40 + gap) + gap

    canvas = Image.new('RGBA', (canvas_w, canvas_h), (235, 235, 235, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((gap, 8), f"Piece #{TARGET_PID} Side {TARGET_SI} - Relaxed Candidates",
              fill=(0, 0, 0, 255), font=title_font)
    draw.text((gap, 45),
              f"Top {len(show_candidates)}/{len(candidates)} candidates | "
              f"OLD: len<={sides.SIDE_MAX_LENGTH_DISCREPANCY}, err<={sides.SIDE_MAX_ERROR_TO_MATCH} | "
              f"RELAXED: len<={RELAXED_LEN_DISC}, err<={RELAXED_ERROR}",
              fill=(80, 80, 80, 255), font=label_font)

    for idx, c in enumerate(show_candidates):
        col = idx % cols
        row = idx // cols
        x_off = gap + col * (cell_w + gap)
        y_off = header_h + row * (cell_h + 40 + gap)

        other_pid = c['pid']
        other_si = c['si']

        old_pass_tag = "PASS" if (c['passed_old_len'] and c['passed_old_error']) else "RELAX"

        cell = _render_pair_cell_v2(
            TARGET_PID, TARGET_SI, other_pid, other_si,
            piece_data, piece_imgs, cell_w, cell_h
        )
        canvas.paste(cell, (x_off, y_off))

        border_color = (0, 128, 0, 255) if old_pass_tag == "PASS" else (200, 100, 0, 255)
        draw.rectangle([x_off, y_off, x_off + cell_w, y_off + cell_h],
                       outline=border_color, width=2)

        label = f"#{other_pid}[s{other_si}] err={c['error']:.3f} ld={c['len_diff']:.3f} [{old_pass_tag}]"
        draw.text((x_off + 6, y_off + cell_h + 5), label, fill=(0, 0, 0, 255), font=label_font)

    out_path = os.path.join(OUTPUT_DIR, 'piece_137_side1_relaxed.png')
    canvas.save(out_path)
    print(f"\nSaved: {out_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
