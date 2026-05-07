import os
import sys
import json
import math
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common import pieces, sides, util
from PIL import Image, ImageDraw, ImageFont


OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new', 'check', 'match_debug')
DEDUPED_DIR = os.path.join(_here, '..', 'output', 'puzzle_new', '4_deduped')


def _load_raw_piece_sides(pid):
    sides_data = []
    for i in range(4):
        path = os.path.join(DEDUPED_DIR, f'side_{pid}_{i}.json')
        with open(path) as f:
            data = json.load(f)
        sides_data.append({
            'vertices': [tuple(v) for v in data['vertices']],
            'is_edge': data.get('is_edge', False),
        })
    return sides_data


def _get_outline(sides_data):
    pts = list(sides_data[0]['vertices'])
    for i in range(1, 4):
        v = sides_data[i]['vertices']
        if pts and abs(pts[-1][0] - v[0][0]) < 5 and abs(pts[-1][1] - v[0][1]) < 5:
            pts.extend(v[1:])
        else:
            pts.extend(v)
    return pts


def _centroid(pts):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return cx, cy


def _draw_piece_with_highlight(pid, highlight_si, all_sides, output_path, title=""):
    outline = _get_outline(all_sides)
    all_pts = outline
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    data_w = max_x - min_x
    data_h = max_y - min_y
    if data_w < 1: data_w = 1
    if data_h < 1: data_h = 1

    margin = max(data_w, data_h) * 0.08
    canvas_w = int(data_w + 2 * margin)
    canvas_h = int(data_h + 2 * margin) + 40

    scale = 1.0
    max_dim = 2000
    if max(canvas_w, canvas_h) > max_dim:
        scale = max_dim / max(canvas_w, canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)

    img = Image.new('RGB', (max(canvas_w, 100), max(canvas_h, 100)), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", max(10, min(20, int(18 * scale))))
        title_font = ImageFont.truetype("arial.ttf", max(12, min(22, int(20 * scale))))
        small_font = ImageFont.truetype("arial.ttf", max(8, min(14, int(12 * scale))))
    except Exception:
        font = ImageFont.load_default()
        title_font = font
        small_font = font

    draw.text((5, 3), title, fill=(0, 0, 0), font=title_font)

    def to_canvas(x, y):
        cx = (x - min_x + margin) * scale
        cy = (y - min_y + margin) * scale + 35
        return (cx, cy)

    canvas_outline = [to_canvas(x, y) for x, y in outline]
    if len(canvas_outline) >= 3:
        draw.polygon(canvas_outline, fill=(230, 230, 230), outline=(100, 100, 100))

    side_colors = [
        (200, 200, 200),
        (200, 200, 200),
        (200, 200, 200),
        (200, 200, 200),
    ]
    side_colors[highlight_si] = (255, 0, 0)

    for si in range(4):
        pts = [to_canvas(x, y) for x, y in all_sides[si]['vertices']]
        if len(pts) >= 2:
            w = max(3, int(5 * scale)) if si == highlight_si else max(1, int(2 * scale))
            draw.line(pts, fill=side_colors[si], width=w)

    for si in range(4):
        v = all_sides[si]['vertices']
        mid_idx = len(v) // 2
        mx, my = v[mid_idx]
        cx, cy = to_canvas(mx, my)
        label = f"s{si}"
        if si == highlight_si:
            label = f"**s{si}**"
        draw.text((cx, cy), label, fill=(0, 0, 0) if si != highlight_si else (255, 0, 0),
                  font=font, anchor="mm")

    cx, cy = _centroid(canvas_outline)
    draw.text((cx, cy), f"P{pid}", fill=(50, 50, 50), font=title_font, anchor="mm")

    img.save(output_path)
    print(f"  Saved: {output_path}")


def _draw_polylines_overlay(polylines, colors, labels, output_path, title="",
                            point_labels=None):
    all_pts = []
    for pl in polylines:
        all_pts.extend(pl)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    data_w = max_x - min_x
    data_h = max_y - min_y
    if data_w < 1: data_w = 1
    if data_h < 1: data_h = 1

    margin = max(data_w, data_h) * 0.1
    canvas_w = int(data_w + 2 * margin)
    canvas_h = int(data_h + 2 * margin) + 40 + len(polylines) * 22

    scale = 1.0
    max_dim = 2500
    if max(canvas_w, canvas_h) > max_dim:
        scale = max_dim / max(canvas_w, canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)

    img = Image.new('RGB', (max(canvas_w, 200), max(canvas_h, 200)), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", max(8, min(14, int(12 * scale))))
        title_font = ImageFont.truetype("arial.ttf", max(10, min(18, int(16 * scale))))
        tiny_font = ImageFont.truetype("arial.ttf", max(6, min(10, int(9 * scale))))
    except Exception:
        font = ImageFont.load_default()
        title_font = font
        tiny_font = font

    draw.text((5, 3), title, fill=(0, 0, 0), font=title_font)

    ly = 30
    for i, (color, label) in enumerate(zip(colors, labels)):
        draw.line([(10, ly + i * 20), (35, ly + i * 20)], fill=color, width=3)
        draw.text((40, ly + i * 20 - 5), label, fill=(0, 0, 0), font=font)

    offset_y = 40 + len(polylines) * 22

    def to_canvas(x, y):
        cx = (x - min_x + margin) * scale
        cy = (y - min_y + margin) * scale + offset_y
        return (cx, cy)

    for pi, (pl, color) in enumerate(zip(polylines, colors)):
        pts = [to_canvas(x, y) for x, y in pl]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=max(2, int(3 * scale)))

        for j, pt in enumerate(pts):
            if j == 0:
                r = max(3, int(5 * scale))
                draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r],
                             fill=(0, 0, 0))
            elif j == len(pts) - 1:
                r = max(3, int(5 * scale))
                draw.rectangle([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r],
                               fill=(200, 0, 0))

        if point_labels and pi < len(point_labels):
            for j, pt in enumerate(pts):
                if j % 2 == 0:
                    lbl = point_labels[pi][j] if j < len(point_labels[pi]) else str(j)
                    draw.text((pt[0] + 2, pt[1] - 8), lbl,
                              fill=color, font=tiny_font)

    img.save(output_path)
    print(f"  Saved: {output_path}")


def _draw_diff_bar_chart(values, output_path, title="", xlabel=""):
    n = len(values)
    bar_w = max(4, min(30, 1000 // n))
    chart_w = max(500, n * bar_w + 120)
    chart_h = 300

    img = Image.new('RGB', (chart_w, chart_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 11)
        title_font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    draw.text((5, 3), title, fill=(0, 0, 0), font=title_font)

    max_val = max(abs(v) for v in values) if values else 1
    if max_val < 0.001: max_val = 1

    ml, mr, mt, mb = 70, 20, 30, 40
    pw = chart_w - ml - mr
    ph = chart_h - mt - mb
    mid_y = mt + ph // 2

    draw.line([(ml, mid_y), (ml + pw, mid_y)], fill=(200, 200, 200), width=1)
    sy = (ph // 2) / max_val

    for i, v in enumerate(values):
        x = ml + int(i * pw / n)
        h = int(abs(v) * sy)
        color = (220, 50, 50) if v > 0 else (50, 50, 220)
        if v >= 0:
            draw.rectangle([x, mid_y - h, x + bar_w - 1, mid_y], fill=color)
        else:
            draw.rectangle([x, mid_y, x + bar_w - 1, mid_y + h], fill=color)

    draw.text((5, mt), f"+{max_val:.2f}", fill=(0, 0, 0), font=font)
    draw.text((5, mid_y - 6), "0", fill=(0, 0, 0), font=font)
    draw.text((5, chart_h - mb - 10), f"-{max_val:.2f}", fill=(0, 0, 0), font=font)
    if xlabel:
        draw.text((ml, chart_h - mb + 10), xlabel, fill=(100, 100, 100), font=font)

    img.save(output_path)
    print(f"  Saved: {output_path}")


def _draw_pair_assembly(pid_a, si_a, pid_b, si_b, output_path):
    sides_a = _load_raw_piece_sides(pid_a)
    sides_b = _load_raw_piece_sides(pid_b)

    src_verts = sides_b[si_b]['vertices']
    src_p1 = src_verts[0]
    src_p2 = src_verts[-1]
    src_mid = ((src_p1[0] + src_p2[0]) / 2.0, (src_p1[1] + src_p2[1]) / 2.0)
    src_theta = math.atan2(src_p2[1] - src_p1[1], src_p2[0] - src_p1[0])

    tgt_verts = sides_a[si_a]['vertices']
    tgt_p1 = tgt_verts[0]
    tgt_p2 = tgt_verts[-1]
    tgt_mid = ((tgt_p1[0] + tgt_p2[0]) / 2.0, (tgt_p1[1] + tgt_p2[1]) / 2.0)
    tgt_theta = math.atan2(tgt_p2[1] - tgt_p1[1], tgt_p2[0] - tgt_p1[0])

    cos_b = math.cos(-src_theta)
    sin_b = math.sin(-src_theta)
    cos_a = math.cos(tgt_theta)
    sin_a = math.sin(tgt_theta)

    transformed_b = []
    for side in sides_b:
        new_verts = []
        for v in side['vertices']:
            dx = v[0] - src_mid[0]
            dy = v[1] - src_mid[1]
            rx = dx * cos_b - dy * sin_b
            ry = dx * sin_b + dy * cos_b
            ry = -ry
            fx = rx * cos_a - ry * sin_a
            fy = rx * sin_a + ry * cos_a
            new_verts.append((fx + tgt_mid[0], fy + tgt_mid[1]))
        transformed_b.append({'vertices': new_verts, 'is_edge': side['is_edge']})

    all_pts = []
    for s in sides_a:
        all_pts.extend(s['vertices'])
    for s in transformed_b:
        all_pts.extend(s['vertices'])

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dw = max_x - min_x
    dh = max_y - min_y
    if dw < 1: dw = 1
    if dh < 1: dh = 1

    margin = max(dw, dh) * 0.1
    cw = int(dw + 2 * margin)
    ch = int(dh + 2 * margin) + 50

    scale = 1.0
    if max(cw, ch) > 2500:
        scale = 2500 / max(cw, ch)
        cw = int(cw * scale)
        ch = int(ch * scale)

    img = Image.new('RGB', (max(cw, 100), max(ch, 100)), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", max(10, min(18, int(16 * scale))))
        title_font = ImageFont.truetype("arial.ttf", max(12, min(20, int(18 * scale))))
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    draw.text((5, 3),
              f"Pair Assembly: P{pid_a}[s{si_a}] <-> P{pid_b}[s{si_b}]  (same as pair_XXX.png)",
              fill=(0, 0, 0), font=title_font)

    def to_canvas(x, y):
        return ((x - min_x + margin) * scale, (y - min_y + margin) * scale + 45)

    outline_a = _get_outline(sides_a)
    ca = [to_canvas(x, y) for x, y in outline_a]
    if len(ca) >= 3:
        draw.polygon(ca, fill=(200, 220, 255), outline=(0, 80, 200))
    c = _centroid(ca)
    draw.text(c, str(pid_a), fill=(0, 0, 150), font=font, anchor="mm")

    outline_b = _get_outline(transformed_b)
    cb = [to_canvas(x, y) for x, y in outline_b]
    if len(cb) >= 3:
        draw.polygon(cb, fill=(255, 220, 200), outline=(200, 80, 0))
    c = _centroid(cb)
    draw.text(c, str(pid_b), fill=(150, 0, 0), font=font, anchor="mm")

    match_a = [to_canvas(x, y) for x, y in sides_a[si_a]['vertices']]
    match_b = [to_canvas(x, y) for x, y in transformed_b[si_b]['vertices']]
    if len(match_a) >= 2:
        draw.line(match_a, fill=(255, 165, 0), width=max(2, int(4 * scale)))
    if len(match_b) >= 2:
        draw.line(match_b, fill=(0, 200, 0), width=max(2, int(4 * scale)))

    img.save(output_path)
    print(f"  Saved: {output_path}")


def diagnose_pair(pid_a, si_a, pid_b, si_b):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" DIAGNOSE: Piece {pid_a}[side{si_a}] <-> Piece {pid_b}[side{si_b}]")
    print(f"{'='*70}\n")

    ps = pieces.Piece.load_all(DEDUPED_DIR, resample=True)
    ps_raw = pieces.Piece.load_all(DEDUPED_DIR, resample=False)

    if pid_a not in ps or pid_b not in ps:
        print(f"ERROR: piece not found.")
        return

    side_a = ps[pid_a].sides[si_a]
    side_b = ps[pid_b].sides[si_b]
    side_a_raw = ps_raw[pid_a].sides[si_a]
    side_b_raw = ps_raw[pid_b].sides[si_b]

    # ── A: Draw full piece outlines with highlighted matching side ──
    print("--- A: Full piece outlines ---")
    raw_a = _load_raw_piece_sides(pid_a)
    raw_b = _load_raw_piece_sides(pid_b)

    _draw_piece_with_highlight(pid_a, si_a, raw_a,
                               os.path.join(OUTPUT_DIR, 'A_piece_full.png'),
                               title=f"Piece {pid_a} (side {si_a} = RED)")
    _draw_piece_with_highlight(pid_b, si_b, raw_b,
                               os.path.join(OUTPUT_DIR, 'B_piece_full.png'),
                               title=f"Piece {pid_b} (side {si_b} = RED)")

    # ── B: Pair assembly image (same logic as show_connectivity.py) ──
    print("\n--- B: Pair assembly (same as pair_XXX.png) ---")
    _draw_pair_assembly(pid_a, si_a, pid_b, si_b,
                        os.path.join(OUTPUT_DIR, 'B_pair_assembly.png'))

    # ── 1: Basic Info ──
    print(f"\n--- 1: Basic Info ---")
    print(f"Side A (Piece {pid_a}[{si_a}]):")
    print(f"  is_edge = {side_a.is_edge}")
    print(f"  Original p1={tuple(side_a_raw.p1)}, p2={tuple(side_a_raw.p2)}")
    print(f"  Straight-line length = {side_a_raw.length:.2f}")
    print(f"  Angle = {side_a_raw.angle * 180/math.pi:.2f} deg")
    print(f"  Polyline length (v_length) = {side_a.v_length:.2f}")
    print(f"  Original vertices count = {len(side_a_raw.vertices)}")
    print()
    print(f"Side B (Piece {pid_b}[{si_b}]):")
    print(f"  is_edge = {side_b.is_edge}")
    print(f"  Original p1={tuple(side_b_raw.p1)}, p2={tuple(side_b_raw.p2)}")
    print(f"  Straight-line length = {side_b_raw.length:.2f}")
    print(f"  Angle = {side_b_raw.angle * 180/math.pi:.2f} deg")
    print(f"  Polyline length (v_length) = {side_b.v_length:.2f}")
    print(f"  Original vertices count = {len(side_b_raw.vertices)}")

    # ── 2: Length check ──
    d_scale = 1.0 - (side_a.length / side_b.length)
    print(f"\n--- 2: Length check ---")
    print(f"  d_scale = {d_scale:.6f}, |d_scale| = {abs(d_scale):.6f}")
    print(f"  threshold = {sides.SIDE_MAX_LENGTH_DISCREPANCY}")
    print(f"  PASS = {abs(d_scale) <= sides.SIDE_MAX_LENGTH_DISCREPANCY}")
    if abs(d_scale) > sides.SIDE_MAX_LENGTH_DISCREPANCY:
        print("  >>> SKIP")
        return

    # ── 3: Original raw side shapes (separate, NOT overlaid) ──
    print(f"\n--- 3: Original raw side shapes ---")
    orig_a = [(float(x), float(y)) for x, y in side_a_raw.vertices]
    orig_b = [(float(x), float(y)) for x, y in side_b_raw.vertices]

    _draw_polylines_overlay(
        [orig_a], [(0, 100, 220)],
        [f"A: Piece {pid_a}[{si_a}] ({len(orig_a)} pts)"],
        os.path.join(OUTPUT_DIR, '3a_side_A_raw.png'),
        title=f"Side A raw shape (Piece {pid_a}[{si_a}])"
    )
    _draw_polylines_overlay(
        [orig_b], [(220, 80, 0)],
        [f"B: Piece {pid_b}[{si_b}] ({len(orig_b)} pts)"],
        os.path.join(OUTPUT_DIR, '3b_side_B_raw.png'),
        title=f"Side B raw shape (Piece {pid_b}[{si_b}])"
    )

    # ── 4: After resample (before rotation) ──
    print(f"\n--- 4: After resample (before rotation) ---")
    resamp_a, len_a = util.resample_polyline(orig_a, n=sides.SIDE_RESAMPLE_VERTEX_COUNT)
    resamp_b, len_b = util.resample_polyline(orig_b, n=sides.SIDE_RESAMPLE_VERTEX_COUNT)
    print(f"  A: {len(resamp_a)} pts, length={len_a:.2f}")
    print(f"  B: {len(resamp_b)} pts, length={len_b:.2f}")

    idx_labels_a = [str(i) for i in range(len(resamp_a))]
    idx_labels_b = [str(i) for i in range(len(resamp_b))]
    _draw_polylines_overlay(
        [resamp_a], [(0, 100, 220)],
        [f"A resampled ({len(resamp_a)} pts)"],
        os.path.join(OUTPUT_DIR, '4a_side_A_resampled.png'),
        title=f"Side A after resample (27 pts)",
        point_labels=[idx_labels_a]
    )
    _draw_polylines_overlay(
        [resamp_b], [(220, 80, 0)],
        [f"B resampled ({len(resamp_b)} pts)"],
        os.path.join(OUTPUT_DIR, '4b_side_B_resampled.png'),
        title=f"Side B after resample (27 pts)",
        point_labels=[idx_labels_b]
    )

    # ── 5: After rotation - show A.vertices and B.vertices (both at angle=0) ──
    print(f"\n--- 5: After rotation ---")
    angle_a_raw = side_a_raw.angle
    angle_b_raw = side_b_raw.angle
    print(f"  A original angle = {angle_a_raw * 180/math.pi:.2f} deg")
    print(f"  B original angle = {angle_b_raw * 180/math.pi:.2f} deg")

    verts_a = side_a.vertices
    verts_b_normal = side_b.vertices
    verts_b_flipped = side_b.vertices_flipped

    list_a = [(float(x), float(y)) for x, y in verts_a]
    list_b_normal = [(float(x), float(y)) for x, y in verts_b_normal]
    list_b_flipped = [(float(x), float(y)) for x, y in verts_b_flipped]

    print(f"  A.vertices: p1={list_a[0]}, p2={list_a[-1]}")
    print(f"  B.vertices: p1={list_b_normal[0]}, p2={list_b_normal[-1]}")
    print(f"  B.vertices_flipped: p1={list_b_flipped[0]}, p2={list_b_flipped[-1]}")

    _draw_polylines_overlay(
        [list_a, list_b_normal],
        [(0, 100, 220), (220, 80, 0)],
        [f"A.vertices (rotated to 0)",
         f"B.vertices (rotated to 0)"],
        os.path.join(OUTPUT_DIR, '5a_both_at_0deg.png'),
        title=f"Both sides rotated to 0 deg (NO flip)"
    )

    # ── 6: THE KEY COMPARISON: A.vertices vs B.vertices_flipped ──
    print(f"\n--- 6: KEY COMPARISON: A.vertices vs B.vertices_flipped ---")
    _draw_polylines_overlay(
        [list_a, list_b_flipped],
        [(0, 100, 220), (220, 80, 0)],
        [f"A.vertices",
         f"B.vertices_flipped (rot to PI + reversed)"],
        os.path.join(OUTPUT_DIR, '6_actual_comparison.png'),
        title=f"ACTUAL comparison used by error_when_fit_with()"
    )

    # ── 7: How vertices_flipped is made (step by step) ──
    print(f"\n--- 7: How B.vertices_flipped is constructed ---")
    resamp_b_arr = np.array(resamp_b)

    o = resamp_b_arr[0]
    translated_b = resamp_b_arr - o
    angle_diff_b = math.pi - angle_b_raw
    rotated_b = []
    for v in translated_b:
        rx = v[0] * math.cos(angle_diff_b) - v[1] * math.sin(angle_diff_b)
        ry = v[0] * math.sin(angle_diff_b) + v[1] * math.cos(angle_diff_b)
        rotated_b.append((float(rx), float(ry)))

    min_x_rb = min(v[0] for v in rotated_b)
    shifted_b = [(v[0] - min_x_rb, v[1]) for v in rotated_b]

    reversed_b = list(reversed(shifted_b))

    print(f"  Step7a: B resampled (before any rotation)")
    print(f"  Step7b: B translated to origin")
    print(f"  Step7c: B rotated by (PI - {angle_b_raw*180/math.pi:.1f}deg) = {(math.pi-angle_b_raw)*180/math.pi:.1f}deg")
    print(f"  Step7d: B shifted so min_x=0")
    print(f"  Step7e: B reversed (list[::-1])")
    print(f"  Result p1={reversed_b[0]}, p2={reversed_b[-1]}")

    _draw_polylines_overlay(
        [resamp_b, shifted_b, reversed_b],
        [(180, 180, 180), (220, 80, 0), (0, 150, 0)],
        [f"B original resampled",
         f"B after rot to PI + shift",
         f"B after reverse (= vertices_flipped)"],
        os.path.join(OUTPUT_DIR, '7_flipped_construction.png'),
        title=f"How B.vertices_flipped is made"
    )

    # ── 8: Error calculation ──
    print(f"\n--- 8: Error calculation ---")
    p1 = np.array(verts_a)
    p2 = np.array(verts_b_flipped)

    diff = np.abs(p1 - p2)
    raw_err = float(np.sum(diff))
    mean_shift = np.sum(diff - p1 + p2, axis=0) / len(p1)
    ex, ey = float(mean_shift[0]), float(mean_shift[1])

    print(f"  Raw error = {raw_err:.2f}")
    print(f"  Mean shift: ex={ex:.2f}, ey={ey:.2f}")

    ey_c = max(-5.0, min(5.0, ey))
    p1_shifted = np.array([(x - ex, y - ey_c) for x, y in p1])
    diff_s = np.abs(p1_shifted - p2)
    raw_err_s = float(np.sum(diff_s))

    final_err = min(raw_err, raw_err_s) / side_b.v_length

    print(f"  Shifted error = {raw_err_s:.2f}")
    print(f"  min(raw, shifted) = {min(raw_err, raw_err_s):.2f}")
    print(f"  Divided by v_length({side_b.v_length:.2f}) = {final_err:.6f}")
    print(f"  Threshold = {sides.SIDE_MAX_ERROR_TO_MATCH}")
    print(f"  MATCH = {final_err <= sides.SIDE_MAX_ERROR_TO_MATCH}")

    list_a_shifted = [(float(x), float(y)) for x, y in p1_shifted]
    _draw_polylines_overlay(
        [list_a_shifted, list_b_flipped],
        [(0, 100, 220), (220, 80, 0)],
        [f"A shifted (dx={-ex:.1f}, dy={-ey_c:.1f})",
         f"B.vertices_flipped"],
        os.path.join(OUTPUT_DIR, '8_shifted_overlay.png'),
        title=f"After shift compensation (error={final_err:.4f})"
    )

    # ── 9: Per-point X and Y differences ──
    dx_orig = [float(p1[i][0] - p2[i][0]) for i in range(len(p1))]
    dy_orig = [float(p1[i][1] - p2[i][1]) for i in range(len(p1))]
    dx_shift = [float(p1_shifted[i][0] - p2[i][0]) for i in range(len(p1_shifted))]
    dy_shift = [float(p1_shifted[i][1] - p2[i][1]) for i in range(len(p1_shifted))]

    _draw_diff_bar_chart(dx_orig, os.path.join(OUTPUT_DIR, '9a_dx_before_shift.png'),
                         title="X-diff per point (before shift)")
    _draw_diff_bar_chart(dy_orig, os.path.join(OUTPUT_DIR, '9b_dy_before_shift.png'),
                         title="Y-diff per point (before shift)")
    _draw_diff_bar_chart(dx_shift, os.path.join(OUTPUT_DIR, '9c_dx_after_shift.png'),
                         title="X-diff per point (after shift)")
    _draw_diff_bar_chart(dy_shift, os.path.join(OUTPUT_DIR, '9d_dy_after_shift.png'),
                         title="Y-diff per point (after shift)")

    # ── 10: Without flip for comparison ──
    p2_nf = np.array(verts_b_normal)
    diff_nf = np.abs(p1 - p2_nf)
    raw_nf = float(np.sum(diff_nf))
    ms_nf = np.sum(diff_nf - p1 + p2_nf, axis=0) / len(p1)
    ex_nf, ey_nf = float(ms_nf[0]), max(-5.0, min(5.0, float(ms_nf[1])))
    p1s_nf = np.array([(x - ex_nf, y - ey_nf) for x, y in p1])
    raw_s_nf = float(np.sum(np.abs(p1s_nf - p2_nf)))
    err_nf = min(raw_nf, raw_s_nf) / side_b.v_length

    print(f"\n--- 10: Without flip ---")
    print(f"  Error without flip = {err_nf:.6f}")
    print(f"  Error with flip = {final_err:.6f}")

    _draw_polylines_overlay(
        [list_a, list_b_normal],
        [(0, 100, 220), (220, 80, 0)],
        [f"A.vertices",
         f"B.vertices (NO flip)"],
        os.path.join(OUTPUT_DIR, '10_noflip.png'),
        title=f"Without flip (error={err_nf:.4f})"
    )

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f"  Piece {pid_a}[{si_a}] <-> Piece {pid_b}[{si_b}]")
    print(f"  Side A: angle={angle_a_raw*180/math.pi:.1f} deg, len={side_a.length:.0f}, v_len={side_a.v_length:.0f}")
    print(f"  Side B: angle={angle_b_raw*180/math.pi:.1f} deg, len={side_b.length:.0f}, v_len={side_b.v_length:.0f}")
    print(f"  Length check: |d_scale|={abs(d_scale):.4f}")
    print(f"  Error (with flip):    {final_err:.6f}  {'MATCH' if final_err <= sides.SIDE_MAX_ERROR_TO_MATCH else 'NO MATCH'}")
    print(f"  Error (without flip): {err_nf:.6f}")
    print(f"  Threshold: {sides.SIDE_MAX_ERROR_TO_MATCH}")
    print(f"  Images saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    diagnose_pair(1, 0, 70, 0)
