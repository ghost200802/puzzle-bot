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


def _centroid(pts):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return cx, cy


def diagnose_alignment(pid_a, si_a, pid_b, si_b):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_a = _load_raw_piece_sides(pid_a)
    raw_b = _load_raw_piece_sides(pid_b)

    side_a_verts = raw_a[si_a]['vertices']
    side_b_verts = raw_b[si_b]['vertices']

    a_p1 = side_a_verts[0]
    a_p2 = side_a_verts[-1]
    b_p1 = side_b_verts[0]
    b_p2 = side_b_verts[-1]

    a_mid = ((a_p1[0] + a_p2[0]) / 2.0, (a_p1[1] + a_p2[1]) / 2.0)
    b_mid = ((b_p1[0] + b_p2[0]) / 2.0, (b_p1[1] + b_p2[1]) / 2.0)
    a_theta = math.atan2(a_p2[1] - a_p1[1], a_p2[0] - a_p1[0])
    b_theta = math.atan2(b_p2[1] - b_p1[1], b_p2[0] - b_p1[0])

    print(f"=== ORIGINAL COORDINATES ===")
    print(f"Side A (P{pid_a}[{si_a}]): p1={a_p1}, p2={a_p2}")
    print(f"  midpoint={a_mid}")
    print(f"  angle={a_theta*180/math.pi:.2f} deg")
    print(f"  straight-line length={math.sqrt((a_p2[0]-a_p1[0])**2+(a_p2[1]-a_p1[1])**2):.2f}")
    print(f"Side B (P{pid_b}[{si_b}]): p1={b_p1}, p2={b_p2}")
    print(f"  midpoint={b_mid}")
    print(f"  angle={b_theta*180/math.pi:.2f} deg")
    print(f"  straight-line length={math.sqrt((b_p2[0]-b_p1[0])**2+(b_p2[1]-b_p1[1])**2):.2f}")

    # ── How _transform_piece works ──
    print(f"\n=== _transform_piece TRANSFORMATION (show_connectivity.py) ===")
    print(f"Goal: transform piece B so that B's side {si_b} aligns with A's side {si_a}")
    print(f"Steps:")
    print(f"  1. For each vertex of B: dx = v - B_mid, dy = v - B_mid")
    print(f"  2. Rotate by -B_angle (cancel B's direction)")
    print(f"  3. Mirror Y: ry = -ry")
    print(f"  4. Rotate by A_angle (apply A's direction)")
    print(f"  5. Translate to A_mid")

    cos_b = math.cos(-b_theta)
    sin_b = math.sin(-b_theta)
    cos_a = math.cos(a_theta)
    sin_a = math.sin(a_theta)

    transformed_b_sides = []
    for si in range(4):
        new_verts = []
        for v in raw_b[si]['vertices']:
            dx = v[0] - b_mid[0]
            dy = v[1] - b_mid[1]
            rx = dx * cos_b - dy * sin_b
            ry = dx * sin_b + dy * cos_b
            ry = -ry
            fx = rx * cos_a - ry * sin_a
            fy = rx * sin_a + ry * cos_a
            new_verts.append((fx + a_mid[0], fy + a_mid[1]))
        transformed_b_sides.append({'vertices': new_verts, 'is_edge': raw_b[si]['is_edge']})

    transformed_side_b = transformed_b_sides[si_b]

    print(f"\n=== ALIGNMENT CHECK ===")
    print(f"Side A vertices (first 5 and last 5):")
    for i in [0, 1, 2, -3, -2, -1]:
        idx = i if i >= 0 else len(side_a_verts) + i
        print(f"  [{idx}] {side_a_verts[idx]}")

    print(f"\nTransformed Side B vertices (first 5 and last 5):")
    for i in [0, 1, 2, -3, -2, -1]:
        idx = i if i >= 0 else len(transformed_side_b['vertices']) + i
        print(f"  [{idx}] {transformed_side_b['vertices'][idx]}")

    print(f"\n=== ENDPOINT COMPARISON ===")
    print(f"Side A: p1={side_a_verts[0]}  p2={side_a_verts[-1]}")
    print(f"Trans B: p1={transformed_side_b['vertices'][0]}  p2={transformed_side_b['vertices'][-1]}")

    a_first = side_a_verts[0]
    a_last = side_a_verts[-1]
    b_first = transformed_side_b['vertices'][0]
    b_last = transformed_side_b['vertices'][-1]

    gap_first = (b_first[0] - a_first[0], b_first[1] - a_first[1])
    gap_last = (b_last[0] - a_last[0], b_last[1] - a_last[1])
    print(f"\nGap at p1 (B-A): dx={gap_first[0]:.1f}, dy={gap_first[1]:.1f}, dist={math.sqrt(gap_first[0]**2+gap_first[1]**2):.1f}")
    print(f"Gap at p2 (B-A): dx={gap_last[0]:.1f}, dy={gap_last[1]:.1f}, dist={math.sqrt(gap_last[0]**2+gap_last[1]**2):.1f}")

    # Sample a few midpoints too
    print(f"\n=== MIDPOINT COMPARISON ===")
    for frac in [0.25, 0.5, 0.75]:
        a_idx = int(frac * (len(side_a_verts) - 1))
        b_idx = int(frac * (len(transformed_side_b['vertices']) - 1))
        a_pt = side_a_verts[a_idx]
        b_pt = transformed_side_b['vertices'][b_idx]
        gap = (b_pt[0] - a_pt[0], b_pt[1] - a_pt[1])
        print(f"  {int(frac*100)}%: A={a_pt}  B={b_pt}  gap=({gap[0]:.1f}, {gap[1]:.1f})")

    # ── The REAL question: what does the error_when_fit_with actually compute? ──
    print(f"\n=== WHAT error_when_fit_with ACTUALLY COMPARES ===")
    print(f"It does NOT use original coordinates.")
    print(f"It uses resampled+rotated+flipped NORMALIZED curves.")
    print(f"Both curves are: 27 points, starting at (0,0), ending at (~757/772, 0)")
    print(f"Error = sum(|A[i] - B_flipped[i]|) / polyline_length")
    print(f"This is a SHAPE comparison, not a POSITION comparison.")
    print(f"Two curves can have similar SHAPE but different actual placement.")

    # ── Draw detailed alignment image with coordinate labels ──
    all_pts = list(side_a_verts) + list(transformed_side_b['vertices'])
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dw = max_x - min_x
    dh = max_y - min_y
    if dw < 1: dw = 1
    if dh < 1: dh = 1

    margin = max(dw, dh) * 0.15
    cw = int(dw + 2 * margin)
    ch = int(dh + 2 * margin) + 60

    scale = 1.0
    if max(cw, ch) > 3000:
        scale = 3000 / max(cw, ch)
        cw = int(cw * scale)
        ch = int(ch * scale)

    img = Image.new('RGB', (max(cw, 200), max(ch, 200)), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", max(10, min(16, int(14 * scale))))
        title_font = ImageFont.truetype("arial.ttf", max(12, min(20, int(18 * scale))))
        tiny_font = ImageFont.truetype("arial.ttf", max(7, min(11, int(9 * scale))))
    except Exception:
        font = ImageFont.load_default()
        title_font = font
        tiny_font = font

    draw.text((5, 3),
              f"Alignment Debug: P{pid_a}[s{si_a}] vs P{pid_b}[s{si_b}] (transformed)",
              fill=(0, 0, 0), font=title_font)

    offset_y = 45

    def to_canvas(x, y):
        return ((x - min_x + margin) * scale, (y - min_y + margin) * scale + offset_y)

    draw.line([to_canvas(*side_a_verts[0]), to_canvas(*side_a_verts[-1])],
              fill=(200, 200, 200), width=1)
    draw.line([to_canvas(*transformed_side_b['vertices'][0]),
               to_canvas(*transformed_side_b['vertices'][-1])],
              fill=(200, 200, 200), width=1)

    pts_a = [to_canvas(x, y) for x, y in side_a_verts]
    if len(pts_a) >= 2:
        draw.line(pts_a, fill=(0, 100, 220), width=max(2, int(3 * scale)))

    pts_b = [to_canvas(x, y) for x, y in transformed_side_b['vertices']]
    if len(pts_b) >= 2:
        draw.line(pts_b, fill=(220, 80, 0), width=max(2, int(3 * scale)))

    for j in [0, len(side_a_verts) // 4, len(side_a_verts) // 2,
              3 * len(side_a_verts) // 4, len(side_a_verts) - 1]:
        pt = to_canvas(*side_a_verts[j])
        r = max(3, int(4 * scale))
        draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=(0, 100, 220))
        lbl = f"({side_a_verts[j][0]:.0f},{side_a_verts[j][1]:.0f})"
        draw.text((pt[0] + 5, pt[1] - 8), lbl, fill=(0, 80, 180), font=tiny_font)

    for j in [0, len(transformed_side_b['vertices']) // 4,
              len(transformed_side_b['vertices']) // 2,
              3 * len(transformed_side_b['vertices']) // 4,
              len(transformed_side_b['vertices']) - 1]:
        pt = to_canvas(*transformed_side_b['vertices'][j])
        r = max(3, int(4 * scale))
        draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=(220, 80, 0))
        lbl = f"({transformed_side_b['vertices'][j][0]:.0f},{transformed_side_b['vertices'][j][1]:.0f})"
        draw.text((pt[0] + 5, pt[1] + 3), lbl, fill=(200, 60, 0), font=tiny_font)

    draw.text((10, ch - 30),
              f"Blue=P{pid_a}[s{si_a}]  Orange=P{pid_b}[s{si_b}](transformed)",
              fill=(0, 0, 0), font=font)

    out_path = os.path.join(OUTPUT_DIR, 'alignment_debug.png')
    img.save(out_path)
    print(f"\n  Saved: {out_path}")

    # ── Also check: what does "ideal" alignment look like? ──
    # If we force the endpoints to match, what's the residual?
    print(f"\n=== IF WE FORCE ENDPOINTS TO MATCH ===")
    offset = (a_p1[0] - b_first[0], a_p1[1] - b_first[1])
    print(f"  Required offset to align p1: ({offset[0]:.1f}, {offset[1]:.1f})")
    forced_b_p2 = (b_last[0] + offset[0], b_last[1] + offset[1])
    p2_gap = (forced_b_p2[0] - a_last[0], forced_b_p2[1] - a_last[1])
    print(f"  After forcing p1, gap at p2: ({p2_gap[0]:.1f}, {p2_gap[1]:.1f})")

    # Try aligning midpoints instead
    b_mid_trans = _centroid(transformed_side_b['vertices'])
    mid_offset = (a_mid[0] - b_mid_trans[0], a_mid[1] - b_mid_trans[1])
    print(f"\n=== IF WE ALIGN MIDPOINTS ===")
    print(f"  A midpoint: {a_mid}")
    print(f"  B transformed midpoint: ({b_mid_trans[0]:.1f}, {b_mid_trans[1]:.1f})")
    print(f"  Offset: ({mid_offset[0]:.1f}, {mid_offset[1]:.1f})")
    shifted_b_first = (b_first[0] + mid_offset[0], b_first[1] + mid_offset[1])
    shifted_b_last = (b_last[0] + mid_offset[0], b_last[1] + mid_offset[1])
    gap_f = (shifted_b_first[0] - a_first[0], shifted_b_first[1] - a_first[1])
    gap_l = (shifted_b_last[0] - a_last[0], shifted_b_last[1] - a_last[1])
    print(f"  After aligning midpoints:")
    print(f"    Gap at p1: ({gap_f[0]:.1f}, {gap_f[1]:.1f})")
    print(f"    Gap at p2: ({gap_l[0]:.1f}, {gap_l[1]:.1f})")

    # The core issue: midpoint alignment doesn't account for different lengths
    a_len = math.sqrt((a_p2[0]-a_p1[0])**2 + (a_p2[1]-a_p1[1])**2)
    b_len = math.sqrt((b_p2[0]-b_p1[0])**2 + (b_p2[1]-b_p1[1])**2)
    print(f"\n=== ROOT CAUSE ===")
    print(f"  Side A straight length: {a_len:.1f}")
    print(f"  Side B straight length: {b_len:.1f}")
    print(f"  Difference: {abs(a_len - b_len):.1f} pixels ({abs(a_len-b_len)/max(a_len,b_len)*100:.1f}%)")
    print(f"  _transform_piece aligns MIDPOINTS, so endpoints will be off by ~{abs(a_len-b_len)/2:.1f} px each side")
    print(f"  But the actual gap may be larger because of curve shape differences")


if __name__ == '__main__':
    diagnose_alignment(1, 0, 70, 0)
