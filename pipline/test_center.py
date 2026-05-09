import os
import sys
import json
import math
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common import pieces, sides, util
from PIL import Image, ImageDraw, ImageFont

from config import get_output_dir, get_deduped_path

OUTPUT_DIR = os.path.join(get_output_dir(), 'check', 'match_debug')
DEDUPED_DIR = get_deduped_path()


def _signed_distance_to_line(point, line_p1, line_p2):
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    cross = dx * (point[1] - line_p1[1]) - dy * (point[0] - line_p1[0])
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.001:
        return 0
    return cross / length


def analyze_center_constraint(pid_a, si_a, pid_b, si_b):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ps_raw = pieces.Piece.load_all(DEDUPED_DIR, resample=False)

    side_a = ps_raw[pid_a].sides[si_a]
    side_b = ps_raw[pid_b].sides[si_b]

    a_p1 = tuple(side_a.p1)
    a_p2 = tuple(side_a.p2)
    b_p1 = tuple(side_b.p1)
    b_p2 = tuple(side_b.p2)

    a_center = tuple(side_a.piece_center)
    b_center = tuple(side_b.piece_center)

    a_angle = math.atan2(a_p2[1] - a_p1[1], a_p2[0] - a_p1[0])
    b_angle = math.atan2(b_p2[1] - b_p1[1], b_p2[0] - b_p1[0])

    print(f"=== CENTER CONSTRAINT ANALYSIS ===")
    print(f"P{pid_a}[s{si_a}] vs P{pid_b}[s{si_b}]\n")

    print(f"Side A: p1={a_p1}, p2={a_p2}")
    print(f"  piece_center = {a_center}")
    print(f"  angle = {a_angle*180/math.pi:.2f} deg")

    print(f"\nSide B: p1={b_p1}, p2={b_p2}")
    print(f"  piece_center = {b_center}")
    print(f"  angle = {b_angle*180/math.pi:.2f} deg")

    sd_a = _signed_distance_to_line(a_center, a_p1, a_p2)
    sd_b = _signed_distance_to_line(b_center, b_p1, b_p2)

    print(f"\n--- Signed distance of piece_center to edge line ---")
    print(f"  A center to A edge: {sd_a:.2f}  ({'LEFT of p1->p2' if sd_a > 0 else 'RIGHT of p1->p2'})")
    print(f"  B center to B edge: {sd_b:.2f}  ({'LEFT of p1->p2' if sd_b > 0 else 'RIGHT of p1->p2'})")

    print(f"\n--- Constraint: For a valid match, centers must be on OPPOSITE sides ---")
    print(f"  Same sign = SAME side = INVALID (pieces would overlap)")
    print(f"  Different sign = OPPOSITE sides = VALID")

    if sd_a * sd_b > 0:
        print(f"\n  *** VIOLATION! Both centers on SAME side ({'positive' if sd_a > 0 else 'negative'}) ***")
        print(f"  These pieces CANNOT match on these sides!")
        print(f"  The current algorithm doesn't check this constraint.")
    else:
        print(f"\n  Centers on OPPOSITE sides - geometrically possible match")

    print(f"\n--- Visual explanation ---")
    print(f"  Side A goes from p1 to p2.")
    print(f"  Looking along p1->p2 direction:")
    print(f"    A center is on the {'LEFT' if sd_a > 0 else 'RIGHT'}")
    print(f"  For B to mate with A, B must be on the OTHER side of the edge.")
    print(f"  But B's center is on the {'LEFT' if sd_b > 0 else 'RIGHT'} of B's own edge.")
    print(f"  After aligning B's edge to A's edge (with flip),")
    print(f"  B's center would be on the {'SAME' if sd_a * sd_b > 0 else 'OPPOSITE'} side as A's center.")

    _draw_center_visual(
        pid_a, si_a, side_a, a_center, a_p1, a_p2, sd_a,
        pid_b, si_b, side_b, b_center, b_p1, b_p2, sd_b,
        os.path.join(OUTPUT_DIR, 'center_constraint.png')
    )


def _draw_center_visual(pid_a, si_a, side_a, a_center, a_p1, a_p2, sd_a,
                         pid_b, si_b, side_b, b_center, b_p1, b_p2, sd_b,
                         output_path):
    a_verts = [(float(x), float(y)) for x, y in side_a.vertices]
    b_verts = [(float(x), float(y)) for x, y in side_b.vertices]

    a_angle = math.atan2(a_p2[1] - a_p1[1], a_p2[0] - a_p1[0])

    def normalize_side(verts, p1, center, angle):
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        result = []
        for v in verts:
            dx = v[0] - p1[0]
            dy = v[1] - p1[1]
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            result.append((rx, ry))
        cdx = center[0] - p1[0]
        cdy = center[1] - p1[1]
        cx = cdx * cos_a - cdy * sin_a
        cy = cdx * sin_a + cdy * cos_a
        return result, (cx, cy)

    norm_a, center_a = normalize_side(a_verts, a_p1, a_center, a_angle)
    norm_b, center_b = normalize_side(b_verts, b_p1, b_center, a_angle)

    flipped_b = [(x, -y) for x, y in norm_b]
    center_b_flipped = (center_b[0], -center_b[1])

    all_pts = list(norm_a) + list(flipped_b) + [center_a, center_b_flipped]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dw = max_x - min_x
    dh = max_y - min_y
    if dw < 1: dw = 1
    if dh < 1: dh = 1

    margin = max(dw, dh) * 0.12
    cw = int(dw + 2 * margin)
    ch = int(dh + 2 * margin) + 80

    scale = 1.0
    if max(cw, ch) > 2500:
        scale = 2500 / max(cw, ch)
        cw = int(cw * scale)
        ch = int(ch * scale)

    img = Image.new('RGB', (max(cw, 200), max(ch, 200)), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", max(10, min(16, int(14 * scale))))
        title_font = ImageFont.truetype("arial.ttf", max(12, min(20, int(18 * scale))))
        small_font = ImageFont.truetype("arial.ttf", max(8, min(12, int(10 * scale))))
    except Exception:
        font = ImageFont.load_default()
        title_font = font
        small_font = font

    valid = sd_a * sd_b < 0
    status = "VALID (opposite sides)" if valid else "INVALID (same side!)"
    color_status = (0, 150, 0) if valid else (220, 0, 0)

    draw.text((5, 3), f"Center Constraint: P{pid_a}[s{si_a}] vs P{pid_b}[s{si_b}]",
              fill=(0, 0, 0), font=title_font)
    draw.text((5, 25), status, fill=color_status, font=font)

    offset_y = 55

    def to_canvas(x, y):
        return ((x - min_x + margin) * scale, (y - min_y + margin) * scale + offset_y)

    edge_y = to_canvas(0, 0)[1]
    draw.line([(0, edge_y), (cw, edge_y)], fill=(200, 200, 200), width=1)
    draw.text((cw - 100, edge_y - 15), "edge line (y=0)", fill=(200, 200, 200), font=small_font)

    pts_a = [to_canvas(x, y) for x, y in norm_a]
    if len(pts_a) >= 2:
        draw.line(pts_a, fill=(0, 100, 220), width=max(2, int(3 * scale)))

    pts_bf = [to_canvas(x, y) for x, y in flipped_b]
    if len(pts_bf) >= 2:
        draw.line(pts_bf, fill=(220, 80, 0), width=max(2, int(3 * scale)))

    ca = to_canvas(*center_a)
    r = max(5, int(8 * scale))
    draw.ellipse([ca[0]-r, ca[1]-r, ca[0]+r, ca[1]+r], fill=(0, 100, 220))
    draw.text((ca[0]+r+3, ca[1]-8), f"A center (y={center_a[1]:.0f})",
              fill=(0, 80, 180), font=font)

    cb = to_canvas(*center_b_flipped)
    draw.ellipse([cb[0]-r, cb[1]-r, cb[0]+r, cb[1]+r], fill=(220, 80, 0))
    draw.text((cb[0]+r+3, cb[1]-8), f"B center flipped (y={center_b_flipped[1]:.0f})",
              fill=(200, 60, 0), font=font)

    draw.text((10, ch - 22),
              f"Blue=A  Orange=B(flipped)  Same side={'YES -> INVALID' if center_a[1]*center_b_flipped[1]>0 else 'NO -> OK'}",
              fill=(0, 0, 0), font=font)

    img.save(output_path)
    print(f"  Saved: {output_path}")


if __name__ == '__main__':
    analyze_center_constraint(1, 0, 70, 0)
