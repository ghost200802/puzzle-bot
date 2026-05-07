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


def _draw_polylines_overlay(polylines, colors, labels, output_path, title=""):
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
    except Exception:
        font = ImageFont.load_default()
        title_font = font

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
        if len(pts) >= 1:
            r = max(3, int(5 * scale))
            draw.ellipse([pts[0][0]-r, pts[0][1]-r, pts[0][0]+r, pts[0][1]+r], fill=(0,0,0))
        if len(pts) >= 2:
            r = max(3, int(5 * scale))
            draw.rectangle([pts[-1][0]-r, pts[-1][1]-r, pts[-1][0]+r, pts[-1][1]+r], fill=(200,0,0))

    img.save(output_path)
    print(f"  Saved: {output_path}")


def analyze_flip(pid_a, si_a, pid_b, si_b):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ps = pieces.Piece.load_all(DEDUPED_DIR, resample=True)

    side_a = ps[pid_a].sides[si_a]
    side_b = ps[pid_b].sides[si_b]

    verts_a = [(float(x), float(y)) for x, y in side_a.vertices]
    verts_b = [(float(x), float(y)) for x, y in side_b.vertices]
    verts_b_flipped = [(float(x), float(y)) for x, y in side_b.vertices_flipped]

    print(f"=== FLIP ANALYSIS: P{pid_a}[{si_a}] vs P{pid_b}[{si_b}] ===\n")

    print(f"B.vertices (normalized, angle=0):")
    print(f"  p1={verts_b[0]}, p2={verts_b[-1]}")
    print(f"  Y range: [{min(y for _,y in verts_b):.1f}, {max(y for _,y in verts_b):.1f}]")
    print(f"  This is B's side shape in original orientation (tab/blank as-is)")

    print(f"\nB.vertices_flipped (rotated to PI + reversed):")
    print(f"  p1={verts_b_flipped[0]}, p2={verts_b_flipped[-1]}")
    print(f"  Y range: [{min(y for _,y in verts_b_flipped):.1f}, {max(y for _,y in verts_b_flipped):.1f}]")
    print(f"  This is B's side shape FLIPPED (Y-mirrored)")

    print(f"\nA.vertices (normalized, angle=0):")
    print(f"  p1={verts_a[0]}, p2={verts_a[-1]}")
    print(f"  Y range: [{min(y for _,y in verts_a):.1f}, {max(y for _,y in verts_a):.1f}]")

    print(f"\n=== KEY QUESTION: Is B.vertices or B.vertices_flipped more similar to A? ===\n")

    p1_a = np.array(verts_a)
    p1_b = np.array(verts_b)
    p1_bf = np.array(verts_b_flipped)

    def calc_error(pa, pb, label):
        diff = np.abs(pa - pb)
        raw = float(np.sum(diff))
        ms = np.sum(diff - pa + pb, axis=0) / len(pa)
        ex, ey = float(ms[0]), max(-5.0, min(5.0, float(ms[1])))
        ps2 = np.array([(x - ex, y - ey) for x, y in pa])
        raw_s = float(np.sum(np.abs(ps2 - pb)))
        err = min(raw, raw_s) / side_b.v_length
        print(f"  {label}:")
        print(f"    raw={raw:.1f}, shifted={raw_s:.1f}, final={err:.6f}")
        return err

    err_flip = calc_error(p1_a, p1_bf, "A vs B_flipped (CURRENT ALGORITHM)")
    err_noflip = calc_error(p1_a, p1_b, "A vs B_noflip")

    print(f"\n  Flip gives LOWER error ({err_flip:.4f} < {err_noflip:.4f})")
    print(f"  This means B_flipped looks MORE like A than B_noflip does")
    print(f"  The flip makes B's shape look like A's shape")

    print(f"\n=== ANALYZING THE FLIP ===\n")

    y_a = [y for _, y in verts_a]
    y_b = [y for _, y in verts_b]
    y_bf = [y for _, y in verts_b_flipped]

    print(f"  A Y-values (first 10): {[f'{y:.1f}' for y in y_a[:10]]}")
    print(f"  B Y-values (first 10): {[f'{y:.1f}' for y in y_b[:10]]}")
    print(f"  B_flip Y-values (first 10): {[f'{y:.1f}' for y in y_bf[:10]]}")

    has_tab_a = max(y_a) > 10
    has_blank_a = min(y_a) < -10
    has_tab_b = max(y_b) > 10
    has_blank_b = min(y_b) < -10

    print(f"\n  A profile: tab={'YES' if has_tab_a else 'NO'} (max={max(y_a):.1f}), blank={'YES' if has_blank_a else 'NO'} (min={min(y_a):.1f})")
    print(f"  B profile: tab={'YES' if has_tab_b else 'NO'} (max={max(y_b):.1f}), blank={'YES' if has_blank_b else 'NO'} (min={min(y_b):.1f})")

    print(f"\n  B_flipped profile: tab={'YES' if max(y_bf)>10 else 'NO'} (max={max(y_bf):.1f}), blank={'YES' if min(y_bf)<-10 else 'NO'} (min={min(y_bf):.1f})")

    if has_tab_a and has_tab_b:
        print(f"\n  *** BOTH A and B have TABS (convex) ***")
        print(f"  After flipping B, B's tab becomes a blank-like shape")
        print(f"  So A(tab) vs B_flipped(blank-like) = looks like a match")
        print(f"  But physically, two tabs CANNOT interlock!")
        print(f"  THE FLIP IS CREATING A FALSE MATCH!")

    if has_blank_a and has_blank_b:
        print(f"\n  *** BOTH A and B have BLANKS (concave) ***")
        print(f"  After flipping B, B's blank becomes a tab-like shape")
        print(f"  So A(blank) vs B_flipped(tab-like) = looks like a match")
        print(f"  But physically, two blanks CANNOT interlock!")
        print(f"  THE FLIP IS CREATING A FALSE MATCH!")

    if (has_tab_a and has_blank_b) or (has_blank_a and has_tab_b):
        print(f"\n  A has tab, B has blank (or vice versa)")
        print(f"  The flip is CORRECT - tab should match blank")
        print(f"  The matching may be valid, but the shapes don't align well")

    _draw_polylines_overlay(
        [verts_a, verts_b],
        [(0, 100, 220), (220, 80, 0)],
        [f"A.vertices (tab/blank as-is)",
         f"B.vertices (tab/blank as-is, NO FLIP)"],
        os.path.join(OUTPUT_DIR, 'flip_1_both_noflip.png'),
        title=f"Both sides at angle=0, NO flip (err={err_noflip:.4f})"
    )

    _draw_polylines_overlay(
        [verts_a, verts_b_flipped],
        [(0, 100, 220), (220, 80, 0)],
        [f"A.vertices",
         f"B.vertices_flipped (Y-mirrored + reversed)"],
        os.path.join(OUTPUT_DIR, 'flip_2_with_flip.png'),
        title=f"WITH flip (err={err_flip:.4f})"
    )

    y_a_neg = [(x, -y) for x, y in verts_a]
    _draw_polylines_overlay(
        [y_a_neg, verts_b],
        [(0, 100, 220), (220, 80, 0)],
        [f"A with Y-mirrored (A upside down)",
         f"B.vertices (original)"],
        os.path.join(OUTPUT_DIR, 'flip_3_A_mirrored_vs_B.png'),
        title=f"A(Y-mirrored) vs B(original) - should match if flip is correct"
    )

    print(f"\n=== SUMMARY ===")
    print(f"  flip_1_both_noflip.png: Both sides as-is (error={err_noflip:.4f})")
    print(f"  flip_2_with_flip.png: A vs B_flipped (error={err_flip:.4f})")
    print(f"  flip_3_A_mirrored_vs_B.png: A(Y-mirrored) vs B(original)")
    print(f"  Look at these images to see if the flip creates false similarity")


if __name__ == '__main__':
    analyze_flip(1, 0, 70, 0)
