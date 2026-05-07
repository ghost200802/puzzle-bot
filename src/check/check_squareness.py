import os, sys, json, math, re
from PIL import Image, ImageDraw, ImageFont
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))
from common.config import VECTOR_DIR, CHECK_DIR

CHECK_THRESHOLD = 0.80


def run_check(output_dir):
    vector_dir = os.path.join(output_dir, VECTOR_DIR)
    check_dir = os.path.join(output_dir, CHECK_DIR)
    os.makedirs(check_dir, exist_ok=True)

    svg_files = sorted([f for f in os.listdir(vector_dir) if f.endswith('.svg')])
    if not svg_files:
        print("\nNo SVG files found, skipping check.")
        return

    problematic = []
    all_results = []

    for fname in svg_files:
        m = re.search(r'piece_(\d+)', fname)
        if not m:
            continue
        pid = int(m.group(1))

        sides = []
        skip = False
        for i in range(4):
            jp = os.path.join(vector_dir, f'side_{pid}_{i}.json')
            if not os.path.exists(jp):
                skip = True
                break
            with open(jp, 'r', encoding='utf-8') as f:
                sides.append(json.load(f))
        if skip:
            continue

        corners_photo = [s['vertices'][0] for s in sides]
        top_len = math.hypot(corners_photo[1][0] - corners_photo[0][0], corners_photo[1][1] - corners_photo[0][1])
        right_len = math.hypot(corners_photo[2][0] - corners_photo[1][0], corners_photo[2][1] - corners_photo[1][1])
        bottom_len = math.hypot(corners_photo[2][0] - corners_photo[3][0], corners_photo[2][1] - corners_photo[3][1])
        left_len = math.hypot(corners_photo[3][0] - corners_photo[0][0], corners_photo[3][1] - corners_photo[0][1])

        tb_ratio = min(top_len, bottom_len) / max(top_len, bottom_len) if max(top_len, bottom_len) > 0 else 0
        lr_ratio = min(left_len, right_len) / max(left_len, right_len) if max(left_len, right_len) > 0 else 0
        sq = min(tb_ratio, lr_ratio)

        n0 = len(sides[0]['vertices'])
        n1 = len(sides[1]['vertices'])
        n2 = len(sides[2]['vertices'])
        corner_indices = [0, n0, n0 + n1, n0 + n1 + n2]

        entry = {
            'id': pid,
            'fname': fname,
            'squareness': sq,
            'tb_ratio': tb_ratio,
            'lr_ratio': lr_ratio,
            'top': top_len,
            'bottom': bottom_len,
            'left': left_len,
            'right': right_len,
            'corner_indices': corner_indices,
        }
        all_results.append(entry)
        if sq < CHECK_THRESHOLD:
            problematic.append(entry)

    all_results.sort(key=lambda x: x['squareness'])
    problematic.sort(key=lambda x: x['squareness'])

    results_txt_path = os.path.join(check_dir, 'squareness_report.txt')
    with open(results_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Puzzle Piece Squareness Report\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Total pieces: {len(all_results)}\n")
        f.write(f"Threshold: {CHECK_THRESHOLD}\n")
        f.write(f"Flagged pieces (squareness < {CHECK_THRESHOLD}): {len(problematic)}\n")
        f.write(f"{'=' * 70}\n\n")

        f.write(f"--- All pieces sorted by squareness ---\n")
        for r in all_results:
            flag = ' <<< FLAGGED' if r['squareness'] < CHECK_THRESHOLD else ''
            f.write(f"  #{r['id']:>3}  sq={r['squareness']:.4f}  tb={r['tb_ratio']:.4f}  lr={r['lr_ratio']:.4f}  "
                    f"T={r['top']:.0f} B={r['bottom']:.0f} L={r['left']:.0f} R={r['right']:.0f}{flag}\n")

        if problematic:
            f.write(f"\n--- Flagged pieces detail ---\n")
            for p in problematic:
                f.write(f"\n  #{p['id']}  squareness={p['squareness']:.4f}\n")
                f.write(f"    tb_ratio={p['tb_ratio']:.4f}  lr_ratio={p['lr_ratio']:.4f}\n")
                f.write(f"    top={p['top']:.1f}  bottom={p['bottom']:.1f}  left={p['left']:.1f}  right={p['right']:.1f}\n")
                f.write(f"    corner_indices={p['corner_indices']}\n")

    if not problematic:
        print(f"\n{'=' * 60}")
        print(f"Check: All {len(all_results)} pieces pass (squareness >= {CHECK_THRESHOLD})")
        print(f"Report: {results_txt_path}")
        print(f"{'=' * 60}")
        return

    print(f"\n{'=' * 60}")
    print(f"Check: {len(problematic)} / {len(all_results)} pieces flagged (squareness < {CHECK_THRESHOLD})")
    print(f"Report: {results_txt_path}")
    print(f"{'=' * 60}")

    for p in problematic:
        print(f"  #{p['id']:>3}  sq={p['squareness']:.4f}  tb={p['tb_ratio']:.4f}  lr={p['lr_ratio']:.4f}")

    print(f"\nGenerating visual grid for flagged pieces...")
    _generate_grid(problematic, all_results, vector_dir, check_dir)


def _generate_grid(problematic, all_results, vector_dir, check_dir):
    CELL_W = 220
    CELL_H = 240
    PAD = 10
    COLS = 12
    ROWS = math.ceil(len(problematic) / COLS)

    img_w = COLS * (CELL_W + PAD) + PAD
    img_h = ROWS * (CELL_H + PAD + 50) + PAD + 30

    img = Image.new('RGB', (img_w, img_h), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    try:
        font_label = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 13)
        font_id = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
        font_title = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)
    except Exception:
        font_label = ImageFont.load_default()
        font_id = ImageFont.load_default()
        font_title = ImageFont.load_default()

    title_text = f"Flagged Pieces (squareness < {CHECK_THRESHOLD}): {len(problematic)} / {len(all_results)}"
    draw.text((PAD, PAD // 2), title_text, fill=(255, 255, 255), font=font_title)
    title_offset = 30

    for idx, piece in enumerate(problematic):
        row = idx // COLS
        col = idx % COLS

        x0 = PAD + col * (CELL_W + PAD)
        y0 = title_offset + PAD + row * (CELL_H + PAD + 50)

        svg_path = os.path.join(vector_dir, piece['fname'])
        try:
            drawing = svg2rlg(svg_path)
            svg_img = renderPM.drawToPIL(drawing).convert('RGB')
        except Exception:
            cell_img = Image.new('RGB', (CELL_W, CELL_H), (40, 0, 0))
            cell_draw = ImageDraw.Draw(cell_img)
            cell_draw.text((10, CELL_H // 2), "SVG error", fill=(255, 0, 0))
            img.paste(cell_img, (x0, y0))
            label_y = y0 + CELL_H + 2
            draw.text((x0 + 2, label_y), f"#{piece['id']}", fill=(255, 255, 255), font=font_id)
            continue

        svg_points = _parse_svg_points(svg_path)
        vb = _parse_svg_viewbox(svg_path)

        vb_x, vb_y, vb_w, vb_h = vb
        svg_pixel_w = drawing.width
        svg_pixel_h = drawing.height
        scale_x = svg_pixel_w / vb_w if vb_w > 0 else 1
        scale_y = svg_pixel_h / vb_h if vb_h > 0 else 1

        iw, ih = svg_img.size
        margin = 8
        draw_w = CELL_W - 2 * margin
        draw_h = CELL_H - 2 * margin
        scale = min(draw_w / iw, draw_h / ih) if iw > 0 and ih > 0 else 1
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        svg_resized = svg_img.resize((new_w, new_h), Image.LANCZOS)

        cell_img = Image.new('RGB', (CELL_W, CELL_H), (15, 15, 15))
        paste_x = margin + (draw_w - new_w) // 2
        paste_y = margin + (draw_h - new_h) // 2
        cell_img.paste(svg_resized, (paste_x, paste_y))

        cell_draw = ImageDraw.Draw(cell_img)

        corner_pts = []
        for ci in piece['corner_indices']:
            if ci < len(svg_points):
                px, py = svg_points[ci]
                sx = (px - vb_x) * scale_x * scale + paste_x
                sy = (py - vb_y) * scale_y * scale + paste_y
                corner_pts.append((sx, sy))

        if len(corner_pts) == 4:
            cell_draw.polygon(corner_pts, outline=(70, 180, 255), fill=None)
            for sx, sy in corner_pts:
                r = 3
                cell_draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(255, 80, 80))

        img.paste(cell_img, (x0, y0))

        label_y = y0 + CELL_H + 2
        sq_color = (255, 80, 80) if piece['squareness'] < 0.7 else (255, 200, 80) if piece['squareness'] < 0.8 else (200, 200, 200)

        draw.text((x0 + 2, label_y), f"#{piece['id']}", fill=(255, 255, 255), font=font_id)

        sq_text = f"{piece['squareness']:.3f}"
        bbox = draw.textbbox((0, 0), sq_text, font=font_label)
        tw = bbox[2] - bbox[0]
        draw.text((x0 + CELL_W - tw - 4, label_y), sq_text, fill=sq_color, font=font_label)

        tb_text = f"tb={piece['tb_ratio']:.2f}"
        draw.text((x0 + 35, label_y), tb_text, fill=(160, 160, 160), font=font_label)

        lr_text = f"lr={piece['lr_ratio']:.2f}"
        draw.text((x0 + 35, label_y + 15), lr_text, fill=(160, 160, 160), font=font_label)

    grid_path = os.path.join(check_dir, 'flagged_pieces_grid.png')
    img.save(grid_path, 'PNG', dpi=(150, 150))
    print(f"Grid saved to: {grid_path}")
    print(f"Image size: {img_w} x {img_h} pixels, {COLS} cols x {ROWS} rows")


def _parse_svg_points(svg_path):
    with open(svg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    points_all = []
    for match in re.finditer(r'points="([^"]+)"', content):
        pts_str = match.group(1)
        coords = pts_str.strip().split()
        for c in coords:
            x, y = c.split(',')
            points_all.append((float(x), float(y)))
    return points_all


def _parse_svg_viewbox(svg_path):
    with open(svg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.search(r'viewBox="([^"]+)"', content)
    if m:
        parts = m.group(1).split()
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    return 0, 0, 1, 1


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'puzzle_new')
    run_check(output_dir)
