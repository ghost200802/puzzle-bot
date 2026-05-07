import os
import re
import json
import math
from PIL import Image, ImageDraw, ImageFont
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

_here = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(_here, '..', '..', 'output', 'puzzle_new')
svg_dir = os.path.join(base_dir, '3_vector')
output_path = os.path.join(_here, '..', '..', 'problematic_pieces_grid.png')

THRESHOLD = 0.90

def parse_svg_points(svg_path):
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

def parse_svg_viewbox(svg_path):
    with open(svg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.search(r'viewBox="([^"]+)"', content)
    if m:
        parts = m.group(1).split()
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    return None

def load_piece_data(piece_id):
    sides = []
    for i in range(4):
        json_path = os.path.join(svg_dir, f'side_{piece_id}_{i}.json')
        if not os.path.exists(json_path):
            return None
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sides.append(data)
    return sides

def get_corner_indices_from_json(sides):
    n0 = len(sides[0]['vertices'])
    n1 = len(sides[1]['vertices'])
    n2 = len(sides[2]['vertices'])
    return [0, n0, n0 + n1, n0 + n1 + n2]

def compute_squareness(sides):
    corners_photo = []
    for s in sides:
        corners_photo.append(s['vertices'][0])

    top_len = math.hypot(corners_photo[1][0]-corners_photo[0][0], corners_photo[1][1]-corners_photo[0][1])
    right_len = math.hypot(corners_photo[2][0]-corners_photo[1][0], corners_photo[2][1]-corners_photo[1][1])
    bottom_len = math.hypot(corners_photo[2][0]-corners_photo[3][0], corners_photo[2][1]-corners_photo[3][1])
    left_len = math.hypot(corners_photo[3][0]-corners_photo[0][0], corners_photo[3][1]-corners_photo[0][1])

    tb_ratio = min(top_len, bottom_len) / max(top_len, bottom_len)
    lr_ratio = min(left_len, right_len) / max(left_len, right_len)
    return min(tb_ratio, lr_ratio)

svg_files = sorted([f for f in os.listdir(svg_dir) if f.endswith('.svg')])

problematic = []
for fname in svg_files:
    m = re.search(r'piece_(\d+)', fname)
    if not m:
        continue
    piece_id = int(m.group(1))

    sides = load_piece_data(piece_id)
    if sides is None:
        continue

    sq = compute_squareness(sides)
    if sq < THRESHOLD:
        corner_indices = get_corner_indices_from_json(sides)
        problematic.append({
            'id': piece_id,
            'fname': fname,
            'squareness': sq,
            'corner_indices': corner_indices,
        })

problematic.sort(key=lambda x: x['squareness'])

CELL_W = 200
CELL_H = 220
PAD = 10
LABEL_H = 35
COLS = 12
ROWS = math.ceil(len(problematic) / COLS)

img_w = COLS * (CELL_W + PAD) + PAD
img_h = ROWS * (CELL_H + PAD + LABEL_H) + PAD + 30

img = Image.new('RGB', (img_w, img_h), (30, 30, 30))
draw = ImageDraw.Draw(img)

try:
    font_label = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 13)
    font_id = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
    font_title = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)
except:
    font_label = ImageFont.load_default()
    font_id = ImageFont.load_default()
    font_title = ImageFont.load_default()

title_text = f"Problematic Puzzle Pieces (squareness < {THRESHOLD}): {len(problematic)} / {len(svg_files)}"
draw.text((PAD, PAD // 2), title_text, fill=(255, 255, 255), font=font_title)
title_offset = 30

for idx, piece in enumerate(problematic):
    row = idx // COLS
    col = idx % COLS

    x0 = PAD + col * (CELL_W + PAD)
    y0 = title_offset + PAD + row * (CELL_H + PAD + LABEL_H)

    svg_path = os.path.join(svg_dir, piece['fname'])
    drawing = svg2rlg(svg_path)
    svg_img = renderPM.drawToPIL(drawing).convert('RGB')

    svg_points = parse_svg_points(svg_path)
    vb = parse_svg_viewbox(svg_path)

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
            cell_draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=(255, 80, 80))

    img.paste(cell_img, (x0, y0))

    label_y = y0 + CELL_H + 2
    sq_color = (255, 80, 80) if piece['squareness'] < 0.7 else (255, 200, 80) if piece['squareness'] < 0.8 else (200, 200, 200)

    id_text = f"#{piece['id']}"
    draw.text((x0 + 2, label_y), id_text, fill=(255, 255, 255), font=font_id)

    sq_text = f"{piece['squareness']:.3f}"
    bbox = draw.textbbox((0, 0), sq_text, font=font_label)
    tw = bbox[2] - bbox[0]
    draw.text((x0 + CELL_W - tw - 4, label_y), sq_text, fill=sq_color, font=font_label)

img.save(output_path, 'PNG', dpi=(150, 150))
print(f"Saved to: {output_path}")
print(f"Image size: {img_w} x {img_h} pixels")
print(f"Grid: {COLS} cols x {ROWS} rows, {len(problematic)} pieces")
print(f"\nWorst 10 pieces:")
for p in problematic[:10]:
    print(f"  #{p['id']:>3}  squareness={p['squareness']:.4f}  ({p['fname']})")
