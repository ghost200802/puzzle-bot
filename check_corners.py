import os
import re
import math

svg_dir = r'f:\work_Puzzle_github\puzzle-bot\output\puzzle_new\3_vector'

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

def find_corners(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    targets = [
        ('TL', min_x, min_y),
        ('TR', max_x, min_y),
        ('BR', max_x, max_y),
        ('BL', min_x, max_y),
    ]

    corners = {}
    for name, tx, ty in targets:
        best_dist = float('inf')
        best_pt = None
        for p in points:
            d = math.hypot(p[0]-tx, p[1]-ty)
            if d < best_dist:
                best_dist = d
                best_pt = p
        corners[name] = best_pt

    return corners

output_lines = []
all_results = []
svg_files = sorted([f for f in os.listdir(svg_dir) if f.endswith('.svg')])

for fname in svg_files:
    svg_path = os.path.join(svg_dir, fname)
    points = parse_svg_points(svg_path)
    if not points or len(points) < 4:
        continue

    corners = find_corners(points)
    ordered = [corners['TL'], corners['TR'], corners['BR'], corners['BL']]

    top_len = math.hypot(ordered[1][0]-ordered[0][0], ordered[1][1]-ordered[0][1])
    right_len = math.hypot(ordered[2][0]-ordered[1][0], ordered[2][1]-ordered[1][1])
    bottom_len = math.hypot(ordered[2][0]-ordered[3][0], ordered[2][1]-ordered[3][1])
    left_len = math.hypot(ordered[3][0]-ordered[0][0], ordered[3][1]-ordered[0][1])

    tb_ratio = min(top_len, bottom_len) / max(top_len, bottom_len)
    lr_ratio = min(left_len, right_len) / max(left_len, right_len)
    squareness = min(tb_ratio, lr_ratio)

    piece_num = re.search(r'piece_(\d+)', fname)
    piece_id = piece_num.group(1) if piece_num else '?'

    all_results.append({
        'id': piece_id,
        'fname': fname,
        'top': top_len,
        'right': right_len,
        'bottom': bottom_len,
        'left': left_len,
        'tb_ratio': tb_ratio,
        'lr_ratio': lr_ratio,
        'squareness': squareness,
        'corners': corners,
    })

THRESHOLD = 0.90

problematic = [r for r in all_results if r['squareness'] < THRESHOLD]
problematic.sort(key=lambda x: x['squareness'])

all_results.sort(key=lambda x: int(x['id']) if x['id'].isdigit() else 0)

output_lines.append(f'Total SVG files: {len(svg_files)}')
output_lines.append(f'Squareness threshold: {THRESHOLD}')
output_lines.append(f'Problematic pieces (squareness < {THRESHOLD}): {len(problematic)}')
output_lines.append(f'  - tb_ratio = min(top,bottom)/max(top,bottom)')
output_lines.append(f'  - lr_ratio = min(left,right)/max(left,right)')
output_lines.append(f'  - squareness = min(tb_ratio, lr_ratio)')
output_lines.append('')
output_lines.append('=' * 70)
output_lines.append('All pieces sorted by squareness (worst first):')
output_lines.append('=' * 70)
output_lines.append('')

for r in problematic:
    output_lines.append(f'Piece #{r["id"]} ({r["fname"]})')
    output_lines.append(f'  Squareness: {r["squareness"]:.4f}  (tb_ratio={r["tb_ratio"]:.4f}, lr_ratio={r["lr_ratio"]:.4f})')
    output_lines.append(f'  Side lengths: top={r["top"]:.2f} right={r["right"]:.2f} bottom={r["bottom"]:.2f} left={r["left"]:.2f}')
    output_lines.append('')

output_lines.append('=' * 70)
output_lines.append('All pieces summary (sorted by piece id):')
output_lines.append('=' * 70)
output_lines.append('')
output_lines.append(f'{"Piece":<10} {"Squareness":<12} {"tb_ratio":<12} {"lr_ratio":<12} {"top":<10} {"right":<10} {"bottom":<10} {"left":<10}')
output_lines.append('-' * 96)
for r in all_results:
    marker = ' *' if r['squareness'] < THRESHOLD else '  '
    output_lines.append(f'#{r["id"]:<8} {r["squareness"]:<12.4f} {r["tb_ratio"]:<12.4f} {r["lr_ratio"]:<12.4f} {r["top"]:<10.2f} {r["right"]:<10.2f} {r["bottom"]:<10.2f} {r["left"]:<10.2f}{marker}')

output_text = '\n'.join(output_lines)
print(output_text)

with open(r'f:\work_Puzzle_github\puzzle-bot\corner_check_results.txt', 'w', encoding='utf-8') as f:
    f.write(output_text)
