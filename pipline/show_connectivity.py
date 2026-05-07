import os
import sys
import json
import math
from collections import deque
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from PIL import Image, ImageDraw, ImageFont


def show(connectivity_dir, deduped_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(connectivity_dir, 'connectivity.json')) as f:
        connectivity = json.load(f)
    with open(os.path.join(connectivity_dir, 'piece_edge_info.json')) as f:
        piece_edge_info = json.load(f)
    piece_edge_info = {int(k): v for k, v in piece_edge_info.items()}

    piece_data = _load_piece_data(deduped_dir)

    corners = sorted([pid for pid, flags in piece_edge_info.items()
                      if sum(1 for f in flags if f) >= 2])
    edges = sorted([pid for pid, flags in piece_edge_info.items()
                    if sum(1 for f in flags if f) == 1])
    inner = sorted([pid for pid, flags in piece_edge_info.items()
                    if sum(1 for f in flags if f) == 0])

    print(f"\n  Pieces: {len(piece_data)} total, {len(corners)} corners, {len(edges)} edges, {len(inner)} inner")

    placed, match_log = _greedy_assemble(connectivity, piece_data, piece_edge_info)

    _draw_assembly_png(
        placed, piece_edge_info, match_log,
        os.path.join(output_dir, 'assembly.png'),
        title=f"Greedy Assembly ({len(placed)}/{len(piece_data)} pieces)"
    )

    _draw_pair_images(connectivity, piece_data, piece_edge_info, output_dir)

    _write_report(placed, match_log, piece_edge_info, connectivity,
                  os.path.join(output_dir, 'report.txt'))

    print(f"  Assembly: {len(placed)}/{len(piece_data)} pieces placed")
    print(f"  Output: {output_dir}")


def _load_piece_data(deduped_dir):
    pieces = {}
    vp = Path(deduped_dir)
    for path in sorted(vp.glob("side_*_0.json")):
        pid = int(path.parts[-1].split('_')[1])
        sides = []
        for i in range(4):
            spath = vp / f'side_{pid}_{i}.json'
            with open(spath) as f:
                data = json.load(f)
            sides.append({
                'vertices': [tuple(v) for v in data['vertices']],
                'is_edge': data.get('is_edge', False),
            })
        pieces[pid] = sides
    return pieces


def _side_angle(vertices):
    p1, p2 = vertices[0], vertices[-1]
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def _side_midpoint(vertices):
    p1, p2 = vertices[0], vertices[-1]
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _transform_piece(src_sides, src_side_idx, target_side_verts):
    src_verts = src_sides[src_side_idx]['vertices']
    src_mid = _side_midpoint(src_verts)
    src_theta = _side_angle(src_verts)

    tgt_mid = _side_midpoint(target_side_verts)
    tgt_theta = _side_angle(target_side_verts)

    cos_b = math.cos(-src_theta)
    sin_b = math.sin(-src_theta)
    cos_a = math.cos(tgt_theta)
    sin_a = math.sin(tgt_theta)

    transformed = []
    for side in src_sides:
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

        transformed.append({
            'vertices': new_verts,
            'is_edge': side['is_edge'],
        })
    return transformed


def _greedy_assemble(connectivity, piece_data, piece_edge_info):
    all_pids = set(int(p) for p in connectivity.keys())

    corners = sorted([pid for pid, flags in piece_edge_info.items()
                      if sum(1 for f in flags if f) >= 2])
    remaining = sorted(all_pids - set(corners))
    start_order = list(corners) + remaining

    placed = {}
    match_log = []
    used_sides = {}

    for start_pid in start_order:
        if start_pid in placed:
            continue
        if start_pid not in piece_data:
            continue

        placed[start_pid] = piece_data[start_pid]
        used_sides[start_pid] = set()
        queue = deque([start_pid])

        while queue:
            current = queue.popleft()
            pid_str = str(current)
            if pid_str not in connectivity:
                continue

            fits = connectivity[pid_str]
            current_sides = placed[current]

            for si in range(4):
                if si in used_sides.get(current, set()):
                    continue

                side_matches = fits[si] if si < len(fits) else []
                if not side_matches:
                    continue

                world_side = current_sides[si]['vertices']

                for other_pid, other_si, error in side_matches:
                    if other_pid in placed:
                        continue
                    if other_pid not in piece_data:
                        continue

                    transformed = _transform_piece(
                        piece_data[other_pid], other_si, world_side
                    )

                    placed[other_pid] = transformed
                    used_sides[other_pid] = {other_si}
                    used_sides.setdefault(current, set()).add(si)
                    match_log.append((current, si, other_pid, other_si, error))
                    queue.append(other_pid)
                    break

    return placed, match_log


def _get_outline(sides):
    pts = list(sides[0]['vertices'])
    for i in range(1, 4):
        v = sides[i]['vertices']
        if pts and abs(pts[-1][0] - v[0][0]) < 5 and abs(pts[-1][1] - v[0][1]) < 5:
            pts.extend(v[1:])
        else:
            pts.extend(v)
    return pts


def _get_centroid(vertices):
    cx = sum(p[0] for p in vertices) / len(vertices)
    cy = sum(p[1] for p in vertices) / len(vertices)
    return (cx, cy)


def _piece_color(pid, piece_edge_info):
    ef = piece_edge_info.get(pid, [False] * 4)
    flat_count = sum(1 for f in ef if f)
    if flat_count >= 2:
        return (255, 80, 80, 180), (200, 0, 0, 255)
    elif flat_count >= 1:
        return (80, 130, 255, 180), (0, 60, 200, 255)
    else:
        return (80, 200, 80, 180), (0, 140, 0, 255)


def _draw_assembly_png(placed, piece_edge_info, match_log, output_path, title=""):
    if not placed:
        return

    all_pts = []
    for pid, sides in placed.items():
        outline = _get_outline(sides)
        all_pts.extend(outline)

    if not all_pts:
        return

    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_y = max(p[1] for p in all_pts)

    data_w = max_x - min_x
    data_h = max_y - min_y
    if data_w == 0:
        data_w = 1
    if data_h == 0:
        data_h = 1

    margin = max(data_w, data_h) * 0.05
    canvas_w = int(data_w + 2 * margin)
    canvas_h = int(data_h + 2 * margin)

    header_h = 60
    canvas_h += header_h

    max_size = 8000
    if max(canvas_w, canvas_h) > max_size:
        scale = max_size / max(canvas_w, canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)
    else:
        scale = 1.0

    img = Image.new('RGBA', (max(canvas_w, 100), max(canvas_h, 100)), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", max(10, min(24, int(20 * scale))))
        title_font = ImageFont.truetype("arial.ttf", max(12, min(28, int(24 * scale))))
        small_font = ImageFont.truetype("arial.ttf", max(8, min(16, int(14 * scale))))
    except Exception:
        font = ImageFont.load_default()
        title_font = font
        small_font = font

    draw.text((10, 5), title, fill=(0, 0, 0, 255), font=title_font)

    legend_y = 35
    draw.rectangle([10, legend_y, 25, legend_y + 12], fill=(255, 80, 80, 180), outline=(200, 0, 0, 255))
    draw.text((30, legend_y - 2), "Corner", fill=(0, 0, 0, 255), font=small_font)
    draw.rectangle([100, legend_y, 115, legend_y + 12], fill=(80, 130, 255, 180), outline=(0, 60, 200, 255))
    draw.text((120, legend_y - 2), "Edge", fill=(0, 0, 0, 255), font=small_font)
    draw.rectangle([180, legend_y, 195, legend_y + 12], fill=(80, 200, 80, 180), outline=(0, 140, 0, 255))
    draw.text((200, legend_y - 2), "Inner", fill=(0, 0, 0, 255), font=small_font)

    def to_canvas(x, y):
        cx = (x - min_x + margin) * scale
        cy = (y - min_y + margin) * scale + header_h
        return (cx, cy)

    for pid, sides in placed.items():
        fill_color, outline_color = _piece_color(pid, piece_edge_info)
        outline = _get_outline(sides)
        canvas_pts = [to_canvas(x, y) for x, y in outline]

        if len(canvas_pts) >= 3:
            draw.polygon(canvas_pts, fill=fill_color, outline=outline_color)

        cx, cy = _get_centroid(canvas_pts)
        draw.text((cx, cy), str(pid), fill=(0, 0, 0, 255), font=font, anchor="mm")

    img.save(output_path)
    print(f"  Saved: {output_path} ({canvas_w}x{canvas_h})")


def _draw_pair_images(connectivity, piece_data, piece_edge_info, output_dir):
    pairs_dir = os.path.join(output_dir, 'pairs')
    os.makedirs(pairs_dir, exist_ok=True)

    drawn_pairs = set()

    for pid_str, fits in connectivity.items():
        pid = int(pid_str)
        if pid not in piece_data:
            continue

        for si in range(4):
            side_matches = fits[si] if si < len(fits) else []
            if not side_matches:
                continue

            other_pid, other_si, error = side_matches[0]

            pair_key = (min(pid, other_pid), max(pid, other_pid),
                        min(si, other_si), max(si, other_si))
            if pair_key in drawn_pairs:
                continue
            drawn_pairs.add(pair_key)

            if other_pid not in piece_data:
                continue

            _draw_single_pair(
                pid, si, other_pid, other_si, error,
                piece_data, piece_edge_info,
                os.path.join(pairs_dir, f'pair_{pid}s{si}_{other_pid}s{other_si}.png')
            )

    print(f"  Generated {len(drawn_pairs)} pair images in {pairs_dir}")


def _draw_single_pair(pid_a, si_a, pid_b, si_b, error, piece_data, piece_edge_info, output_path):
    sides_a = piece_data[pid_a]
    sides_b = piece_data[pid_b]

    transformed_b = _transform_piece(sides_b, si_b, sides_a[si_a]['vertices'])

    all_pts = []
    for side in sides_a:
        all_pts.extend(side['vertices'])
    for side in transformed_b:
        all_pts.extend(side['vertices'])

    if not all_pts:
        return

    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_y = max(p[1] for p in all_pts)

    data_w = max_x - min_x
    data_h = max_y - min_y
    if data_w == 0:
        data_w = 1
    if data_h == 0:
        data_h = 1

    margin = max(data_w, data_h) * 0.15
    canvas_w = int(data_w + 2 * margin)
    canvas_h = int(data_h + 2 * margin)

    header_h = 40
    canvas_h += header_h

    max_dim = 1500
    if max(canvas_w, canvas_h) > max_dim:
        scale = max_dim / max(canvas_w, canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)
    else:
        scale = 1.0

    img = Image.new('RGBA', (max(canvas_w, 50), max(canvas_h, 50)), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", max(8, min(18, int(16 * scale))))
        title_font = ImageFont.truetype("arial.ttf", max(10, min(20, int(18 * scale))))
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    err_display = error / 1000.0
    draw.text((5, 3), f"#{pid_a}[{si_a}] <-> #{pid_b}[{si_b}]  err={err_display:.3f}",
              fill=(0, 0, 0, 255), font=title_font)

    def to_canvas(x, y):
        cx = (x - min_x + margin) * scale
        cy = (y - min_y + margin) * scale + header_h
        return (cx, cy)

    fill_a, outline_a = _piece_color(pid_a, piece_edge_info)
    outline_a_pts = _get_outline(sides_a)
    canvas_pts_a = [to_canvas(x, y) for x, y in outline_a_pts]
    if len(canvas_pts_a) >= 3:
        draw.polygon(canvas_pts_a, fill=fill_a, outline=outline_a)
    cx, cy = _get_centroid(canvas_pts_a)
    draw.text((cx, cy), str(pid_a), fill=(0, 0, 0, 255), font=font, anchor="mm")

    fill_b, outline_b = _piece_color(pid_b, piece_edge_info)
    outline_b_pts = _get_outline(transformed_b)
    canvas_pts_b = [to_canvas(x, y) for x, y in outline_b_pts]
    if len(canvas_pts_b) >= 3:
        draw.polygon(canvas_pts_b, fill=fill_b, outline=outline_b)
    cx, cy = _get_centroid(canvas_pts_b)
    draw.text((cx, cy), str(pid_b), fill=(0, 0, 0, 255), font=font, anchor="mm")

    match_pts_a = [to_canvas(x, y) for x, y in sides_a[si_a]['vertices']]
    match_pts_b = [to_canvas(x, y) for x, y in transformed_b[si_b]['vertices']]

    if len(match_pts_a) >= 2:
        draw.line(match_pts_a, fill=(255, 165, 0, 255), width=max(1, int(3 * scale)))
    if len(match_pts_b) >= 2:
        draw.line(match_pts_b, fill=(255, 165, 0, 255), width=max(1, int(3 * scale)))

    img.save(output_path)


def _write_report(placed, match_log, piece_edge_info, connectivity, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Connectivity Visualization Report\n")
        f.write("=" * 70 + "\n\n")

        total_pieces = len(piece_edge_info)
        corners = sorted([pid for pid, flags in piece_edge_info.items()
                          if sum(1 for fl in flags if fl) >= 2])
        edges = sorted([pid for pid, flags in piece_edge_info.items()
                        if sum(1 for fl in flags if fl) == 1])
        inner = sorted([pid for pid, flags in piece_edge_info.items()
                        if sum(1 for fl in flags if fl) == 0])

        f.write(f"Total pieces: {total_pieces}\n")
        f.write(f"  Corners: {len(corners)} -> {corners}\n")
        f.write(f"  Edges:   {len(edges)}\n")
        f.write(f"  Inner:   {len(inner)}\n\n")

        f.write(f"Greedy assembly placed: {len(placed)}/{total_pieces}\n")
        f.write(f"Matches used: {len(match_log)}\n\n")

        unplaced = sorted(set(int(p) for p in connectivity.keys()) - set(placed.keys()))
        if unplaced:
            f.write(f"Unplaced pieces ({len(unplaced)}): {unplaced}\n\n")

        f.write("-" * 70 + "\n")
        f.write("Match Log (from -> to):\n")
        f.write("-" * 70 + "\n")
        for from_pid, from_si, to_pid, to_si, error in match_log:
            f.write(f"  #{from_pid}[side{from_si}] -> #{to_pid}[side{to_si}]  error={error / 1000.0:.3f}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("Per-piece connectivity summary:\n")
        f.write("-" * 70 + "\n")
        for pid_str, fits in sorted(connectivity.items(), key=lambda x: int(x[0])):
            pid = int(pid_str)
            ef = piece_edge_info.get(pid, [False] * 4)
            flat = sum(1 for fl in ef if fl)
            ptype = "CORNER" if flat >= 2 else ("EDGE" if flat >= 1 else "INNER")
            f.write(f"\n  Piece #{pid} ({ptype}):\n")
            for si in range(4):
                is_edge_side = ef[si] if si < len(ef) else False
                matches = fits[si] if si < len(fits) else []
                edge_mark = " [EDGE]" if is_edge_side else ""
                if matches:
                    top3 = matches[:3]
                    match_str = ", ".join(
                        f"#{m[0]}[s{m[1]}] err={m[2] / 1000.0:.3f}" for m in top3
                    )
                    more = f" ... +{len(matches) - 3} more" if len(matches) > 3 else ""
                    f.write(f"    side{si}{edge_mark}: {match_str}{more}\n")
                else:
                    f.write(f"    side{si}{edge_mark}: (no matches)\n")

    print(f"  Saved: {output_path}")


if __name__ == '__main__':
    OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
    DEDUPED_PATH = os.path.join(OUTPUT_DIR, '4_deduped')
    CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, '5_connectivity')
    CHECK_PATH = os.path.join(OUTPUT_DIR, 'check', 'connectivity')

    show(CONNECTIVITY_PATH, DEDUPED_PATH, CHECK_PATH)
