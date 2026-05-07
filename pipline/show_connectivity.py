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
    piece_imgs = _load_piece_images(deduped_dir)

    corners = sorted([pid for pid, flags in piece_edge_info.items()
                      if sum(1 for f in flags if f) >= 2])
    edges = sorted([pid for pid, flags in piece_edge_info.items()
                    if sum(1 for f in flags if f) == 1])
    inner = sorted([pid for pid, flags in piece_edge_info.items()
                    if sum(1 for f in flags if f) == 0])

    print(f"\n  Pieces: {len(piece_data)} total, {len(corners)} corners, {len(edges)} edges, {len(inner)} inner")
    print(f"  Color images loaded: {len(piece_imgs)}")

    placed, match_log = _greedy_assemble(connectivity, piece_data, piece_edge_info)

    _draw_assembly_png(
        placed, piece_data, piece_imgs, piece_edge_info, match_log,
        os.path.join(output_dir, 'assembly.png'),
        title=f"Assembly ({len(placed)}/{len(piece_data)} pieces)"
    )

    _draw_piece_pages(connectivity, piece_data, piece_imgs, piece_edge_info, output_dir)

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


def _load_piece_images(deduped_dir):
    imgs = {}
    parent = Path(deduped_dir).parent
    color_dir = parent / '2_piece_colors'
    if not color_dir.exists():
        print(f"  Warning: {color_dir} not found, images will not be loaded")
        return imgs
    for path in sorted(color_dir.glob("piece_*.png")):
        pid = int(path.stem.split('_')[1])
        imgs[pid] = Image.open(str(path)).convert('RGBA')
    return imgs


def _side_angle(vertices):
    p1, p2 = vertices[0], vertices[-1]
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def _side_midpoint(vertices):
    p1, p2 = vertices[0], vertices[-1]
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _transform_vertices(src_sides, src_side_idx, target_side_verts):
    src_verts = src_sides[src_side_idx]['vertices']
    src_mid = _side_midpoint(src_verts)
    src_theta = _side_angle(src_verts)
    tgt_mid = _side_midpoint(target_side_verts)
    tgt_theta = _side_angle(target_side_verts)
    rot_angle = tgt_theta + math.pi - src_theta
    cos_r = math.cos(rot_angle)
    sin_r = math.sin(rot_angle)

    transformed = []
    for side in src_sides:
        new_verts = []
        for v in side['vertices']:
            dx = v[0] - src_mid[0]
            dy = v[1] - src_mid[1]
            fx = dx * cos_r - dy * sin_r
            fy = dx * sin_r + dy * cos_r
            new_verts.append((fx + tgt_mid[0], fy + tgt_mid[1]))
        transformed.append({
            'vertices': new_verts,
            'is_edge': side['is_edge'],
        })
    return transformed


def _transform_image(img, src_sides, src_side_idx, target_side_verts):
    src_verts = src_sides[src_side_idx]['vertices']
    src_mid = _side_midpoint(src_verts)
    src_theta = _side_angle(src_verts)
    tgt_mid = _side_midpoint(target_side_verts)
    tgt_theta = _side_angle(target_side_verts)

    rot_angle = tgt_theta + math.pi - src_theta
    cos_r = math.cos(rot_angle)
    sin_r = math.sin(rot_angle)
    cos_neg = math.cos(-rot_angle)
    sin_neg = math.sin(-rot_angle)

    w, h = img.size
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    out_corners = []
    for x, y in corners:
        dx = x - src_mid[0]
        dy = y - src_mid[1]
        ox = dx * cos_r - dy * sin_r + tgt_mid[0]
        oy = dx * sin_r + dy * cos_r + tgt_mid[1]
        out_corners.append((ox, oy))

    all_x = [c[0] for c in out_corners]
    all_y = [c[1] for c in out_corners]
    out_min_x = min(all_x)
    out_min_y = min(all_y)
    out_max_x = max(all_x)
    out_max_y = max(all_y)

    out_w = int(math.ceil(out_max_x - out_min_x)) + 1
    out_h = int(math.ceil(out_max_y - out_min_y)) + 1

    a = cos_neg
    b = -sin_neg
    c = cos_neg * (out_min_x - tgt_mid[0]) - sin_neg * (out_min_y - tgt_mid[1]) + src_mid[0]
    d = sin_neg
    e = cos_neg
    f = sin_neg * (out_min_x - tgt_mid[0]) + cos_neg * (out_min_y - tgt_mid[1]) + src_mid[1]

    transformed = img.transform(
        (out_w, out_h),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BICUBIC,
    )

    return transformed, (out_min_x, out_min_y)


def _greedy_assemble(connectivity, piece_data, piece_edge_info):
    all_pids = set(int(p) for p in connectivity.keys())

    corners = sorted([pid for pid, flags in piece_edge_info.items()
                      if sum(1 for f in flags if f) >= 2])
    remaining = sorted(all_pids - set(corners))
    start_order = list(corners) + remaining

    placed = {}
    placed_src_side = {}
    match_log = []
    used_sides = {}

    for start_pid in start_order:
        if start_pid in placed:
            continue
        if start_pid not in piece_data:
            continue

        placed[start_pid] = piece_data[start_pid]
        placed_src_side[start_pid] = None
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

                for m in side_matches:
                    other_pid = m['pid']
                    other_si = m['si']
                    error = m['error']

                    if other_pid in placed:
                        continue
                    if other_pid not in piece_data:
                        continue

                    transformed = _transform_vertices(
                        piece_data[other_pid], other_si, world_side
                    )

                    placed[other_pid] = transformed
                    placed_src_side[other_pid] = other_si
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


def _draw_assembly_png(placed, piece_data, piece_imgs, piece_edge_info, match_log, output_path, title=""):
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

    max_size = 12000
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
        if pid in piece_imgs:
            src_sides = piece_data[pid]
            src_side_idx = 0
            for si in range(4):
                sv = src_sides[si]['vertices']
                tv = sides[si]['vertices']
                if abs(sv[0][0] - tv[0][0]) < 5 and abs(sv[0][1] - tv[0][1]) < 5:
                    src_side_idx = si
                    break

            world_side_verts = sides[src_side_idx]['vertices']
            src_side_verts = src_sides[src_side_idx]['vertices']
            src_mid = _side_midpoint(src_side_verts)
            src_theta = _side_angle(src_side_verts)
            tgt_mid = _side_midpoint(world_side_verts)
            tgt_theta = _side_angle(world_side_verts)
            rot_angle = tgt_theta + math.pi - src_theta
            cos_r = math.cos(rot_angle)
            sin_r = math.sin(rot_angle)

            piece_img = piece_imgs[pid]
            w_img, h_img = piece_img.size
            corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
            img_pts = []
            for cx, cy in corners:
                dx = cx - src_mid[0]
                dy = cy - src_mid[1]
                ox = dx * cos_r - dy * sin_r + tgt_mid[0]
                oy = dx * sin_r + dy * cos_r + tgt_mid[1]
                img_pts.append((ox, oy))

            img_all_x = [p[0] for p in img_pts]
            img_all_y = [p[1] for p in img_pts]
            img_min_x = min(img_all_x)
            img_min_y = min(img_all_y)
            img_max_x = max(img_all_x)
            img_max_y = max(img_all_y)

            out_w = int(math.ceil((img_max_x - img_min_x) * scale)) + 2
            out_h = int(math.ceil((img_max_y - img_min_y) * scale)) + 2

            if out_w > 0 and out_h > 0:
                cos_neg = math.cos(-rot_angle)
                sin_neg = math.sin(-rot_angle)
                sm_x = (img_min_x - tgt_mid[0]) * scale
                sm_y = (img_min_y - tgt_mid[1]) * scale
                a = cos_neg * scale
                b = -sin_neg * scale
                c = cos_neg * sm_x - sin_neg * sm_y + src_mid[0]
                d = sin_neg * scale
                e = cos_neg * scale
                f = sin_neg * sm_x + cos_neg * sm_y + src_mid[1]

                try:
                    transformed = piece_img.transform(
                        (out_w, out_h), Image.AFFINE,
                        (a, b, c, d, e, f),
                        resample=Image.BICUBIC,
                    )
                    paste_x = int((img_min_x - min_x + margin) * scale)
                    paste_y = int((img_min_y - min_y + margin) * scale + header_h)
                    img.paste(transformed, (paste_x, paste_y), transformed)
                except Exception:
                    pass
        else:
            fill_color, outline_color = _piece_color(pid, piece_edge_info)
            outline = _get_outline(sides)
            canvas_pts = [to_canvas(x, y) for x, y in outline]
            if len(canvas_pts) >= 3:
                draw.polygon(canvas_pts, fill=fill_color, outline=outline_color)

    for pid, sides in placed.items():
        outline = _get_outline(sides)
        canvas_pts = [to_canvas(x, y) for x, y in outline]
        cx, cy = _get_centroid(canvas_pts)
        draw.text((cx, cy), str(pid), fill=(0, 0, 0, 255), font=font, anchor="mm")

    img.save(output_path)
    print(f"  Saved: {output_path} ({canvas_w}x{canvas_h})")


def _draw_piece_pages(connectivity, piece_data, piece_imgs, piece_edge_info, output_dir):
    pages_dir = os.path.join(output_dir, 'pieces')
    os.makedirs(pages_dir, exist_ok=True)

    total_drawn = 0
    for pid_str, fits in sorted(connectivity.items(), key=lambda x: int(x[0])):
        pid = int(pid_str)
        if pid not in piece_data:
            continue
        _draw_piece_page(
            pid, fits, piece_data, piece_imgs, piece_edge_info,
            os.path.join(pages_dir, f'piece_{pid}.png')
        )
        total_drawn += 1

    print(f"  Generated {total_drawn} piece pages in {pages_dir}")


def _render_pair_cell(pid_a, si_a, pid_b, si_b, piece_data, piece_imgs, cell_w, cell_h):
    sides_a = piece_data[pid_a]
    sides_b = piece_data[pid_b]
    img_a = piece_imgs.get(pid_a)
    img_b = piece_imgs.get(pid_b)

    target_side_verts = sides_a[si_a]['vertices']
    tgt_mid = _side_midpoint(target_side_verts)
    tgt_theta = _side_angle(target_side_verts)
    src_verts = sides_b[si_b]['vertices']
    src_mid = _side_midpoint(src_verts)
    src_theta = _side_angle(src_verts)
    rot_angle = tgt_theta + math.pi - src_theta
    cos_r = math.cos(rot_angle)
    sin_r = math.sin(rot_angle)

    all_pts = []

    if img_a:
        all_pts.extend([(0, 0), (img_a.size[0], 0), (img_a.size[0], img_a.size[1]), (0, img_a.size[1])])
    else:
        outline_a = _get_outline(sides_a)
        all_pts.extend(outline_a)

    if img_b:
        w_b, h_b = img_b.size
        for x, y in [(0, 0), (w_b, 0), (w_b, h_b), (0, h_b)]:
            dx = x - src_mid[0]
            dy = y - src_mid[1]
            ox = dx * cos_r - dy * sin_r + tgt_mid[0]
            oy = dx * sin_r + dy * cos_r + tgt_mid[1]
            all_pts.append((ox, oy))
    else:
        transformed_sides_b = _transform_vertices(sides_b, si_b, target_side_verts)
        outline_b = _get_outline(transformed_sides_b)
        all_pts.extend(outline_b)

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

    transformed_b_img = None
    b_off = None
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
        except Exception:
            transformed_b_img = None

    if transformed_b_img:
        px = int(to_cell(b_off[0], b_off[1])[0])
        py = int(to_cell(b_off[0], b_off[1])[1])
        tw = int(transformed_b_img.size[0] * scale)
        th = int(transformed_b_img.size[1] * scale)
        if tw > 0 and th > 0:
            resized_b = transformed_b_img.resize((tw, th), Image.LANCZOS)
            cell_img.paste(resized_b, (px, py), resized_b)

    if img_a:
        px = int(to_cell(0, 0)[0])
        py = int(to_cell(0, 0)[1])
        tw = int(img_a.size[0] * scale)
        th = int(img_a.size[1] * scale)
        if tw > 0 and th > 0:
            resized_a = img_a.resize((tw, th), Image.LANCZOS)
            cell_img.paste(resized_a, (px, py), resized_a)

    return cell_img


def _draw_piece_page(pid, fits, piece_data, piece_imgs, piece_edge_info, output_path):
    sides_data = piece_data[pid]
    ef = piece_edge_info.get(pid, [False] * 4)
    flat_count = sum(1 for f in ef if f)
    ptype = "CORNER" if flat_count >= 2 else ("EDGE" if flat_count >= 1 else "INNER")

    cells = []
    for si in range(4):
        side_matches = fits[si] if si < len(fits) else []
        for m in side_matches[:3]:
            other_pid = m['pid']
            other_si = m['si']
            if other_pid not in piece_data:
                continue
            err_val = m['error'] / 1000.0
            ld_val = m.get('len_diff', 0) / 1000.0
            cells.append((si, other_pid, other_si, err_val, ld_val))

    cells = cells[:12]

    if not cells:
        return

    n_cells = len(cells) + 1
    cols = min(4, n_cells)
    rows = math.ceil(n_cells / cols)

    cell_w = 600
    cell_h = 600
    gap = 10
    header_h = 50
    label_h = 22

    canvas_w = cols * (cell_w + gap) + gap
    canvas_h = rows * (cell_h + gap + label_h) + header_h + gap

    canvas = Image.new('RGBA', (canvas_w, canvas_h), (230, 230, 230, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        title_font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        title_font = font
        small_font = font

    draw.text((gap, 5), f"Piece #{pid} ({ptype})", fill=(0, 0, 0, 255), font=title_font)

    match_summary = []
    for si in range(4):
        n = len(fits[si]) if si < len(fits) else 0
        is_edge = ef[si] if si < len(ef) else False
        tag = "EDGE" if is_edge else f"{n}m"
        match_summary.append(f"S{si}:{tag}")
    draw.text((gap, 28), "  |  ".join(match_summary), fill=(80, 80, 80, 255), font=font)

    col, row = 0, 0
    x_off = gap + col * (cell_w + gap)
    y_off = header_h + row * (cell_h + gap + label_h)

    center_img = piece_imgs.get(pid)
    if center_img:
        scale_c = min((cell_w - 20) / center_img.size[0], (cell_h - 20) / center_img.size[1])
        cw = int(center_img.size[0] * scale_c)
        ch = int(center_img.size[1] * scale_c)
        if cw > 0 and ch > 0:
            resized_c = center_img.resize((cw, ch), Image.LANCZOS)
            cx = x_off + (cell_w - cw) // 2
            cy = y_off + (cell_h - ch) // 2
            canvas.paste(resized_c, (cx, cy), resized_c)

    draw.rectangle([x_off, y_off, x_off + cell_w, y_off + cell_h],
                   outline=(200, 150, 0, 255), width=2)
    draw.text((x_off + 2, y_off + cell_h + 2), f"#{pid} (center)", fill=(200, 150, 0, 255), font=small_font)

    col += 1

    for si, other_pid, other_si, err_val, ld_val in cells:
        if col >= cols:
            col = 0
            row += 1
        x_off = gap + col * (cell_w + gap)
        y_off = header_h + row * (cell_h + gap + label_h)

        cell = _render_pair_cell(
            pid, si, other_pid, other_si,
            piece_data, piece_imgs, cell_w, cell_h
        )
        canvas.paste(cell, (x_off, y_off))

        draw.rectangle([x_off, y_off, x_off + cell_w, y_off + cell_h],
                       outline=(180, 180, 180, 255), width=1)
        label = f"#{other_pid}[s{other_si}] e={err_val:.3f} ld={ld_val:.3f}"
        draw.text((x_off + 2, y_off + cell_h + 2), label, fill=(0, 0, 0, 255), font=small_font)

        col += 1

    canvas.save(output_path)


def _write_report(placed, match_log, piece_edge_info, connectivity, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Connectivity Report\n")
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
                    match_strs = []
                    for m in top3:
                        err_val = m['error'] / 1000.0
                        ld_val = m.get('len_diff', 0) / 1000.0
                        match_strs.append(f"#{m['pid']}[s{m['si']}] e={err_val:.3f} ld={ld_val:.3f}")
                    match_str = ", ".join(match_strs)
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
