import os
import sys
import json
import math

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common import util


def compute_piece_transforms(solution, deduped_dir):
    pw, ph = solution.width, solution.height

    def _angle(p1, p2):
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    placed_pids = set()
    for gy in range(ph):
        for gx in range(pw):
            cell = solution.get(gx, gy)
            if cell is not None:
                placed_pids.add(cell[0])

    piece_sides_cache = {}
    for pid in placed_pids:
        sides = []
        for i in range(4):
            json_path = os.path.join(deduped_dir, f'side_{pid}_{i}.json')
            if not os.path.exists(json_path):
                sides.append(None)
                continue
            with open(json_path, 'r') as f:
                sides.append(json.load(f))
        piece_sides_cache[pid] = sides

    x, y = 0, 0
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    direction = directions[0]

    placed_pieces = {}
    piece_transforms = {}
    spiral_width, spiral_height = 0, 0

    for _ in range(pw * ph):
        cell = solution.get(x, y)
        if cell is None:
            next_x, next_y = x + direction[0], y + direction[1]
            if next_x < 0 or next_x >= pw or next_y < 0 or next_y >= ph or placed_pieces.get((next_x, next_y)):
                direction = directions[(directions.index(direction) + 1) % 4]
            x, y = x + direction[0], y + direction[1]
            continue

        piece_id, fits, orientation = cell
        sides = piece_sides_cache.get(piece_id)
        if sides is None:
            next_x, next_y = x + direction[0], y + direction[1]
            if next_x < 0 or next_x >= pw or next_y < 0 or next_y >= ph or placed_pieces.get((next_x, next_y)):
                direction = directions[(directions.index(direction) + 1) % 4]
            x, y = x + direction[0], y + direction[1]
            continue

        if y == 0:
            placed_pieces[(x, y - 1)] = [[], [], [(100, 0), (0, 0)], []]
        if x == 0:
            placed_pieces[(x - 1, y)] = [[], [(0, 0), (0, 100)], [], []]
        if x == pw - 1:
            placed_pieces[(x + 1, y)] = [[], [], [], [(spiral_width, 100), (spiral_width, 0)]]
        if y == ph - 1:
            placed_pieces[(x, y + 1)] = [[(0, spiral_height), (100, spiral_height)], [], [], []]

        neighbor_above = placed_pieces.get((x, y - 1), [[], [], [], []])[2]
        neighbor_right = placed_pieces.get((x + 1, y), [[], [], [], []])[3]
        neighbor_below = placed_pieces.get((x, y + 1), [[], [], [], []])[0]
        neighbor_left = placed_pieces.get((x - 1, y), [[], [], [], []])[1]

        neighbor_above_angle = _angle(neighbor_above[0], neighbor_above[-1]) % (2 * math.pi) if neighbor_above else None
        neighbor_right_angle = _angle(neighbor_right[0], neighbor_right[-1]) % (2 * math.pi) if neighbor_right else None
        neighbor_below_angle = _angle(neighbor_below[0], neighbor_below[-1]) % (2 * math.pi) if neighbor_below else None
        neighbor_left_angle = _angle(neighbor_left[0], neighbor_left[-1]) % (2 * math.pi) if neighbor_left else None

        side_angles = []
        for side in sides:
            if side is None:
                side_angles.append(None)
                continue
            verts = side['vertices']
            side_angles.append(_angle(verts[0], verts[-1]) % (2 * math.pi))

        new_sides = util.rotate_list([0, 1, 2, 3], -orientation)
        new_top, new_right, new_bottom, new_left = new_sides

        rotations = []
        if neighbor_above_angle is not None and side_angles[new_top] is not None:
            rotations.append(neighbor_above_angle - side_angles[new_top] - math.pi)
        if neighbor_right_angle is not None and side_angles[new_right] is not None:
            rotations.append(neighbor_right_angle - side_angles[new_right] - math.pi)
        if neighbor_below_angle is not None and side_angles[new_bottom] is not None:
            rotations.append(neighbor_below_angle - side_angles[new_bottom] - math.pi)
        if neighbor_left_angle is not None and side_angles[new_left] is not None:
            rotations.append(neighbor_left_angle - side_angles[new_left] - math.pi)

        if rotations:
            rotation = util.average_angles(rotations)
        elif x == 0 and y == 0:
            rotation = -side_angles[new_top] if side_angles[new_top] is not None else 0
        else:
            rotation = 0

        ic = tuple(sides[0]['incenter']) if sides[0] else (0, 0)
        rotated_sides = []
        for side in sides:
            if side is None:
                rotated_sides.append([])
                continue
            rotated_sides.append([util.rotate(pt, ic, rotation) for pt in side['vertices']])

        if y == 0:
            if neighbor_left:
                origin_x = neighbor_left[0][0]
            else:
                origin_x = 0
            origin_y = 0
            w = rotated_sides[new_top][-1][0] - rotated_sides[new_top][0][0]
            neighbor_above = [(origin_x + w, origin_y), (origin_x, origin_y)]
        if x == pw - 1:
            if y == 0:
                piece_width = rotated_sides[new_top][-1][0] - rotated_sides[new_top][0][0]
                if neighbor_left:
                    spiral_width = neighbor_left[0][0] + piece_width
                else:
                    spiral_width = piece_width
            origin_x = spiral_width
            if neighbor_above:
                origin_y = neighbor_above[0][1]
            else:
                origin_y = 0
            h = rotated_sides[new_right][-1][1] - rotated_sides[new_right][0][1]
            neighbor_right = [(origin_x, origin_y + h), (origin_x, origin_y)]
        if y == ph - 1:
            if x == pw - 1:
                piece_height = rotated_sides[new_right][-1][1] - rotated_sides[new_right][0][1]
                if neighbor_above:
                    spiral_height = neighbor_above[0][1] + piece_height
                else:
                    spiral_height = piece_height
            if neighbor_right:
                origin_x = neighbor_right[0][0]
            else:
                origin_x = spiral_width
            origin_y = spiral_height
            w = rotated_sides[new_bottom][-1][0] - rotated_sides[new_bottom][0][0]
            neighbor_below = [(origin_x - w, origin_y), (origin_x, origin_y)]
        if x == 0 and y != 0:
            if y == ph - 1:
                origin_x = 0
                origin_y = spiral_height
                w = rotated_sides[new_bottom][0][0] - rotated_sides[new_bottom][-1][0]
            else:
                origin_x = 0
                if neighbor_below:
                    origin_y = neighbor_below[0][1]
                else:
                    origin_y = 0
            h = rotated_sides[new_left][-1][1] - rotated_sides[new_left][0][1]
            neighbor_left = [(origin_x, origin_y - h), (origin_x, origin_y)]

        samples = []
        if neighbor_above:
            samples.append(util.subtract(neighbor_above[-1], rotated_sides[new_top][0]))
        if neighbor_right:
            samples.append(util.subtract(neighbor_right[-1], rotated_sides[new_right][0]))
        if neighbor_below:
            samples.append(util.subtract(neighbor_below[-1], rotated_sides[new_bottom][0]))
        if neighbor_left:
            samples.append(util.subtract(neighbor_left[-1], rotated_sides[new_left][0]))

        if x == 0 and y == 0:
            translation = util.subtract((0, 0), rotated_sides[new_top][0])
        elif samples:
            translation = util.multimidpoint(samples)
        else:
            translation = (0, 0)

        piece_transforms[piece_id] = (rotation, translation, ic)

        translated_rotated_sides = [util.translate_polyline(side, translation) for side in rotated_sides]
        placed_pieces[(x, y)] = [
            translated_rotated_sides[new_top],
            translated_rotated_sides[new_right],
            translated_rotated_sides[new_bottom],
            translated_rotated_sides[new_left]
        ]

        next_x, next_y = x + direction[0], y + direction[1]
        if next_x < 0 or next_x >= pw or next_y < 0 or next_y >= ph or placed_pieces.get((next_x, next_y)):
            direction = directions[(directions.index(direction) + 1) % 4]
        x, y = x + direction[0], y + direction[1]

    if not piece_transforms:
        return None

    all_pts = []
    for pid, (rot, trans, ic) in piece_transforms.items():
        sides = piece_sides_cache[pid]
        for side in sides:
            if side is None:
                continue
            for v in side['vertices']:
                rv = util.rotate(v, ic, rot)
                tv = (rv[0] + trans[0], rv[1] + trans[1])
                all_pts.append(tv)
    if not all_pts:
        return {}, {}, {}

    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_y = max(p[1] for p in all_pts)

    canvas_info = {
        'min_x': min_x, 'min_y': min_y,
        'max_x': max_x, 'max_y': max_y,
    }

    return piece_transforms, piece_sides_cache, canvas_info


def generate_assembly_png(solution, deduped_dir, output_dir, output_path):
    from PIL import Image, ImageDraw, ImageFont

    pw, ph = solution.width, solution.height

    piece_transforms, piece_sides_cache, canvas_info = compute_piece_transforms(solution, deduped_dir)
    if not piece_transforms:
        return

    color_dir = os.path.join(output_dir, '2_piece_colors')
    piece_imgs = {}
    for pid in piece_transforms:
        img_path = os.path.join(color_dir, f'piece_{pid}.png')
        if os.path.exists(img_path):
            piece_imgs[pid] = Image.open(img_path).convert('RGBA')

    min_x = canvas_info['min_x']
    min_y = canvas_info['min_y']
    max_x = canvas_info['max_x']
    max_y = canvas_info['max_y']

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

    canvas = Image.new('RGBA', (max(canvas_w, 100), max(canvas_h, 100)), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arialbd.ttf", max(18, min(48, int(36 * scale))))
        title_font = ImageFont.truetype("arial.ttf", max(12, min(28, int(24 * scale))))
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", max(18, min(48, int(36 * scale))))
            title_font = font
        except Exception:
            font = ImageFont.load_default()
            title_font = font

    title = f"Assembly ({solution.placed_count}/{pw * ph} pieces)"
    draw.text((10, 5), title, fill=(0, 0, 0, 255), font=title_font)

    def to_canvas(wx, wy):
        cx = (wx - min_x + margin) * scale
        cy = (wy - min_y + margin) * scale + header_h
        return (cx, cy)

    for pid, (rotation, translation, ic) in piece_transforms.items():
        if pid not in piece_imgs:
            continue

        piece_img = piece_imgs[pid]
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        w_img, h_img = piece_img.size

        corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
        img_pts = []
        for cx, cy in corners:
            dx = cx - ic[0]
            dy = cy - ic[1]
            ox = dx * cos_r - dy * sin_r + ic[0] + translation[0]
            oy = dx * sin_r + dy * cos_r + ic[1] + translation[1]
            img_pts.append((ox, oy))

        img_all_x = [p[0] for p in img_pts]
        img_all_y = [p[1] for p in img_pts]
        img_min_x = min(img_all_x)
        img_min_y = min(img_all_y)
        img_max_x = max(img_all_x)
        img_max_y = max(img_all_y)

        out_w = int(math.ceil((img_max_x - img_min_x) * scale)) + 2
        out_h = int(math.ceil((img_max_y - img_min_y) * scale)) + 2

        if out_w <= 0 or out_h <= 0:
            continue

        cos_neg = math.cos(-rotation)
        sin_neg = math.sin(-rotation)
        sm_x = (img_min_x - ic[0] - translation[0]) * scale
        sm_y = (img_min_y - ic[1] - translation[1]) * scale

        a = cos_neg * scale
        b = -sin_neg * scale
        c = cos_neg * sm_x - sin_neg * sm_y + ic[0]
        d = sin_neg * scale
        e = cos_neg * scale
        f = sin_neg * sm_x + cos_neg * sm_y + ic[1]

        try:
            transformed = piece_img.transform(
                (out_w, out_h), Image.AFFINE,
                (a, b, c, d, e, f),
                resample=Image.BICUBIC,
            )
            paste_x = int((img_min_x - min_x + margin) * scale)
            paste_y = int((img_min_y - min_y + margin) * scale + header_h)
            canvas.paste(transformed, (paste_x, paste_y), transformed)
        except Exception:
            pass

    for pid, (rotation, translation, ic) in piece_transforms.items():
        sides = piece_sides_cache.get(pid)
        if not sides or sides[0] is None:
            continue
        orig_ic = tuple(sides[0]['incenter'])
        rv = util.rotate(orig_ic, ic, rotation)
        tv = (rv[0] + translation[0], rv[1] + translation[1])
        cx, cy = to_canvas(tv[0], tv[1])
        label = str(pid)
        stroke_w = max(2, int(3 * scale))
        draw.text((cx, cy), label, fill=(0, 180, 0, 255), font=font, anchor="mm",
                  stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255))

    canvas.save(output_path)
    print(f"PNG saved: {output_path}")
    return canvas_info
