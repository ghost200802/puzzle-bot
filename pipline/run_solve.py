import os
import sys
import json
import math
from collections import Counter

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR
from common import board, output as board_output
from common.board import build_from_corner, Board

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)
SOLUTION_PATH = os.path.join(OUTPUT_DIR, SOLUTION_DIR)

NCC_PRIORITY_WEIGHT = 1000.0


def load_ncc_lookup(report_path):
    if not os.path.exists(report_path):
        return {}
    with open(report_path, 'r') as f:
        report = json.load(f)
    lookup = {}
    for pid_str, sides in report.items():
        pid = int(pid_str)
        for si, matches in enumerate(sides):
            for m in matches:
                key = (pid, si, m['pid'], m['si'])
                lookup[key] = {
                    'ncc': m['ncc'],
                    'reject': m.get('reject', False),
                }
    return lookup


def load_connectivity_raw(connectivity_file):
    with open(connectivity_file, 'r') as f:
        connectivity = json.load(f)
    ps = {}
    for pid_str, fits_list in connectivity.items():
        pid = int(pid_str)
        ps[pid] = [[], [], [], []]
        for i in range(4):
            for m in fits_list[i]:
                ps[pid][i].append((m['pid'], m['si'], m['error']))
    return ps


def build_ncc_ps(ps_raw, ncc_lookup):
    ps = {}
    ncc_count = 0
    fb_count = 0
    for pid, sides in ps_raw.items():
        ps[pid] = [[], [], [], []]
        for si in range(4):
            ncc_list = []
            fb_list = []
            for other_pid, other_si, error in sides[si]:
                key = (pid, si, other_pid, other_si)
                rev_key = (other_pid, other_si, pid, si)
                info = ncc_lookup.get(key) or ncc_lookup.get(rev_key)
                if info and not info['reject'] and info['ncc'] > 0:
                    composite = error / (info['ncc'] * NCC_PRIORITY_WEIGHT)
                    ncc_list.append((other_pid, other_si, composite))
                    ncc_count += 1
                else:
                    fb_list.append((other_pid, other_si, error))
                    fb_count += 1
            ncc_list.sort(key=lambda x: x[2])
            fb_list.sort(key=lambda x: x[2])
            ps[pid][si] = ncc_list + fb_list
    print(f"  NCC composite: {ncc_count}, Fallback: {fb_count}")
    return ps


def trace_border_edge(start_pid, start_out_side, ps, piece_edge_info):
    count = 1
    current_pid = start_pid
    out_side = start_out_side
    visited = {start_pid}
    for _ in range(200):
        fits = ps.get(current_pid, [[] for _ in range(4)])
        side_fits = fits[out_side] if out_side < len(fits) else []
        if not side_fits:
            break
        edge_candidates = []
        for other_pid, other_side, error in side_fits:
            if other_pid in visited:
                continue
            nf = piece_edge_info.get(other_pid, [False] * 4)
            ec = sum(1 for f in nf if f)
            if ec >= 1:
                edge_candidates.append((other_pid, other_side, error, ec))
        if not edge_candidates:
            break
        edge_candidates.sort(key=lambda x: x[2])
        next_pid, in_side, _, ec = edge_candidates[0]
        visited.add(next_pid)
        count += 1
        if ec >= 2:
            break
        nf = piece_edge_info.get(next_pid, [False] * 4)
        out_side = None
        for i in range(4):
            if i == in_side:
                continue
            if nf[i]:
                continue
            if len(ps.get(next_pid, [[] for _ in range(4)])[i]) > 0:
                out_side = i
                break
        if out_side is None:
            break
        current_pid = next_pid
    return count


def determine_dimensions(ps, corners, piece_edge_info):
    print("\nDetermining dimensions by tracing border edges from corners...")
    results = []
    for c in corners:
        ef = piece_edge_info.get(c, [False] * 4)
        flat = [i for i, f in enumerate(ef) if f]
        non_flat = [i for i in range(4) if i not in flat]
        if len(non_flat) != 2:
            continue
        d1 = trace_border_edge(c, non_flat[0], ps, piece_edge_info)
        d2 = trace_border_edge(c, non_flat[1], ps, piece_edge_info)
        results.append((c, d1, d2))
        print(f"  Corner {c}: side[{non_flat[0]}]={d1} pcs, side[{non_flat[1]}]={d2} pcs -> {d1}x{d2}")

    if not results:
        return None, None

    dim_pairs = Counter()
    for c, d1, d2 in results:
        dim_pairs[(d1, d2)] += 1
        dim_pairs[(d2, d1)] += 1

    best_pair, count = dim_pairs.most_common(1)[0]
    w, h = best_pair
    print(f"\n  Most common dimension pair: {w} x {h} (from {count} observations)")
    return w, h


def generate_assembly_png(solution, piece_edge_info, output_path):
    from PIL import Image, ImageDraw, ImageFont
    from common import util

    color_dir = os.path.join(OUTPUT_DIR, '2_piece_colors')
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
    piece_imgs = {}
    for pid in placed_pids:
        sides = []
        for i in range(4):
            json_path = os.path.join(DEDUPED_PATH, f'side_{pid}_{i}.json')
            if not os.path.exists(json_path):
                sides.append(None)
                continue
            with open(json_path, 'r') as f:
                sides.append(json.load(f))
        piece_sides_cache[pid] = sides
        img_path = os.path.join(color_dir, f'piece_{pid}.png')
        if os.path.exists(img_path):
            piece_imgs[pid] = Image.open(img_path).convert('RGBA')

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
        return

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

    canvas = Image.new('RGBA', (max(canvas_w, 100), max(canvas_h, 100)), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", max(10, min(24, int(20 * scale))))
        title_font = ImageFont.truetype("arial.ttf", max(12, min(28, int(24 * scale))))
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
        draw.text((cx, cy), str(pid), fill=(0, 0, 0, 255), font=font, anchor="mm")

    canvas.save(output_path)
    print(f"PNG saved: {output_path}")


def main():
    print("=" * 60)
    print("Puzzle Solve (NCC Priority + Spiral Assembly)")
    print("=" * 60)

    connectivity_file = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
    edge_info_file = os.path.join(CONNECTIVITY_PATH, 'piece_edge_info.json')
    ncc_report_file = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')

    with open(connectivity_file, 'r') as f:
        connectivity_raw = json.load(f)
    with open(edge_info_file, 'r') as f:
        piece_edge_info = {int(k): v for k, v in json.load(f).items()}

    ncc_lookup = load_ncc_lookup(ncc_report_file)
    print(f"NCC lookup: {len(ncc_lookup)} entries")

    ps_raw = load_connectivity_raw(connectivity_file)
    n_pieces = len(ps_raw)

    corners = []
    for pid in ps_raw:
        if pid in piece_edge_info:
            edge_count = sum(1 for f in piece_edge_info[pid] if f)
        else:
            edge_count = sum(1 for f in ps_raw[pid] if len(f) == 0)
        if edge_count >= 2:
            corners.append(pid)

    print(f"Pieces: {n_pieces}")
    print(f"Corners: {corners}")

    os.makedirs(SOLUTION_PATH, exist_ok=True)

    print("\n--- Building NCC-enhanced connectivity ---")
    ps_ncc = build_ncc_ps(ps_raw, ncc_lookup)

    w, h = determine_dimensions(ps_raw, corners, piece_edge_info)
    if w is None or w < 2 or h < 2:
        print("Failed to determine dimensions.")
        return

    edge_length = 2 * (w + h) - 4

    import common.board as board_mod
    board_mod.MAX_ITERATIONS_TO_FIND_BORDER = 50000
    board_mod.MAX_ITERATIONS = 300000000
    print(f"MAX_ITERATIONS_TO_FIND_BORDER: {board_mod.MAX_ITERATIONS_TO_FIND_BORDER}")

    print(f"\n{'=' * 60}")
    print(f"Solving {w}x{h} ({w * h} pieces, {n_pieces} available)")
    print(f"{'=' * 60}")

    corners_sorted = sorted(
        corners,
        key=lambda c: sum(len(fits) for fits in ps_ncc[c]),
        reverse=True
    )

    best_solution = None
    best_count = 0

    for i, corner_id in enumerate(corners_sorted):
        print(f"\n  Trying corner {i}: piece {corner_id}...")
        solution = build_from_corner(
            ps_ncc, start_piece_id=corner_id,
            edge_length=edge_length,
            puzzle_width=pw if (pw := w) else None,
            puzzle_height=h
        )
        if solution.placed_count > best_count:
            best_solution = solution
            best_count = solution.placed_count
        if solution.placed_count == w * h:
            break

    if best_solution is None:
        print("\nNo solution at all.")
        return

    is_full = best_solution.placed_count == w * h
    print(f"\n{'=' * 60}")
    print(f"{'FULL SOLUTION' if is_full else 'PARTIAL SOLUTION'}: "
          f"{best_solution.placed_count}/{w * h}")
    print(f"{'=' * 60}")
    print(best_solution)

    print("\n--- Generating outputs ---")
    board_output.generate_solution_grid(best_solution, SOLUTION_PATH)
    board_output.generate_solution_svg(best_solution, DEDUPED_PATH, SOLUTION_PATH)
    generate_assembly_png(best_solution, piece_edge_info,
                          os.path.join(SOLUTION_PATH, 'assembly.png'))

    eval_result = board.evaluate_solution(best_solution)
    print(f"\nSolution evaluation:")
    print(f"  Coverage: {eval_result['coverage']:.1%}")
    print(f"  Matched edges: {eval_result['matched_edges']}/{eval_result['total_possible_edges']}")
    print(f"  Match quality: {eval_result['match_quality']:.1%}")

    print(f"\nAll outputs saved to {SOLUTION_PATH}/")


if __name__ == '__main__':
    main()
