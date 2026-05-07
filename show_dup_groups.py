import os, sys, json, itertools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from common.config import VECTOR_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'puzzle_new')
VECTOR_PATH = os.path.join(OUTPUT_DIR, VECTOR_DIR)
BMP_DIR = os.path.join(OUTPUT_DIR, '2_piece_bmps')

PROFILE_N = 50
SIDE_RMSE_THRESHOLD = 0.025
SIDE_LENGTH_RATIO_MIN = 0.85


def load_pieces(vector_path):
    pieces = {}
    vp = Path(vector_path)
    for path in sorted(vp.glob("side_*_0.json")):
        pid = int(path.parts[-1].split('_')[1])
        sides_data = []
        valid = True
        for j in range(4):
            jpath = vp / f'side_{pid}_{j}.json'
            try:
                with open(jpath) as f:
                    data = json.load(f)
                if not data.get('vertices') or len(data['vertices']) < 2:
                    valid = False
                    break
                sides_data.append(data)
            except Exception:
                valid = False
                break
        if valid and len(sides_data) == 4:
            pieces[pid] = sides_data
    return pieces


def classify_side(side_data):
    if side_data['is_edge']:
        return 'F'
    vertices = np.array(side_data['vertices'], dtype=float)
    piece_center = np.array(side_data.get('piece_center', [0, 0]), dtype=float)
    if len(vertices) < 3:
        return 'F'
    p1, p2 = vertices[0], vertices[-1]
    edge_vec = p2 - p1
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1:
        return 'F'
    perp = np.array([-edge_vec[1], edge_vec[0]])
    center_side = np.dot(piece_center - p1, perp)
    mid_verts = vertices[len(vertices) // 4:3 * len(vertices) // 4]
    mid_offsets = np.dot(mid_verts - p1, perp)
    avg_offset = np.mean(mid_offsets)
    if center_side * avg_offset < 0:
        return 'T'
    else:
        return 'B'


def get_side_length(side_data):
    vertices = np.array(side_data['vertices'], dtype=float)
    if len(vertices) < 2:
        return 0
    return np.linalg.norm(vertices[-1] - vertices[0])


def resample_polyline(vertices, n):
    pts = np.array(vertices, dtype=float)
    if len(pts) < 2:
        return np.tile(pts[0] if len(pts) > 0 else np.array([0, 0]), (n, 1))
    dists = [0.0]
    for i in range(1, len(pts)):
        dists.append(dists[-1] + np.linalg.norm(pts[i] - pts[i - 1]))
    total = dists[-1]
    if total < 1e-6:
        return np.tile(pts[0], (n, 1))
    target = np.linspace(0, total, n)
    result = np.zeros((n, 2))
    for i, td in enumerate(target):
        idx = np.searchsorted(dists, td, side='right') - 1
        idx = max(0, min(idx, len(pts) - 2))
        seg = dists[idx + 1] - dists[idx]
        if seg < 1e-6:
            result[i] = pts[idx]
        else:
            t = (td - dists[idx]) / seg
            result[i] = pts[idx] + t * (pts[idx + 1] - pts[idx])
    return result


def get_side_profile(side_data, n=PROFILE_N):
    vertices = np.array(side_data['vertices'], dtype=float)
    if len(vertices) < 2:
        return np.zeros(n)
    p1, p2 = vertices[0], vertices[-1]
    length = np.linalg.norm(p2 - p1)
    if length < 1:
        return np.zeros(n)
    direction = (p2 - p1) / length
    normal = np.array([-direction[1], direction[0]])
    resampled = resample_polyline(vertices, n)
    profile = np.dot(resampled - p1, normal) / length
    return profile


def get_piece_signature(sides_data):
    types = [classify_side(s) for s in sides_data]
    edge_count = sum(1 for t in types if t == 'F')
    return edge_count, types


def normalize_types(types):
    n = len(types)
    rotations = [tuple(types[i:] + types[:i]) for i in range(n)]
    return min(rotations)


def compare_side_profiles(prof_a, prof_b):
    if len(prof_a) != len(prof_b):
        return 999.0
    return np.sqrt(np.mean((prof_a - prof_b) ** 2))


def match_two_pieces(sides_a, sides_b, threshold=SIDE_RMSE_THRESHOLD):
    best_error = float('inf')
    best_rot = -1
    n_matching = 0
    for rot in range(4):
        rot_b = sides_b[rot:] + sides_b[:rot]
        total_err = 0.0
        matched = 0
        ok = True
        for i in range(4):
            if sides_a[i]['is_edge'] and rot_b[i]['is_edge']:
                continue
            len_a = get_side_length(sides_a[i])
            len_b = get_side_length(rot_b[i])
            if len_a > 0 and len_b > 0:
                ratio = min(len_a, len_b) / max(len_a, len_b)
                if ratio < SIDE_LENGTH_RATIO_MIN:
                    ok = False
                    break
            prof_a = get_side_profile(sides_a[i])
            prof_b = get_side_profile(rot_b[i])
            err = compare_side_profiles(prof_a, prof_b)
            if err > threshold:
                ok = False
                break
            total_err += err
            matched += 1
        non_edge_count = sum(1 for i in range(4)
                            if not (sides_a[i]['is_edge'] and rot_b[i]['is_edge']))
        if ok and matched >= non_edge_count and non_edge_count > 0:
            if total_err < best_error:
                best_error = total_err
                best_rot = rot
                n_matching = matched
    return best_rot >= 0, best_rot, best_error, n_matching


def load_bmp(pid):
    bmp_path = os.path.join(BMP_DIR, f'piece_{pid}.bmp')
    if not os.path.exists(bmp_path):
        return None
    img = Image.open(bmp_path).convert('RGBA')
    return img


def main():
    print("Loading pieces and computing duplicate groups...")
    pieces = load_pieces(VECTOR_PATH)
    print(f"Loaded {len(pieces)} pieces")

    signatures = {}
    for pid, sd in pieces.items():
        signatures[pid] = get_piece_signature(sd)

    sig_groups = {}
    for pid, (ec, types) in signatures.items():
        norm_sig = (ec, normalize_types(types))
        sig_groups.setdefault(norm_sig, []).append(pid)

    parent = {pid: pid for pid in pieces}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    dup_pairs = []
    for sig, pids in sig_groups.items():
        if len(pids) < 2:
            continue
        for i, j in itertools.combinations(range(len(pids)), 2):
            pa, pb = pids[i], pids[j]
            is_match, rot, err, n_match = match_two_pieces(pieces[pa], pieces[pb])
            if is_match:
                dup_pairs.append((pa, pb, err, n_match, rot))
                union(pa, pb)

    groups = {}
    for pid in pieces:
        root = find(pid)
        groups.setdefault(root, []).append(pid)

    dup_groups = {root: members for root, members in groups.items() if len(members) > 1}
    dup_groups = dict(sorted(dup_groups.items(), key=lambda x: len(x[1]), reverse=True))

    print(f"Found {len(dup_groups)} duplicate groups, {len(dup_pairs)} pairs")

    if not dup_groups:
        print("No duplicate groups found!")
        return

    THUMB_W = 180
    GAP = 10
    LABEL_H = 28
    GROUP_LABEL_H = 36
    PADDING = 20

    group_images = []

    for idx, (root, members) in enumerate(dup_groups.items()):
        member_imgs = []
        for pid in sorted(members):
            img = load_bmp(pid)
            if img is not None:
                ratio = THUMB_W / img.width
                thumb_h = int(img.height * ratio)
                thumb = img.resize((THUMB_W, thumb_h), Image.LANCZOS)
                member_imgs.append((pid, thumb, thumb_h))

        if not member_imgs:
            continue

        max_h = max(h for _, _, h in member_imgs)
        n = len(member_imgs)
        row_w = PADDING * 2 + n * THUMB_W + (n - 1) * GAP
        row_h = GROUP_LABEL_H + LABEL_H + max_h + PADDING * 2

        canvas = Image.new('RGBA', (row_w, row_h), (40, 40, 40, 255))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

        draw.text((PADDING, PADDING), f"Group {idx + 1}: pieces {sorted(members)}", fill=(255, 255, 100, 255), font=font)

        x_offset = PADDING
        for pid, thumb, thumb_h in member_imgs:
            y_thumb = GROUP_LABEL_H + LABEL_H + (max_h - thumb_h) // 2
            canvas.paste(thumb, (x_offset, y_thumb))
            draw.text((x_offset + THUMB_W // 2, GROUP_LABEL_H + 4), f"#{pid}", fill=(200, 200, 255, 255), font=font_small, anchor='mt')
            x_offset += THUMB_W + GAP

        group_images.append(canvas)

    COLS = 3
    rows = []
    for i in range(0, len(group_images), COLS):
        row_imgs = group_images[i:i + COLS]
        max_row_h = max(img.height for img in row_imgs)
        row_w = sum(img.width for img in row_imgs) + GAP * (len(row_imgs) - 1)
        row_canvas = Image.new('RGBA', (row_w + 40, max_row_h), (30, 30, 30, 255))
        x = 20
        for img in row_imgs:
            row_canvas.paste(img, (x, 0))
            x += img.width + GAP
        rows.append(row_canvas)

    total_w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows) + GAP * (len(rows) - 1)
    final = Image.new('RGBA', (total_w, total_h), (30, 30, 30, 255))
    y = 0
    for row in rows:
        final.paste(row, (0, y))
        y += row.height + GAP

    out_path = os.path.join(OUTPUT_DIR, 'dup_groups_visual.png')
    final.save(out_path)
    print(f"Saved visualization to {out_path}")
    print(f"Image size: {final.width}x{final.height}")

    print(f"\nSummary: {len(dup_groups)} duplicate groups:")
    for idx, (root, members) in enumerate(dup_groups.items()):
        types = [classify_side(s) for s in pieces[members[0]]]
        ec = sum(1 for t in types if t == 'F')
        print(f"  Group {idx + 1}: pieces {sorted(members)} (edges={ec}, pattern={''.join(types)})")


if __name__ == '__main__':
    main()
