import os, sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from common.config import VECTOR_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'puzzle_new')
VECTOR_PATH = os.path.join(OUTPUT_DIR, VECTOR_DIR)
COLOR_DIR = os.path.join(OUTPUT_DIR, '2_piece_colors')
META_PATH = os.path.join(OUTPUT_DIR, 'dedup_match_meta.json')

import run_dedup as dedup


def load_piece_image(pid):
    color_path = os.path.join(COLOR_DIR, f'piece_{pid}.png')
    if os.path.exists(color_path):
        return Image.open(color_path).convert('RGBA')
    return None


def get_pair_meta(pid_a, pid_b, meta_dict):
    key = f"{min(pid_a, pid_b)}_{max(pid_a, pid_b)}"
    return meta_dict.get(key)


def find_group_meta(members, meta_dict):
    result = []
    members_sorted = sorted(members)
    for i in range(len(members_sorted)):
        for j in range(i + 1, len(members_sorted)):
            m = get_pair_meta(members_sorted[i], members_sorted[j], meta_dict)
            if m:
                result.append((members_sorted[i], members_sorted[j], m))
    return result


def main():
    print("Loading dedup match metadata...")
    with open(META_PATH) as f:
        meta_dict = json.load(f)

    pieces = dedup.load_pieces(VECTOR_PATH)
    print(f"Loaded {len(pieces)} pieces, {len(meta_dict)} match records")

    confirmed_pairs = []
    for key, m in meta_dict.items():
        if m['confirmed']:
            pa, pb = key.split('_')
            confirmed_pairs.append((int(pa), int(pb), m))

    print(f"Confirmed pairs: {len(confirmed_pairs)}")

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

    for pa, pb, m in confirmed_pairs:
        union(pa, pb)

    groups = {}
    for pid in pieces:
        root = find(pid)
        groups.setdefault(root, []).append(pid)

    dup_groups = {root: members for root, members in groups.items() if len(members) > 1}
    dup_groups = dict(sorted(dup_groups.items(), key=lambda x: len(x[1]), reverse=True))

    print(f"Duplicate groups: {len(dup_groups)}")

    if not dup_groups:
        print("No duplicate groups found!")
        return

    THUMB_W = 200
    GAP = 12
    GROUP_LABEL_H = 56
    LABEL_H = 24
    PADDING = 16

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 13)
        font_tiny = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_small = font
        font_tiny = font

    group_images = []

    for idx, (root, members) in enumerate(dup_groups.items()):
        pair_metas = find_group_meta(members, meta_dict)
        member_imgs = []
        for pid in sorted(members):
            img = load_piece_image(pid)
            if img is not None:
                ratio = THUMB_W / img.width
                thumb_h = int(img.height * ratio)
                thumb = img.resize((THUMB_W, thumb_h), Image.LANCZOS)
                member_imgs.append((pid, thumb, thumb_h))

        if not member_imgs:
            continue

        max_h = max(h for _, _, h in member_imgs)
        n = len(member_imgs)

        meta_lines = []
        for pa, pb, m in pair_metas:
            stage_label = f"S{m['stage']}"
            meta_lines.append(
                f"#{pa}~#{pb}: RMSE={m['total_rmse']:.4f}(<{m['rmse_thresh']}) "
                f"NCC={m['ncc']:.3f}(>={m['ncc_thresh']}) rot={m['rot']} [{stage_label}]"
            )
        meta_h = max(len(meta_lines) * 16, 16)

        row_w = PADDING * 2 + n * THUMB_W + (n - 1) * GAP
        row_h = GROUP_LABEL_H + LABEL_H + max_h + meta_h + PADDING * 2

        canvas = Image.new('RGBA', (row_w, row_h), (40, 40, 40, 255))
        draw = ImageDraw.Draw(canvas)

        draw.text((PADDING, PADDING),
                  f"Group {idx + 1}: pieces {sorted(members)}",
                  fill=(255, 255, 100, 255), font=font)

        x_offset = PADDING
        for pid, thumb, thumb_h in member_imgs:
            y_thumb = GROUP_LABEL_H + LABEL_H + (max_h - thumb_h) // 2
            canvas.paste(thumb, (x_offset, y_thumb), thumb)
            draw.text((x_offset + THUMB_W // 2, GROUP_LABEL_H + 4),
                      f"#{pid}", fill=(200, 200, 255, 255), font=font_small, anchor='mt')
            x_offset += THUMB_W + GAP

        y_meta = GROUP_LABEL_H + LABEL_H + max_h + 4
        for line in meta_lines:
            draw.text((PADDING, y_meta), line, fill=(150, 255, 150, 255), font=font_tiny)
            y_meta += 16

        group_images.append(canvas)

    COLS = 2
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
    print(f"\nSaved visualization to {out_path}")
    print(f"Image size: {final.width}x{final.height}")

    print(f"\nSummary: {len(dup_groups)} duplicate groups (texture verified)")
    for idx, (root, members) in enumerate(dup_groups.items()):
        types = [dedup.classify_side(s) for s in pieces[members[0]]]
        ec = sum(1 for t in types if t == 'F')
        pair_metas = find_group_meta(members, meta_dict)
        ncc_vals = [m['ncc'] for _, _, m in pair_metas]
        ncc_str = ', '.join(f"{n:.3f}" for n in ncc_vals)
        print(f"  Group {idx + 1}: {sorted(members)} edges={ec} {''.join(types)} NCC=[{ncc_str}]")


if __name__ == '__main__':
    main()
