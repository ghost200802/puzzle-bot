import os, sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from common.config import VECTOR_DIR, DEDUPED_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'puzzle_new')
VECTOR_PATH = os.path.join(OUTPUT_DIR, VECTOR_DIR)
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
COLOR_DIR = os.path.join(OUTPUT_DIR, '2_piece_colors')

import run_dedup as dedup


def main():
    print("Loading deduped pieces...")
    pieces = dedup.load_pieces(DEDUPED_PATH)
    print(f"Loaded {len(pieces)} unique pieces")

    signatures = {}
    for pid, sd in pieces.items():
        signatures[pid] = dedup.get_piece_signature(sd)

    sig_groups = {}
    for pid, (ec, types) in signatures.items():
        norm_sig = (ec, dedup.normalize_types(types))
        sig_groups.setdefault(norm_sig, []).append(pid)

    sig_groups = dict(sorted(sig_groups.items(), key=lambda x: -len(x[1])))

    print(f"\nSignature groups: {len(sig_groups)}")
    for sig, pids in sig_groups.items():
        ec, types = sig
        print(f"  edges={ec} {''.join(types)}: {len(pids)} pieces -> {sorted(pids)}")

    THUMB_W = 160
    THUMB_H_MAX = 180
    GAP = 8
    GROUP_HEADER_H = 32
    PIECE_LABEL_H = 20
    PADDING = 12
    COLS = 12

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
        font_title = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_small = font
        font_title = font

    group_canvases = []

    for sig_idx, (sig, pids) in enumerate(sig_groups.items()):
        ec, types = sig
        sig_label = f"Sig {sig_idx + 1}: edges={ec} {''.join(types)} ({len(pids)} pcs)"
        pids_sorted = sorted(pids)

        thumbs = []
        for pid in pids_sorted:
            color_path = os.path.join(COLOR_DIR, f'piece_{pid}.png')
            if os.path.exists(color_path):
                img = Image.open(color_path).convert('RGBA')
                ratio = min(THUMB_W / img.width, THUMB_H_MAX / img.height)
                tw = int(img.width * ratio)
                th = int(img.height * ratio)
                thumb = img.resize((tw, th), Image.LANCZOS)
                thumbs.append((pid, thumb, tw, th))
            else:
                dummy = Image.new('RGBA', (THUMB_W, 60), (60, 60, 60, 255))
                thumbs.append((pid, dummy, THUMB_W, 60))

        if not thumbs:
            continue

        n_pcs = len(thumbs)
        rows_needed = (n_pcs + COLS - 1) // COLS

        cell_w = THUMB_W + GAP
        cell_h = THUMB_H_MAX + PIECE_LABEL_H + GAP
        canvas_w = PADDING * 2 + COLS * cell_w
        canvas_h = GROUP_HEADER_H + rows_needed * cell_h + PADDING * 2

        canvas = Image.new('RGBA', (canvas_w, canvas_h), (35, 35, 45, 255))
        draw = ImageDraw.Draw(canvas)

        draw.text((PADDING, PADDING), sig_label, fill=(255, 220, 100, 255), font=font)

        for i, (pid, thumb, tw, th) in enumerate(thumbs):
            row = i // COLS
            col = i % COLS
            x = PADDING + col * cell_w
            y = GROUP_HEADER_H + row * cell_h + PIECE_LABEL_H

            x_center = x + THUMB_W // 2
            draw.text((x_center, y - PIECE_LABEL_H + 2),
                      f"#{pid}", fill=(180, 200, 255, 255), font=font_small, anchor='mt')

            x_off = x + (THUMB_W - tw) // 2
            y_off = y + (THUMB_H_MAX - th) // 2
            canvas.paste(thumb, (x_off, y_off), thumb)

        group_canvases.append(canvas)

    if not group_canvases:
        print("No pieces to display!")
        return

    total_w = max(c.width for c in group_canvases)
    total_h = sum(c.height for c in group_canvases) + GAP * (len(group_canvases) - 1)
    final = Image.new('RGBA', (total_w, total_h), (25, 25, 35, 255))

    y = 0
    for canvas in group_canvases:
        final.paste(canvas, ((total_w - canvas.width) // 2, y))
        y += canvas.height + GAP

    out_path = os.path.join(OUTPUT_DIR, 'sig_groups_visual.png')
    final.save(out_path)
    print(f"\nSaved visualization to {out_path}")
    print(f"Image size: {final.width}x{final.height}")

    print(f"\nSummary: {len(pieces)} unique pieces in {len(sig_groups)} signature groups")
    for sig_idx, (sig, pids) in enumerate(sig_groups.items()):
        ec, types = sig
        print(f"  Sig {sig_idx + 1}: edges={ec} {''.join(types)} -> {sorted(pids)}")


if __name__ == '__main__':
    main()
