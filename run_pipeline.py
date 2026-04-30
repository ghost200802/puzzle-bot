#!/usr/bin/env python3
"""
Full end-to-end pipeline for input/puzzles/1.png

Adaptive preprocessing:
  1. Detect pieces at original resolution to determine typical piece size
  2. Calculate scale factor to reach project-configured target size
  3. For each piece: crop original -> scale up (INTER_CUBIC) -> smooth threshold
     This preserves edge detail (jigsaw tabs/blanks) at high resolution.
"""
import os
import sys
import json
import time
import shutil
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from common.config import (
    VECTOR_DIR, DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR,
    PHONE_TARGET_PIECE_SIZE,
)
from common.find_islands import save_island_as_bmp
from common import vector, dedupe

INPUT_IMAGE = 'input/puzzles/1.png'
OUTPUT_DIR = 'output/puzzle_run'
TARGET_PIECE_SIZE = 1000


def segment_and_prepare(image_path, target_size=TARGET_PIECE_SIZE):
    """
    Adaptive segmentation pipeline:
      Pass 1 - Detect pieces at original res, measure typical size
      Pass 2 - Crop original grayscale, scale to target, smooth threshold

    Returns: (pieces, n_cols, n_rows)
    """
    img = Image.open(image_path)
    rgba = np.array(img)
    rgb = rgba[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # --- Pass 1: detect pieces, measure typical piece size ---
    binary_detect = (gray > 50).astype(np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_detect = cv2.morphologyEx(binary_detect, cv2.MORPH_CLOSE, kernel3)
    binary_detect = cv2.morphologyEx(binary_detect, cv2.MORPH_OPEN, kernel3)

    labeled, num = ndlabel(binary_detect)

    raw_pieces = []
    for i in range(1, num + 1):
        area = np.sum(labeled == i)
        if area < 1500:
            continue
        ys, xs = np.where(labeled == i)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        raw_pieces.append({
            'label': i,
            'area': area,
            'bbox': (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
            'size': (w, h),
            'max_dim': max(w, h),
        })

    if not raw_pieces:
        return [], 0, 0

    # Measure typical piece size (median of max_dim)
    typical_max_dim = int(np.median([p['max_dim'] for p in raw_pieces]))
    scale_factor = target_size / typical_max_dim

    print(f"Detected {len(raw_pieces)} pieces")
    print(f"Typical piece size: {typical_max_dim}px (max dimension)")
    print(f"Target size: {target_size}px -> scale factor: {scale_factor:.1f}x")

    # --- Pass 2: crop original, scale up, smooth threshold ---
    pieces = []
    pad = max(3, int(typical_max_dim * 0.05))

    for rp in raw_pieces:
        x0, y0, x1, y1 = rp['bbox']
        py0 = max(0, y0 - pad)
        py1 = min(gray.shape[0], y1 + pad)
        px0 = max(0, x0 - pad)
        px1 = min(gray.shape[1], x1 + pad)

        orig_crop = gray[py0:py1, px0:px1]
        h, w = orig_crop.shape

        new_w = max(int(w * scale_factor), 10)
        new_h = max(int(h * scale_factor), 10)

        # Scale up with cubic interpolation (preserves edge gradients)
        scaled = cv2.resize(orig_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Smooth threshold at high resolution
        blur_k = max(3, int(scale_factor * 0.8))
        if blur_k % 2 == 0:
            blur_k += 1
        blurred = cv2.GaussianBlur(scaled, (blur_k, blur_k), 0)

        # Threshold: use Otsu on the blurred image for optimal separation
        _, smooth_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        smooth_binary = (smooth_binary > 127).astype(np.uint8)

        # Clean small artifacts
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smooth_binary = cv2.morphologyEx(smooth_binary, cv2.MORPH_OPEN, kern)
        smooth_binary = cv2.morphologyEx(smooth_binary, cv2.MORPH_CLOSE, kern)

        # Keep only the largest connected component (the piece itself)
        lbl, n = ndlabel(smooth_binary)
        if n > 1:
            best = max(range(1, n + 1), key=lambda j: np.sum(lbl == j))
            smooth_binary = (lbl == best).astype(np.uint8)

        pieces.append({
            'id': len(pieces) + 1,
            'binary': smooth_binary,
            'origin': (int(px0), int(py0)),
            'centroid': ((px0 + px1) / 2.0, (py0 + py1) / 2.0),
            'area': rp['area'],
            'target_size': (new_w, new_h),
        })

    # Determine grid dimensions
    def cluster_positions(positions, min_gap=30):
        sp = sorted(positions)
        clusters = [[sp[0]]]
        for pos in sp[1:]:
            if pos - clusters[-1][-1] > min_gap:
                clusters.append([])
            clusters[-1].append(pos)
        return [np.mean(c) for c in clusters]

    xs = sorted(p['centroid'][0] for p in pieces)
    ys = sorted(p['centroid'][1] for p in pieces)
    n_cols = len(cluster_positions(xs))
    n_rows = len(cluster_positions(ys))

    print(f"Grid: {n_cols} cols x {n_rows} rows = {n_cols * n_rows}")

    return pieces, n_cols, n_rows


def step0_segment():
    print("\n" + "=" * 60)
    print("STEP 0: Adaptive segmentation")
    print("=" * 60)
    return segment_and_prepare(INPUT_IMAGE)


def step1_save_pieces(pieces, output_dir):
    print("\n" + "=" * 60)
    print("STEP 1: Saving pieces as BMP")
    print("=" * 60)

    vector_dir = os.path.join(output_dir, VECTOR_DIR)
    os.makedirs(vector_dir, exist_ok=True)

    saved = []
    for p in pieces:
        pid = p['id']
        h, w = p['binary'].shape
        bmp_path = os.path.join(vector_dir, f'piece_{pid}.bmp')
        save_island_as_bmp(p['binary'], bmp_path)

        metadata = {
            'original_photo_name': '1.png',
            'photo_space_origin': p['origin'],
            'photo_space_centroid': [w // 2, h // 2],
            'photo_width': w,
            'photo_height': h,
            'is_complete': True,
        }
        saved.append({'pid': pid, 'bmp_path': bmp_path, 'metadata': metadata})

    print(f"Saved {len(saved)} pieces")
    return saved


def step2_vectorize(saved_pieces, output_dir):
    print("\n" + "=" * 60)
    print("STEP 2: Vectorizing pieces")
    print("=" * 60)

    vector_dir = os.path.join(output_dir, VECTOR_DIR)
    success = 0
    failed = 0
    for s in saved_pieces:
        args = [s['bmp_path'], s['pid'], vector_dir, s['metadata'],
                (0, 0), 1.0, False]
        try:
            vector.load_and_vectorize(args)
            success += 1
        except Exception as e:
            print(f"  Error vectorizing piece {s['pid']}: {e}")
            failed += 1

    print(f"Vectorized: {success} success, {failed} failed")
    return success


def step3_deduplicate(output_dir):
    print("\n" + "=" * 60)
    print("STEP 3: Deduplicating")
    print("=" * 60)

    vector_dir = os.path.join(output_dir, VECTOR_DIR)
    deduped_dir = os.path.join(output_dir, DEDUPED_DIR)
    os.makedirs(deduped_dir, exist_ok=True)

    count = dedupe.deduplicate_phone(vector_dir, deduped_dir)
    print(f"Deduplicated: {count} unique pieces")
    return count


def step4_connectivity(output_dir):
    print("\n" + "=" * 60)
    print("STEP 4: Building connectivity")
    print("=" * 60)

    deduped_dir = os.path.join(output_dir, DEDUPED_DIR)
    conn_dir = os.path.join(output_dir, CONNECTIVITY_DIR)
    os.makedirs(conn_dir, exist_ok=True)

    from common import connect
    connectivity = connect.build(deduped_dir, conn_dir)

    corners = 0
    edges = 0
    for piece_id, piece_data in connectivity.items():
        fits = piece_data if isinstance(piece_data, list) else piece_data.get('fits', piece_data)
        edge_sides = 0
        if isinstance(fits, (list, tuple)):
            for side_fits in fits:
                if len(side_fits) == 0:
                    edge_sides += 1
        if edge_sides >= 2:
            corners += 1
        elif edge_sides >= 1:
            edges += 1

    print(f"Connectivity: {len(connectivity)} pieces, {corners} corners, {edges} edges")
    return connectivity


def step5_solve(output_dir, puzzle_width, puzzle_height):
    print("\n" + "=" * 60)
    print("STEP 5: Solving puzzle")
    print("=" * 60)

    conn_dir = os.path.join(output_dir, CONNECTIVITY_DIR)
    sol_dir = os.path.join(output_dir, SOLUTION_DIR)
    os.makedirs(sol_dir, exist_ok=True)

    from common import board as board_mod, output as output_mod

    print(f"Solving {puzzle_width}x{puzzle_height} puzzle...")

    try:
        puzzle = board_mod.build(
            input_path=conn_dir,
            output_path=sol_dir,
            puzzle_width=puzzle_width,
            puzzle_height=puzzle_height,
        )
        print("\n*** PUZZLE SOLVED! ***")
        print(puzzle)

        output_mod.generate_solution_grid(puzzle, sol_dir)
        output_mod.print_solution_summary(puzzle)
        return puzzle

    except Exception as e:
        print(f"\nSolve failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    start_time = time.time()

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pieces, n_cols, n_rows = step0_segment()
    if not pieces:
        print("No pieces found!")
        return

    saved = step1_save_pieces(pieces, OUTPUT_DIR)

    success_count = step2_vectorize(saved, OUTPUT_DIR)
    if success_count == 0:
        print("\nNo pieces vectorized. Cannot continue.")
        return

    deduped_count = step3_deduplicate(OUTPUT_DIR)
    if deduped_count == 0:
        print("\nNo pieces after deduplication. Cannot continue.")
        return

    connectivity = step4_connectivity(OUTPUT_DIR)
    if not connectivity:
        print("\nNo connectivity data. Cannot continue.")
        return

    puzzle = step5_solve(OUTPUT_DIR, n_cols, n_rows)

    duration = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Total time: {duration:.1f} seconds")
    print(f"Results saved to {OUTPUT_DIR}/")
    if puzzle:
        print(f"Puzzle solved successfully!")
    else:
        print(f"Puzzle solving did not complete.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
