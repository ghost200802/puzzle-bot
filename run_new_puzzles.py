#!/usr/bin/env python3
"""
Run the full puzzle pipeline on all RGBA images in input/puzzles/.

Handles:
  - RGBA images with transparent background
  - Filtering out incomplete pieces (touching image border)
  - Keeping only the largest connected component per piece
  - Cross-image deduplication

Pipeline:
  Step 0: Alpha-channel segmentation
  Step 1: Save pieces as BMP
  Step 2: Vectorize
  Step 3: Deduplicate
  Step 4: Build connectivity
  Step 5: Solve
"""

import os
import sys
import json
import shutil
import time
import math

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

INPUT_DIR = 'input/puzzles'
OUTPUT_DIR = 'output/puzzle_new'


def _touches_border(ys, xs, img_h, img_w, margin=2):
    """Check if a component touches the image border."""
    return (ys.min() < margin or ys.max() >= img_h - margin or
            xs.min() < margin or xs.max() >= img_w - margin)


def _keep_largest_component(binary):
    """Keep only the largest connected component in a binary image."""
    lbl, n = ndlabel(binary)
    if n <= 1:
        return binary
    best = max(range(1, n + 1), key=lambda j: np.sum(lbl == j))
    return (lbl == best).astype(np.uint8)


def segment_and_prepare(image_path, target_size=PHONE_TARGET_PIECE_SIZE):
    """
    Segment pieces from an RGBA image using alpha channel.
    Filters out incomplete pieces (touching border).
    Keeps only the largest connected component per piece.
    Scales each piece to target_size.
    """
    img = Image.open(image_path)
    rgba = np.array(img)

    if rgba.shape[2] == 4:
        alpha = rgba[:, :, 3]
    else:
        gray = np.array(img.convert('L'))
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    img_h, img_w = alpha.shape

    binary = (alpha > 128).astype(np.uint8)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3)

    labeled, num = ndlabel(binary)

    raw_pieces = []
    for i in range(1, num + 1):
        mask = (labeled == i)
        area = np.sum(mask)
        if area < 2000:
            continue

        ys, xs = np.where(mask)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1

        if _touches_border(ys, xs, img_h, img_w):
            print(f"    Skipping incomplete piece at ({xs.min()},{ys.min()})-({xs.max()},{ys.max()}), area={area}")
            continue

        raw_pieces.append({
            'label': i,
            'area': area,
            'bbox': (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
            'max_dim': max(w, h),
        })

    if not raw_pieces:
        return []

    typical_max_dim = int(np.median([p['max_dim'] for p in raw_pieces]))
    scale_factor = target_size / typical_max_dim

    print(f"    Complete pieces: {len(raw_pieces)}, typical size: {typical_max_dim}px, scale: {scale_factor:.1f}x")

    pieces = []
    pad = max(5, int(typical_max_dim * 0.05))

    for rp in raw_pieces:
        x0, y0, x1, y1 = rp['bbox']
        py0 = max(0, y0 - pad)
        py1 = min(img_h, y1 + pad)
        px0 = max(0, x0 - pad)
        px1 = min(img_w, x1 + pad)

        crop_binary = binary[py0:py1, px0:px1].copy()

        crop_binary = _keep_largest_component(crop_binary)

        ch, cw = crop_binary.shape
        new_w = max(int(cw * scale_factor), 10)
        new_h = max(int(ch * scale_factor), 10)

        crop_255 = (crop_binary * 255).astype(np.uint8)
        scaled = cv2.resize(crop_255, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        scaled = (scaled > 127).astype(np.uint8)

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        scaled = cv2.morphologyEx(scaled, cv2.MORPH_OPEN, kern)
        scaled = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kern)

        scaled = _keep_largest_component(scaled)

        pieces.append({
            'id': len(pieces) + 1,
            'binary': scaled,
            'origin': (int(px0), int(py0)),
            'centroid': ((px0 + px1) / 2.0, (py0 + py1) / 2.0),
            'area': int(np.sum(scaled)),
            'target_size': (new_w, new_h),
        })

    return pieces


def step0_segment_all(input_dir, output_dir):
    """Segment all PNG images and return all pieces."""
    print("\n" + "=" * 60)
    print("STEP 0: Alpha-channel segmentation")
    print("=" * 60)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        print("No image files found!")
        return []

    all_pieces = []
    global_id = 1

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        print(f"\n  Processing: {img_file}")

        pieces = segment_and_prepare(img_path)

        for p in pieces:
            p['id'] = global_id
            p['source_file'] = img_file
            all_pieces.append(p)
            global_id += 1

        print(f"    -> {len(pieces)} complete pieces extracted")

    print(f"\n  Total complete pieces across all images: {len(all_pieces)}")

    return all_pieces


def step1_save_pieces(pieces, output_dir):
    """Save piece binaries as BMP files for vectorization."""
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
            'original_photo_name': p.get('source_file', 'unknown'),
            'photo_space_origin': p['origin'],
            'photo_space_centroid': [w // 2, h // 2],
            'photo_width': w,
            'photo_height': h,
            'is_complete': True,
        }
        saved.append({'pid': pid, 'bmp_path': bmp_path, 'metadata': metadata})

    print(f"  Saved {len(saved)} pieces")
    return saved


def step2_vectorize(saved_pieces, output_dir):
    """Vectorize all pieces: find corners, extract sides."""
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
            if success % 20 == 0:
                print(f"  Progress: {success} vectorized...")
        except Exception as e:
            print(f"  Error vectorizing piece {s['pid']}: {str(e)[:120]}")
            failed += 1

    print(f"  Vectorized: {success} success, {failed} failed")
    return success


def step3_deduplicate(output_dir):
    """Deduplicate pieces across all images."""
    print("\n" + "=" * 60)
    print("STEP 3: Deduplicating")
    print("=" * 60)

    vector_dir = os.path.join(output_dir, VECTOR_DIR)
    deduped_dir = os.path.join(output_dir, DEDUPED_DIR)
    os.makedirs(deduped_dir, exist_ok=True)

    count = dedupe.deduplicate_phone(vector_dir, deduped_dir)
    print(f"  Deduplicated: {count} unique pieces")
    return count


def step4_connectivity(output_dir):
    """Build connectivity graph between pieces."""
    print("\n" + "=" * 60)
    print("STEP 4: Building connectivity")
    print("=" * 60)

    deduped_dir = os.path.join(output_dir, DEDUPED_DIR)
    conn_dir = os.path.join(output_dir, CONNECTIVITY_DIR)
    os.makedirs(conn_dir, exist_ok=True)

    from common import connect

    vector_dir = os.path.join(output_dir, VECTOR_DIR)
    connectivity = connect.build(deduped_dir, conn_dir, use_image_matching=True)

    corners = 0
    edges = 0
    piece_edge_info = {}
    for piece_id, piece_data in connectivity.items():
        pid = int(piece_id)
        edge_flags = []
        for si in range(4):
            json_path = os.path.join(vector_dir, f'side_{pid}_{si}.json')
            try:
                with open(json_path) as f:
                    side_data = json.load(f)
                edge_flags.append(side_data.get('is_edge', False))
            except Exception:
                edge_flags.append(False)
        piece_edge_info[pid] = edge_flags
        edge_count = sum(edge_flags)
        if edge_count >= 2:
            corners += 1
        elif edge_count >= 1:
            edges += 1

    print(f"  Connectivity: {len(connectivity)} pieces, {corners} corners, {edges} edges")
    return connectivity, corners, edges, piece_edge_info


def infer_dimensions(total, n_corners, n_edges):
    """Infer puzzle W x H from connectivity stats."""
    candidates = []
    for w in range(2, total + 1):
        if total % w == 0:
            h = total // w
            perim = 2 * (w + h) - 4
            if perim == n_edges:
                candidates.append((w, h, 0))
            elif abs(perim - n_edges) <= 2:
                candidates.append((max(w, h), min(w, h), abs(perim - n_edges)))
        h_up = (total + w - 1) // w
        if h_up >= 2:
            perim = 2 * (w + h_up) - 4
            if perim == n_edges:
                candidates.append((max(w, h_up), min(w, h_up), 0))
            elif abs(perim - n_edges) <= 2:
                candidates.append((max(w, h_up), min(w, h_up), abs(perim - n_edges)))
        if w * w > total * 2:
            break
    candidates.sort(key=lambda c: (c[2], abs(c[0] - c[1])))
    if candidates:
        return candidates[0][0], candidates[0][1]
    s = int(math.sqrt(total))
    return max(2, s), max(2, (total + s - 1) // s)


def step5_solve(output_dir, puzzle_width, puzzle_height, piece_edge_info=None):
    """Solve the puzzle using the connectivity graph."""
    print("\n" + "=" * 60)
    print("STEP 5: Solving puzzle")
    print("=" * 60)

    conn_dir = os.path.join(output_dir, CONNECTIVITY_DIR)
    sol_dir = os.path.join(output_dir, SOLUTION_DIR)
    os.makedirs(sol_dir, exist_ok=True)

    from common import board as board_mod, output as output_mod

    print(f"  Solving {puzzle_width}x{puzzle_height} puzzle...")

    try:
        puzzle = board_mod.build(
            input_path=conn_dir,
            output_path=sol_dir,
            puzzle_width=puzzle_width,
            puzzle_height=puzzle_height,
            piece_edge_info=piece_edge_info,
        )
        print("\n  *** PUZZLE SOLVED! ***")
        print(puzzle)

        output_mod.generate_solution_grid(puzzle, sol_dir)
        output_mod.print_solution_summary(puzzle)
        return puzzle

    except Exception as e:
        print(f"\n  Solve failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    start_time = time.time()

    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), INPUT_DIR)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_DIR)

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Puzzle Pipeline - Multi-image RGBA mode")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    pieces = step0_segment_all(input_dir, output_dir)
    if not pieces:
        print("No complete pieces found!")
        return

    saved = step1_save_pieces(pieces, output_dir)

    success_count = step2_vectorize(saved, output_dir)
    if success_count == 0:
        print("\nNo pieces vectorized. Cannot continue.")
        return

    deduped_count = step3_deduplicate(output_dir)
    if deduped_count == 0:
        print("\nNo pieces after deduplication. Cannot continue.")
        return

    print(f"\n  ** Validation: {deduped_count} unique pieces (expected ~100) **")

    connectivity, n_corners, n_edges, piece_edge_info = step4_connectivity(output_dir)
    if not connectivity:
        print("\nNo connectivity data. Cannot continue.")
        return

    total_pieces = len(connectivity)
    pw, ph = infer_dimensions(total_pieces, n_corners, n_edges)
    print(f"\n  Inferred dimensions: {pw} x {ph} = {pw * ph} "
          f"(actual: {total_pieces}, corners: {n_corners}, edges: {n_edges})")

    puzzle = step5_solve(output_dir, pw, ph, piece_edge_info)

    duration = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Total time: {duration:.1f} seconds")
    print(f"Results saved to {output_dir}/")
    if puzzle:
        print(f"Puzzle solved successfully!")
    else:
        print(f"Puzzle solving did not complete.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
