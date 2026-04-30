"""
Solve a puzzle from a single PNG image with transparent background.

Handles:
  - RGBA images where pieces are on transparent background but touching each other
  - Uses alpha channel + morphological erosion to separate pieces
  - Runs full pipeline: segment → extract → vectorize → deduplicate → connectivity → solve
"""

import os
import sys
import json
import shutil
import time
import pathlib

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

INPUT_IMAGE = 'input/puzzles/1.png'
OUTPUT_DIR = 'output/puzzle_1'


def segment_pieces_from_alpha(image_path):
    """Use alpha channel + morphology to separate touching puzzle pieces."""
    from PIL import Image
    from scipy.ndimage import label as ndlabel
    from common.find_islands import remove_stragglers

    img = Image.open(image_path)
    rgba = np.array(img)

    if rgba.shape[2] == 4:
        alpha = rgba[:, :, 3]
    else:
        gray = np.array(img.convert('L'))
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    binary = (alpha > 128).astype(np.uint8)
    print(f"  Alpha mask: {np.sum(binary)} foreground pixels")

    # Erode to separate touching pieces
    # Try increasing erosion until we get a reasonable number of components
    best_binary = None
    best_count = 0
    best_kernel_size = 0

    for kernel_size in range(3, 25, 2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(binary, kernel, iterations=1)
        labeled, num = ndlabel(eroded)
        # Filter small components
        valid = 0
        for i in range(1, num + 1):
            if np.sum(labeled == i) > 500:
                valid += 1
        print(f"  Erosion kernel={kernel_size}: {valid} valid components")
        if valid > best_count:
            best_count = valid
            best_binary = eroded
            best_kernel_size = kernel_size
        if valid > 30:
            break

    print(f"  Best erosion: kernel={best_kernel_size}, components={best_count}")
    return rgba[:, :, :3], best_binary, best_count


def determine_grid_size(num_pieces):
    """Try to determine the grid dimensions from the number of pieces."""
    common_grids = [
        (2, 2), (2, 3), (3, 2), (3, 3), (4, 3), (3, 4),
        (4, 4), (5, 3), (3, 5), (4, 5), (5, 4), (5, 5),
        (6, 4), (4, 6), (6, 5), (5, 6), (6, 6),
        (7, 5), (5, 7), (7, 6), (6, 7), (8, 6), (6, 8),
        (8, 7), (7, 8), (8, 8), (9, 7), (7, 9),
        (10, 7), (7, 10), (10, 8), (8, 10),
    ]
    for w, h in common_grids:
        if w * h == num_pieces:
            return w, h
    # Fallback: find closest rectangle
    for w in range(2, 15):
        for h in range(2, 15):
            if w * h == num_pieces:
                return w, h
    # Try near-matches
    for w in range(2, 15):
        for h in range(2, 15):
            if abs(w * h - num_pieces) <= 2:
                return w, h
    return None, None


def run_pipeline():
    from common import preprocess, segment_phone, config
    from common.find_islands import save_island_as_bmp
    from common.extract import extract_pieces_from_segmented
    from common import vector, dedupe, connect, board, output

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    photos_dir = os.path.join(OUTPUT_DIR, '0_photos')
    vector_dir = os.path.join(OUTPUT_DIR, '3_vector')
    deduped_dir = os.path.join(OUTPUT_DIR, '4_deduped')
    conn_dir = os.path.join(OUTPUT_DIR, '5_connectivity')
    sol_dir = os.path.join(OUTPUT_DIR, '6_solution')

    for d in [photos_dir, vector_dir, deduped_dir, conn_dir, sol_dir]:
        os.makedirs(d, exist_ok=True)

    shutil.copy2(INPUT_IMAGE, os.path.join(photos_dir, '1.png'))

    # Step 0-1: Segment
    print(f"\n{'='*60}")
    print("Step 0-1: Segmentation (alpha channel + morphology)")
    print(f"{'='*60}")
    bgr, binary, estimated_count = segment_pieces_from_alpha(INPUT_IMAGE)
    print(f"  Estimated pieces: {estimated_count}")

    # Step 2: Extract
    print(f"\n{'='*60}")
    print("Step 2: Piece extraction")
    print(f"{'='*60}")
    pieces = extract_pieces_from_segmented(binary, bgr, 'photo_1')
    print(f"  Extracted {len(pieces)} pieces")

    if len(pieces) < 4:
        print(f"ERROR: Only {len(pieces)} pieces found, need at least 4")
        return False

    # Save piece info
    for i, p in enumerate(pieces):
        print(f"  Piece {i+1}: {p.binary.shape}, pixels={p.pixel_count}, complete={p.is_complete}")

    # Step 3: Vectorize
    print(f"\n{'='*60}")
    print("Step 3: Vectorization")
    print(f"{'='*60}")
    target_size = config.PHONE_TARGET_PIECE_SIZE
    piece_id = 1
    vectorized = 0

    for p in pieces:
        scaled_binary, scaled_color, scale = preprocess.normalize_piece_size(
            p.binary, p.color, target_size=target_size
        )
        bmp_path = os.path.join(vector_dir, f'piece_{piece_id}.bmp')
        save_island_as_bmp(scaled_binary, bmp_path)

        h, w = scaled_binary.shape
        metadata = {
            'original_photo_name': p.photo_id,
            'photo_space_origin': [int(p.origin[0]), int(p.origin[1])],
            'photo_space_centroid': [w // 2, h // 2],
            'photo_width': w,
            'photo_height': h,
            'is_complete': p.is_complete,
        }
        args = [bmp_path, piece_id, vector_dir, metadata, (0, 0), 1.0, False]
        try:
            vector.load_and_vectorize(args)
            vectorized += 1
            print(f"  Vectorized piece {piece_id}: OK")
        except Exception as e:
            print(f"  Vectorized piece {piece_id}: FAILED - {str(e)[:120]}")
        piece_id += 1

    print(f"  Vectorized: {vectorized}/{len(pieces)}")

    if vectorized == 0:
        print("ERROR: No pieces vectorized successfully")
        return False

    # Step 4: Deduplicate
    print(f"\n{'='*60}")
    print("Step 4: Deduplication")
    print(f"{'='*60}")
    count = dedupe.deduplicate_phone(vector_dir, deduped_dir)
    print(f"  Unique pieces: {count}")

    if count < 4:
        print(f"ERROR: Only {count} unique pieces, need at least 4")
        return False

    # Determine grid
    w, h = determine_grid_size(count)
    if w is None:
        print(f"  Could not determine grid for {count} pieces")
        # Try common sizes
        for tw in range(3, 12):
            for th in range(3, 12):
                if tw * th == count:
                    w, h = tw, th
                    break
            if w:
                break
    if w is None:
        w = count // 2
        h = 2
        while w * h > count:
            h -= 1
        if h < 2:
            h = 2
            w = count // h

    print(f"  Grid: {w}x{h} = {w*h} (have {count} pieces)")

    # Step 5: Connectivity
    print(f"\n{'='*60}")
    print("Step 5: Connectivity")
    print(f"{'='*60}")
    try:
        connectivity = connect.build(deduped_dir, conn_dir)
        print(f"  Connectivity entries: {len(connectivity) if connectivity else 0}")
    except Exception as e:
        print(f"  Connectivity failed: {e}")
        return False

    if not connectivity:
        print("ERROR: No connectivity data")
        return False

    # Save connectivity summary
    summary = {}
    for pid, fits_list in connectivity.items():
        fits_info = {}
        for si, fits in enumerate(fits_list):
            if fits:
                fits_info[str(si)] = [[fid, fsj, round(err, 4)] for fid, fsj, err in fits]
        summary[str(pid)] = {'fits': fits_info, 'num_matches': sum(len(f) for f in fits_list)}
    with open(os.path.join(conn_dir, 'connectivity_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Count corners and edges
    corners = []
    edges = []
    for pid, fits_list in connectivity.items():
        empty_sides = sum(1 for f in fits_list if len(f) == 0)
        if empty_sides >= 2:
            corners.append(pid)
        if empty_sides >= 1:
            edges.append(pid)
    print(f"  Corners: {len(corners)}, Edges: {len(edges)}")
    print(f"  Expected: 4 corners, {2*(w+h)-4} edges for {w}x{h} grid")

    # Step 6: Solve
    print(f"\n{'='*60}")
    print("Step 6: Solving")
    print(f"{'='*60}")

    solved = False
    # Try the detected grid first
    grids_to_try = [(w, h)]
    # Also try swapped and nearby sizes
    if (h, w) not in grids_to_try:
        grids_to_try.append((h, w))
    for dw in range(-1, 2):
        for dh in range(-1, 2):
            tw, th = w + dw, h + dh
            if tw >= 2 and th >= 2 and tw * th <= count and (tw, th) not in grids_to_try:
                grids_to_try.append((tw, th))

    for tw, th in grids_to_try:
        print(f"  Trying {tw}x{th} grid...")
        start = time.time()
        try:
            puzzle = board.build(
                connectivity=connectivity,
                input_path=conn_dir,
                output_path=sol_dir,
                puzzle_width=tw,
                puzzle_height=th,
            )
            duration = time.time() - start
            if puzzle is not None:
                placed = len([p for p in puzzle if p is not None])
                print(f"  SUCCESS: {placed} pieces placed in {duration:.1f}s!")
                output.generate_solution_grid(puzzle, sol_dir)
                print(f"  Generated solution_grid.txt")
                solved = True
                break
            else:
                print(f"  No solution for {tw}x{th}")
        except Exception as e:
            print(f"  Failed: {str(e)[:120]}")

    if not solved:
        print(f"\n  Could not solve the puzzle with any grid size")

    # Step 7: Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Input: {INPUT_IMAGE}")
    print(f"  Pieces extracted: {len(pieces)}")
    print(f"  Pieces vectorized: {vectorized}")
    print(f"  Unique pieces after dedup: {count}")
    print(f"  Corners detected: {len(corners)}")
    print(f"  Edges detected: {len(edges)}")
    print(f"  Solved: {solved}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")

    return solved


if __name__ == '__main__':
    run_pipeline()
