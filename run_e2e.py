"""
End-to-end pipeline runner using example_data.

Runs the full pipeline:
  Phase A (using pre-vectorized data from example_data):
    - Connectivity building from 76 deduplicated pieces
    - Board solving
    - Output generation

  Phase B (using raw photos from example_data):
    - Preprocessing → Segmentation → Extraction → Vectorization → Dedup
    - Then connectivity + solving + output

Results are saved to f:\\work_Puzzle_github\\puzzle-bot\\output\\
"""

import os
import sys
import json
import shutil
import time
import pathlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
EXAMPLE_DATA = os.path.join(os.path.dirname(__file__), '..', 'example_data')


def run_connectivity_and_solve():
    """
    Phase 1: Use pre-vectorized deduplicated data to build connectivity and solve.
    """
    from common import connect, board, output, config

    deduped_src = os.path.join(EXAMPLE_DATA, '4_deduped')
    work_dir = os.path.join(OUTPUT_DIR, 'from_deduped')

    if not os.path.isdir(deduped_src):
        print(f"ERROR: {deduped_src} not found")
        return False

    # Clean and create work directory
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    connectivity_dir = os.path.join(work_dir, '5_connectivity')
    solution_dir = os.path.join(work_dir, '6_solution')
    os.makedirs(connectivity_dir, exist_ok=True)
    os.makedirs(solution_dir, exist_ok=True)

    # Copy deduped data
    deduped_dir = os.path.join(work_dir, '4_deduped')
    shutil.copytree(deduped_src, deduped_dir)

    # Count pieces
    side_files = list(pathlib.Path(deduped_dir).glob('side_*_0.json'))
    piece_ids = sorted(set(
        int(f.stem.split('_')[1]) for f in side_files
    ))
    num_pieces = len(piece_ids)
    print(f"\n{'='*60}")
    print(f"Phase 1: Connectivity & Solve ({num_pieces} pieces)")
    print(f"{'='*60}")

    # Step 1: Build connectivity
    print(f"\n--- Step 1: Building connectivity ---")
    start = time.time()
    try:
        connectivity = connect.build(deduped_dir, connectivity_dir)
        duration = time.time() - start
        print(f"  Connectivity built in {duration:.1f}s")
        print(f"  Connectivity entries: {len(connectivity) if connectivity else 0}")
    except Exception as e:
        print(f"  Connectivity failed: {e}")
        return False

    # Save connectivity summary
    if connectivity:
        summary = {}
        for pid, fits_list in connectivity.items():
            fits_info = {}
            for si, fits in enumerate(fits_list):
                if fits:
                    fits_info[str(si)] = [
                        [fid, fsj, round(err, 4)] for fid, fsj, err in fits
                    ]
            summary[str(pid)] = {'fits': fits_info, 'num_matches': sum(len(f) for f in fits_list)}
        with open(os.path.join(connectivity_dir, 'connectivity_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Saved connectivity_summary.json")

    # Step 2: Solve
    print(f"\n--- Step 2: Solving ---")

    # Calculate grid dimensions for available pieces
    # For a subset, try a reasonable grid
    puzzle_width = config.PUZZLE_WIDTH  # 40
    puzzle_height = config.PUZZLE_HEIGHT  # 25
    print(f"  Full puzzle: {puzzle_width}x{puzzle_height} = {puzzle_width * puzzle_height}")
    print(f"  Available pieces: {num_pieces}")

    # Count corners and edges from connectivity data
    corners_count = 0
    edges_count = 0
    for pid, fits_list in connectivity.items():
        empty_sides = sum(1 for f in fits_list if len(f) == 0)
        if empty_sides >= 2:
            corners_count += 1
        if empty_sides >= 1:
            edges_count += 1
    print(f"  Detected: {corners_count} corners, {edges_count} edges")

    if corners_count < 4:
        print(f"\n  Not enough corners to solve full puzzle ({corners_count} < 4)")
        print(f"  Attempting partial solve with subset of pieces...")

        # Try to find a cluster of connected pieces and solve a small section
        # Find the piece with the most connections
        best_pid = max(connectivity.keys(),
                       key=lambda pid: sum(len(f) for f in connectivity[pid]))
        best_connections = sum(len(f) for f in connectivity[best_pid])
        print(f"  Most connected piece: {best_pid} ({best_connections} connections)")

        # Try to build a small grid from this piece
        for w in range(3, 12):
            for h in range(3, 12):
                if w * h <= num_pieces and 2 * (w + h) - 4 <= edges_count:
                    print(f"  Trying {w}x{h} grid...")
                    start = time.time()
                    try:
                        puzzle = board.build(
                            connectivity=connectivity,
                            input_path=connectivity_dir,
                            output_path=solution_dir,
                            puzzle_width=w,
                            puzzle_height=h,
                        )
                        duration = time.time() - start
                        if puzzle is not None:
                            placed = len([p for p in puzzle if p is not None])
                            print(f"  SUCCESS: {placed} pieces placed in {duration:.1f}s!")
                            output.generate_solution_grid(puzzle, solution_dir)
                            print(f"  Generated solution_grid.txt")
                            break
                    except Exception as e:
                        pass
            else:
                continue
            break
        else:
            print(f"  Could not solve any grid size with available data")
    else:
        # Full puzzle solve
        start = time.time()
        try:
            puzzle = board.build(
                connectivity=connectivity,
                input_path=connectivity_dir,
                output_path=solution_dir,
                puzzle_width=puzzle_width,
                puzzle_height=puzzle_height,
            )
            duration = time.time() - start
            print(f"  Solve completed in {duration:.1f}s")

            if puzzle is not None:
                placed = len([p for p in puzzle if p is not None])
                print(f"  SUCCESS: {placed} pieces placed!")
                output.generate_solution_grid(puzzle, solution_dir)
                print(f"  Generated solution_grid.txt")
                output.print_solution_summary(puzzle, num_pieces)
        except Exception as e:
            print(f"  Solve error: {e}")

    print(f"\n  Results saved to: {work_dir}")
    return True


def run_photo_pipeline():
    """
    Phase 2: Use raw photos to run the full phone-mode pipeline.
    """
    from common import preprocess, segment_phone, config
    from common.find_islands import save_island_as_bmp
    from common.extract import extract_pieces_from_segmented
    from common import vector, dedupe

    photos_src = os.path.join(EXAMPLE_DATA, '0_photos')
    work_dir = os.path.join(OUTPUT_DIR, 'from_photos')

    if not os.path.isdir(photos_src):
        print(f"ERROR: {photos_src} not found")
        return False

    # Clean and create work directory
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Use only first 10 photos for reasonable runtime
    photos = sorted([
        f for f in os.listdir(photos_src)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:10]

    print(f"\n{'='*60}")
    print(f"Phase 2: Photo Pipeline ({len(photos)} photos)")
    print(f"{'='*60}")

    # Copy photos
    photos_dir = os.path.join(work_dir, '0_photos')
    os.makedirs(photos_dir, exist_ok=True)
    for p in photos:
        shutil.copy2(os.path.join(photos_src, p), os.path.join(photos_dir, p))

    vector_dir = os.path.join(work_dir, '3_vector')
    deduped_dir = os.path.join(work_dir, '4_deduped')
    os.makedirs(vector_dir, exist_ok=True)
    os.makedirs(deduped_dir, exist_ok=True)

    # Step 0-1: Preprocess + Segment
    print(f"\n--- Steps 0-1: Preprocessing + Segmentation ---")
    all_pieces = []
    piece_id = 1

    for i, photo_file in enumerate(photos):
        photo_path = os.path.join(photos_dir, photo_file)
        photo_id = os.path.splitext(photo_file)[0]

        print(f"  [{i+1}/{len(photos)}] {photo_file}...", end=" ")

        try:
            bgr, gray = preprocess.preprocess_phone_photo(photo_path)
            binary = segment_phone.segment_photo(gray, bgr=bgr, method='adaptive')
            pieces = extract_pieces_from_segmented(binary, bgr, photo_id)

            for piece in pieces:
                scaled_binary, scaled_color, _ = preprocess.normalize_piece_size(
                    piece.binary, piece.color,
                    target_size=config.PHONE_TARGET_PIECE_SIZE
                )
                all_pieces.append((piece_id, scaled_binary, scaled_color, piece))
                piece_id += 1

            print(f"{len(pieces)} pieces")
        except Exception as e:
            print(f"Error: {e}")

    print(f"  Total pieces extracted: {len(all_pieces)}")

    # Step 2: Vectorize
    print(f"\n--- Step 2: Vectorization ---")
    vectorized = 0
    for pid, bmp_data, color_data, piece_orig in all_pieces:
        bmp_path = os.path.join(vector_dir, f'piece_{pid}.bmp')
        save_island_as_bmp(bmp_data, bmp_path)

        h, w = bmp_data.shape
        metadata = {
            'original_photo_name': piece_orig.photo_id,
            'photo_space_origin': piece_orig.origin,
            'photo_space_centroid': [w // 2, h // 2],
            'photo_width': w,
            'photo_height': h,
            'is_complete': piece_orig.is_complete,
        }
        args = [bmp_path, pid, vector_dir, metadata, (0, 0), 1.0, False]
        try:
            vector.load_and_vectorize(args)
            vectorized += 1
        except Exception as e:
            print(f"  Warning: piece {pid} failed: {str(e)[:100]}")

    print(f"  Vectorized: {vectorized}/{len(all_pieces)}")

    # Step 3: Deduplicate
    print(f"\n--- Step 3: Deduplication ---")
    count = dedupe.deduplicate_phone(vector_dir, deduped_dir)
    print(f"  Unique pieces after dedup: {count}")

    # Step 4: Connectivity
    print(f"\n--- Step 4: Connectivity ---")
    connectivity_dir = os.path.join(work_dir, '5_connectivity')
    os.makedirs(connectivity_dir, exist_ok=True)

    try:
        connectivity = connect.build(deduped_dir, connectivity_dir)
        print(f"  Connectivity entries: {len(connectivity) if connectivity else 0}")
    except Exception as e:
        print(f"  Connectivity failed: {e}")
        connectivity = None

    # Step 5: Solve
    if connectivity:
        print(f"\n--- Step 5: Solve ---")
        solution_dir = os.path.join(work_dir, '6_solution')
        os.makedirs(solution_dir, exist_ok=True)

        try:
            puzzle = board.build(
                connectivity=connectivity,
                input_path=connectivity_dir,
                output_path=solution_dir,
                puzzle_width=3,
                puzzle_height=2,
            )
            if puzzle:
                placed = len([p for p in puzzle if p is not None])
                print(f"  Placed {placed} pieces")
                output.generate_solution_grid(puzzle, solution_dir)
            else:
                print(f"  No solution found")
        except Exception as e:
            print(f"  Solve failed: {e}")

    print(f"\n  Results saved to: {work_dir}")
    return True


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Puzzle Bot - End-to-End Pipeline Runner")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Example data: {EXAMPLE_DATA}")

    # Phase 1: Use pre-vectorized data (fast)
    success = run_connectivity_and_solve()

    # Phase 2: Use raw photos (slower, limited to 10 photos)
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        run_photo_pipeline()
    else:
        print(f"\nSkipping photo pipeline (use --full to include)")
        print(f"  python run_e2e.py --full")

    print(f"\nDone! Check {OUTPUT_DIR} for results.")
