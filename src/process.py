"""
Processes photographs of pieces into digitized piece data.

Supports two modes:
  - 'phone': mobile phone photo pipeline
  - 'robot': original robot-based pipeline (preserved for compatibility)
"""

import os
import sys
import time
import multiprocessing
import re
import pathlib
import json

import numpy as np

from common.config import (
    MODE, PUZZLE_WIDTH, PUZZLE_HEIGHT,
    PHOTOS_DIR, PHOTO_BMP_DIR, SEGMENT_DIR, PIECE_BMP_DIR, VECTOR_DIR, DEDUPED_DIR,
    CONNECTIVITY_DIR, SOLUTION_DIR, TIGHTNESS_DIR,
    MIN_PIECE_AREA, MAX_PIECE_DIMENSIONS, CROP_TOP_RIGHT_BOTTOM_LEFT,
    PHONE_TARGET_PIECE_SIZE,
)

if MODE == 'phone':
    from common import preprocess, segment_phone, extract as phone_extract
    from common.find_islands import save_island_as_bmp
    from common import vector, dedupe
else:
    from common import bmp, extract, util, vector, dedupe


def batch_process_photos(path, serialize=False, robot_states=None,
                         id=None, start_at_step=0, stop_before_step=10,
                         puzzle_width=None, puzzle_height=None,
                         segmentation_method='adaptive'):
    """
    Main processing entry point.
    Dispatches to phone or robot mode based on config.
    """
    if MODE == 'phone':
        return _batch_process_phone(
            path, serialize=serialize, id=id,
            start_at_step=start_at_step, stop_before_step=stop_before_step,
            puzzle_width=puzzle_width, puzzle_height=puzzle_height,
            segmentation_method=segmentation_method,
        )
    else:
        return _batch_process_robot(
            path, serialize=serialize, robot_states=robot_states,
            id=id, start_at_step=start_at_step,
            stop_before_step=stop_before_step,
        )


# --- Phone mode pipeline ---

def _batch_process_phone(path, serialize=False, id=None,
                         start_at_step=0, stop_before_step=10,
                         puzzle_width=None, puzzle_height=None,
                         segmentation_method='adaptive'):
    """
    Phone mode pipeline:
      Step 0: Preprocess photos (EXIF rotation, color normalization)
      Step 1: Segment photos (adaptive threshold)
      Step 2: Extract pieces (multi-piece per photo)
      Step 3: Vectorize pieces
      Step 4: Deduplicate
    """
    path = pathlib.Path(path)
    photos_dir = path.joinpath(PHOTOS_DIR)
    preprocessed_dir = path.joinpath('1_preprocessed')
    piece_bmp_dir = path.joinpath(PIECE_BMP_DIR)
    vector_dir = path.joinpath(VECTOR_DIR)
    deduped_dir = path.joinpath(DEDUPED_DIR)

    all_pieces = []
    piece_id_counter = 1

    if start_at_step <= 0 and stop_before_step > 0:
        print(f"\n### 0 - Preprocessing phone photos ###\n")
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

    photo_files = sorted([
        f for f in os.listdir(photos_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if id:
        photo_files = [f for f in photo_files if id in f]

    for photo_file in photo_files:
        photo_path = str(photos_dir.joinpath(photo_file))
        photo_id = os.path.splitext(photo_file)[0]
        print(f"\nProcessing photo: {photo_file}")

        # Step 0-1: Preprocess and segment
        if start_at_step <= 0:
            bgr, gray = preprocess.preprocess_phone_photo(photo_path)
            np.save(str(preprocessed_dir.joinpath(f'{photo_id}_gray.npy')), gray)
            np.save(str(preprocessed_dir.joinpath(f'{photo_id}_bgr.npy')), bgr)
        else:
            gray = np.load(str(preprocessed_dir.joinpath(f'{photo_id}_gray.npy')))
            bgr = np.load(str(preprocessed_dir.joinpath(f'{photo_id}_bgr.npy')))

        if start_at_step <= 1 and stop_before_step > 1:
            binary = segment_phone.segment_with_fallback(gray, bgr)
            np.save(str(preprocessed_dir.joinpath(f'{photo_id}_binary.npy')), binary)
        else:
            binary = np.load(str(preprocessed_dir.joinpath(f'{photo_id}_binary.npy')))

        # Step 2: Extract pieces
        if start_at_step <= 2 and stop_before_step > 2:
            pieces = phone_extract.extract_pieces_from_segmented(
                binary, bgr, photo_id
            )
            for piece in pieces:
                piece_binary_rescaled, piece_color_rescaled, _ = \
                    preprocess.normalize_piece_size(
                        piece.binary, gray, piece.origin,
                        color=piece.color,
                        target_size=PHONE_TARGET_PIECE_SIZE
                    )
                piece.binary = piece_binary_rescaled
                if piece_color_rescaled is not None:
                    piece.color = piece_color_rescaled
                all_pieces.append((piece_id_counter, piece))
                piece_id_counter += 1

    # Step 3: Vectorize
    if start_at_step <= 3 and stop_before_step > 3 and all_pieces:
        print(f"\n### 3 - Vectorizing {len(all_pieces)} pieces ###\n")
        piece_bmp_dir.mkdir(parents=True, exist_ok=True)
        vector_dir.mkdir(parents=True, exist_ok=True)

        args = []
        for pid, piece_data in all_pieces:
            if hasattr(piece_data, 'binary'):
                bmp_data = piece_data.binary
            else:
                bmp_data = piece_data

            h, w = bmp_data.shape
            bmp_path = str(piece_bmp_dir.joinpath(f'piece_{pid}.bmp'))
            save_island_as_bmp(bmp_data, bmp_path)

            metadata = {
                'original_photo_name': getattr(piece_data, 'photo_id', f'piece_{pid}'),
                'photo_space_origin': getattr(piece_data, 'origin', (0, 0)),
                'photo_space_centroid': [w // 2, h // 2],
                'photo_width': w,
                'photo_height': h,
                'is_complete': getattr(piece_data, 'is_complete', True),
            }
            args.append([bmp_path, pid, str(vector_dir), metadata,
                        (0, 0), 1.0, False])

        if serialize:
            for arg in args:
                try:
                    vector.load_and_vectorize(arg)
                except Exception as e:
                    print(f"Error vectorizing piece {arg[1]}: {e}")
        else:
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                results = pool.map(_safe_vectorize, args)
                for r in results:
                    if r is not None:
                        print(f"  Vectorized piece {r}")

        vector.validate_all_edges(str(vector_dir))

    # Step 4: Deduplicate
    if start_at_step <= 4 and stop_before_step > 4:
        print(f"\n### 4 - Deduplicating ###\n")
        deduped_dir.mkdir(parents=True, exist_ok=True)
        count = dedupe.deduplicate_phone(str(vector_dir), str(deduped_dir))
        if puzzle_width and puzzle_height:
            expected = puzzle_width * puzzle_height
            if count > expected:
                print(f"Warning: expected {expected} pieces but got {count}")
            elif count < expected:
                print(f"Warning: expected {expected} pieces but only got {count}. "
                      f"Missing {expected - count} pieces.")

    return all_pieces


def _safe_vectorize(args):
    try:
        vector.load_and_vectorize(args)
        return args[1]
    except Exception as e:
        print(f"Error vectorizing piece {args[1]}: {e}")
        return None


# --- Robot mode pipeline (preserved for compatibility) ---

def _batch_process_robot(path, serialize, robot_states, id,
                         start_at_step, stop_before_step):
    """Original robot mode pipeline."""
    from common import util

    if start_at_step <= 1 and stop_before_step > 1:
        width, height, scale_factor = _bmp_all(
            input_path=pathlib.Path(path).joinpath(PHOTOS_DIR),
            output_path=pathlib.Path(path).joinpath(PHOTO_BMP_DIR),
            id=id
        )
    else:
        # we'll need realistic data when skipping, so do the minimum amount of work
        input_dir = pathlib.Path(path).joinpath(PHOTOS_DIR)
        f = [f for f in os.listdir(input_dir) if re.match(r'.*\.jpe?g', f)][0]
        if os.path.exists("/dev/null"):
            args = [pathlib.Path(input_dir).joinpath(f), "/tmp/trash.bmp"]
        else:
            args = [pathlib.Path(input_dir).joinpath(f), "C:/Temp/trash.bmp"]
        width, height, scale_factor = bmp.photo_to_bmp(args)
        print(f"BMPs are {width}x{height} @ scale {scale_factor}")

    metadata = {
        "robot_state": {},  # will get filled in for each piece when vectorizing
        "scale_factor": scale_factor,
        "bmp_width": width,
        "bmp_height": height,
        "photo_width": width * scale_factor + CROP_TOP_RIGHT_BOTTOM_LEFT[1] + CROP_TOP_RIGHT_BOTTOM_LEFT[3],
        "photo_height": height * scale_factor + CROP_TOP_RIGHT_BOTTOM_LEFT[0] + CROP_TOP_RIGHT_BOTTOM_LEFT[2],
    }

    photo_space_positions = {}
    if start_at_step <= 2 and stop_before_step > 2:
        photo_space_positions = _extract_all(
            input_path=pathlib.Path(path).joinpath(PHOTO_BMP_DIR),
            output_path=pathlib.Path(path).joinpath(SEGMENT_DIR),
            scale_factor=scale_factor
        )
        with open(pathlib.Path(path).joinpath(SEGMENT_DIR).joinpath("photo_space_positions.json"), "w") as f:
            json.dump(photo_space_positions, f)
    else:
        with open(pathlib.Path(path).joinpath(SEGMENT_DIR).joinpath("photo_space_positions.json")) as f:
            photo_space_positions = json.load(f)
        print(f"Loaded {len(photo_space_positions)} photo space positions")

    if start_at_step <= 3 and stop_before_step > 3:
        _vectorize_all(
            input_path=pathlib.Path(path).joinpath(SEGMENT_DIR),
            metadata=metadata,
            robot_states=robot_states,
            output_path=pathlib.Path(path).joinpath(VECTOR_DIR),
            photo_space_positions=photo_space_positions,
            scale_factor=scale_factor,
            id=id,
            serialize=serialize
        )

    if start_at_step <= 4 and stop_before_step > 4:
        count = dedupe.deduplicate(
            batch_data_path=pathlib.Path(path).joinpath(PHOTOS_DIR).joinpath("batch.json"),
            input_path=pathlib.Path(path).joinpath(VECTOR_DIR),
            output_path=pathlib.Path(path).joinpath(DEDUPED_DIR)
        )
        if count > PUZZLE_WIDTH * PUZZLE_HEIGHT:
            raise Exception(f"dedupe: expected {PUZZLE_WIDTH * PUZZLE_HEIGHT} pieces but ended up with {count} unique pieces. Try adjusting DUPLICATE_CENTROID_DELTA_PX in config.py")
        elif count < PUZZLE_WIDTH * PUZZLE_HEIGHT:
            print(f"dedupe: expected {PUZZLE_WIDTH * PUZZLE_HEIGHT} pieces but ended up with {count} unique pieces. This is usually because some pieces are touching and were not separated. Try turning off CROP_TOP_RIGHT_BOTTOM_LEFT in config.py then running again to find the touching pieces.")


def _bmp_all(input_path, output_path, id):
    """
    Loads each photograph in the input directory and saves off a scaled black-and-white BMP in the output directory
    """
    print(f"\n{util.BLUE}### 0 - Segmenting photos into binary images ###{util.WHITE}\n")

    if id:
        fs = [f'{id}.jpeg']
    else:
        fs = [f for f in os.listdir(input_path) if re.match(r'.*\.jpe?g', f)]

    args = []
    for f in fs:
        input_img_path = pathlib.Path(input_path).joinpath(f)
        output_name = f.split('.')[0]
        output_img_path = pathlib.Path(output_path).joinpath(f'{output_name}.bmp')
        args.append([input_img_path, output_img_path])

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        # capture the output from each call to photo_to_bmp
        output = pool.map(bmp.photo_to_bmp, args)

    return output[0]


def _extract_all(input_path, output_path, scale_factor):
    """
    Loads each photograph in the input directory and saves off a scaled black-and-white BMP in the output directory
    """
    print(f"\n{util.BLUE}### 1 - Extracting pieces from photo bitmaps ###{util.WHITE}\n")
    start_time = time.time()
    output = extract.batch_extract(input_path, output_path, scale_factor)
    duration = time.time() - start_time
    print(f"Extracted {len(output)} pieces in {round(duration, 2)} seconds")
    return output


def _vectorize_all(input_path, output_path, metadata, robot_states, photo_space_positions, scale_factor, id, serialize):
    """
    Loads each image.bmp in the input directory, converts it to an SVG in the output directory
    """
    print(f"\n{util.BLUE}### 3 - Vectorizing ###{util.WHITE}\n")

    start_time = time.time()
    i = 1

    args = []

    for f in os.listdir(input_path):
        if not f.endswith('.bmp'):
            continue
        if id and f != id:
            continue

        path = pathlib.Path(input_path).joinpath(f)
        render = (id is not None)
        photo_space_position = photo_space_positions[f]
        original_photo_name = '_'.join(f.split('.')[0].split('_')[:-1]) + ".jpg"  # reverse engineer the BMP name to the JPG
        piece_metadata = metadata.copy()
        piece_metadata["photo_space_origin"] = photo_space_position
        piece_metadata["original_photo_name"] = original_photo_name
        piece_metadata["robot_state"] = {"photo_at_motor_position": robot_states[original_photo_name]}
        args.append([path, i, output_path, piece_metadata, photo_space_position, scale_factor, render])

        i += 1

    if serialize:
        for arg in args:
            vector.load_and_vectorize(arg)
    else:
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            pool.map(vector.load_and_vectorize, args)

    vector.validate_all_edges(output_path)

    duration = time.time() - start_time
    print(f"Vectorizing took {round(duration, 2)} seconds")
