"""
Puzzle piece extraction from segmented photos.

Supports two modes:
  - 'robot' mode: original C-library based extraction from BMP files
  - 'phone' mode: multi-piece extraction from adaptive-segmented binary + color images
"""

import os
import re
import subprocess
import pathlib
import numpy as np

from common.config import (
    MIN_PIECE_AREA, CROP_TOP_RIGHT_BOTTOM_LEFT, PHONE_MIN_PIECE_AREA_RATIO, MODE,
)


class PieceCandidate:
    """
    Represents a single extracted puzzle piece candidate (phone mode).
    """
    __slots__ = (
        'binary', 'color', 'origin', 'photo_id',
        'is_complete', 'pixel_count',
    )

    def __init__(self, binary, color, origin, photo_id, is_complete=True):
        self.binary = binary
        self.color = color
        self.origin = origin
        self.photo_id = photo_id
        self.is_complete = is_complete
        self.pixel_count = int(np.sum(binary == 1))


def extract_pieces_from_segmented(binary, color_image, photo_id,
                                  min_area_ratio=PHONE_MIN_PIECE_AREA_RATIO):
    """
    Extract all puzzle pieces from a pre-segmented binary mask + color image (phone mode).

    binary: uint8 array, 0/1 (0=background, 1=piece)
    color_image: BGR uint8 array
    photo_id: string identifying the source photo

    Returns list[PieceCandidate]
    """
    from common.find_islands import remove_stragglers, extract_islands

    h, w = binary.shape
    min_area = int(h * w * min_area_ratio)
    if min_area < 100:
        min_area = 100

    cleaned = remove_stragglers(binary.copy())
    islands = extract_islands(cleaned, min_area, ignore_border=False)

    if len(islands) == 0:
        lower_min = max(50, min_area // 10)
        islands = extract_islands(cleaned, lower_min, ignore_border=False)
        if len(islands) > 0:
            print(f"  Extraction: lowered min_area {min_area}->{lower_min}, "
                  f"found {len(islands)} pieces")

    pieces = []
    for island_mask, origin_row, origin_col, touches_border in islands:
        bh, bw = island_mask.shape

        binary_crop = island_mask.copy()

        # Extract matching color region
        r0 = max(origin_row, 0)
        c0 = max(origin_col, 0)
        r1 = min(origin_row + bh, color_image.shape[0])
        c1 = min(origin_col + bw, color_image.shape[1])

        local_r0 = r0 - origin_row
        local_c0 = c0 - origin_col
        local_r1 = local_r0 + (r1 - r0)
        local_c1 = local_c0 + (c1 - c0)

        color_crop = np.zeros((bh, bw, 3), dtype=np.uint8)
        if r1 > r0 and c1 > c0:
            color_crop[local_r0:local_r1, local_c0:local_c1] = \
                color_image[r0:r1, c0:c1]
            mask_local = binary_crop[local_r0:local_r1, local_c0:local_c1]
            for ch in range(3):
                channel = color_crop[local_r0:local_r1, local_c0:local_c1, ch]
                channel[mask_local == 0] = 0

        is_complete = not touches_border

        pieces.append(PieceCandidate(
            binary=binary_crop,
            color=color_crop,
            origin=(origin_col, origin_row),
            photo_id=photo_id,
            is_complete=is_complete,
        ))

    print(f"Extracted {len(pieces)} pieces from photo '{photo_id}'")
    return pieces


def batch_extract(input_path, output_path, scale_factor):
    """
    Robot mode: original batch extraction using C library.
    """
    # just-in-time compile the C library to find islands
    source_file = pathlib.Path(os.path.join(os.path.dirname(__file__), '../c/find_islands.c'))
    cmd = fr"gcc -pthread -O3 -march=native -funroll-loops -ffast-math -o find_islands.o '{source_file}'"
    print(cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error compiling: {e}")
        exit(1)

    # invoke the C library to find islands
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    cmd = fr".{os.path.sep}find_islands.o '{input_path}' '{output_path}' {MIN_PIECE_AREA}"
    print(cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running extract lib: {e}")
        exit(1)

    output_photo_space_positions = {}

    fs = [f for f in os.listdir(output_path) if re.match(r'.*\.bmp', f)]
    for f in fs:
        components = f.split('.')[0].split('_')
        origin_component = components[-1]
        origin_x, origin_y = origin_component.strip('(').strip(')').split(',')
        origin = (int(origin_x), int(origin_y))

        photo_space_position = (origin[0] / scale_factor + CROP_TOP_RIGHT_BOTTOM_LEFT[-1], origin[1] / scale_factor + CROP_TOP_RIGHT_BOTTOM_LEFT[0])
        output_photo_space_positions[f] = photo_space_position
        print(f"Extracted {f} at {photo_space_position}, origin {origin}")

    return output_photo_space_positions
