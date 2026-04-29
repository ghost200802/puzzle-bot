"""
Connected component analysis for puzzle piece extraction.

Provides Python-based island extraction (alternative to the C library).
Supports optional border island filtering and dynamic area thresholds.
"""

import os
import numpy as np
from PIL import Image
from scipy import ndimage


def remove_stragglers(grid):
    """
    Iteratively remove isolated pixels (peninsulas) from a binary grid.
    A pixel is a straggler if it has <= 2 neighbors in the 3x3 window
    and is not on the edge of the grid.
    """
    rows, cols = grid.shape
    mask = np.zeros_like(grid, dtype=bool)
    mask[1:-1, 1:-1] = True
    changed = True
    while changed:
        changed = False
        kernel = np.ones((3, 3), dtype=np.int32)
        kernel[1, 1] = 0
        neighbor_count = ndimage.convolve(grid.astype(np.int32), kernel, mode='constant', cval=0)
        stragglers = mask & (grid == 1) & (neighbor_count <= 2)
        if np.any(stragglers):
            grid[stragglers] = 0
            changed = True
    return grid


def extract_islands(grid, min_island_area, ignore_border=True):
    """
    Extract connected components (islands) from a binary grid.
    Returns list of (island_grid, origin_row, origin_col, touches_border).
    If ignore_border=True, islands touching the image border are excluded.
    """
    labeled, num_features = ndimage.label(grid)
    islands = []
    for label_id in range(1, num_features + 1):
        positions = np.argwhere(labeled == label_id)
        area = len(positions)
        if area < min_island_area:
            continue
        min_r, min_c = positions.min(axis=0)
        max_r, max_c = positions.max(axis=0)
        touches_border = (
            min_r == 0 or max_r == grid.shape[0] - 1 or
            min_c == 0 or max_c == grid.shape[1] - 1
        )
        if ignore_border and touches_border:
            continue
        padding = 1
        rows = max_r - min_r + 1 + 2 * padding
        cols = max_c - min_c + 1 + 2 * padding
        island = np.zeros((rows, cols), dtype=np.uint8)
        for r, c in positions:
            island[r - min_r + padding, c - min_c + padding] = 1
        origin_row = min_r - padding
        origin_col = min_c - padding
        islands.append((island, origin_row, origin_col, touches_border))
    return islands


def load_binary_bitmap(path):
    """Load a BMP file as a binary (0/1) numpy array."""
    with Image.open(path) as img:
        img = img.convert('1')
        grid = np.array(img, dtype=np.int8)
        grid = np.where(grid > 0, 1, 0)
    return grid


def save_island_as_bmp(island, filename):
    """Save a binary numpy array as a BMP file."""
    img_data = (island * 255).astype(np.uint8)
    img = Image.fromarray(img_data, mode='L')
    img = img.convert('1')
    img.save(filename)


def process_file(filepath, filename, output_directory, min_island_area,
                 ignore_border=True, color_image=None):
    """Process a single BMP file: extract islands and save them."""
    print(f"Extracting from {filepath}")
    grid = load_binary_bitmap(filepath)
    grid = remove_stragglers(grid)
    islands = extract_islands(grid, min_island_area, ignore_border=ignore_border)
    base_name = filename[:-4]
    results = []
    for island, origin_row, origin_col, touches_border in islands:
        output_filename = os.path.join(
            output_directory,
            f"{base_name}_({origin_col},{origin_row}).bmp"
        )
        save_island_as_bmp(island, output_filename)

        # Optionally extract color crop
        color_crop = None
        if color_image is not None:
            h, w = island.shape
            r0 = max(origin_row, 0)
            c0 = max(origin_col, 0)
            r1 = min(origin_row + h, color_image.shape[0])
            c1 = min(origin_col + w, color_image.shape[1])
            color_crop = color_image[r0:r1, c0:c1].copy()

        results.append({
            'filename': output_filename,
            'origin': (origin_col, origin_row),
            'island': island,
            'touches_border': touches_border,
            'color_crop': color_crop,
        })
    return results


def batch_extract(input_directory, output_directory, min_island_area,
                  ignore_border=True):
    """Batch process all BMP files in a directory."""
    os.makedirs(output_directory, exist_ok=True)
    for filename in sorted(os.listdir(input_directory)):
        if filename.lower().endswith('.bmp'):
            filepath = os.path.join(input_directory, filename)
            process_file(filepath, filename, output_directory,
                         min_island_area, ignore_border=ignore_border)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <input_directory> <output_directory> <min_island_area>")
        sys.exit(1)
    batch_extract(sys.argv[1], sys.argv[2], int(sys.argv[3]))
