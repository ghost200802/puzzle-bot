"""
Common configuration for the puzzle bot

Supports two modes:
  - 'robot': original robot-based puzzle solver
  - 'phone': mobile phone photo-based puzzle assistant

Auto-tuning:
  - auto_tune_for_image(image) adjusts segmentation parameters based on image properties
  - load_config(config_path) / save_config(config_path) for config persistence
"""

import os
import json
import math


# Operating mode: 'phone' (default) or 'robot'
MODE = os.environ.get('PUZZLE_MODE', 'phone')


# dimensions for the puzzle you're solving
PUZZLE_WIDTH = 40
PUZZLE_HEIGHT = 25
PUZZLE_NUM_PIECES = PUZZLE_WIDTH * PUZZLE_HEIGHT
TIGHTEN_RELAX_PX_W = 5.699827119  # positive = add space between pieces, negative = remove space between pieces
TIGHTEN_RELAX_PX_H = 9.121796862


# Paramaters for photo segmentation
SCALE_BMP_TO_WIDTH = None  # scale the BMP to this wide or None to turn off scaling
CROP_TOP_RIGHT_BOTTOM_LEFT = (620, 860, 620, 860)  # crop the BMP by this many pixels on each side
MIN_PIECE_AREA = 400*400
MAX_PIECE_DIMENSIONS = (1420, 1420)  # we use this to catch when two pieces are touching
SEG_THRESH = 145  # raise this to cut tighter into the border


# Robot parameters
APPROX_ROBOT_COUNTS_PER_PIXEL = 10


# Deduplication
DUPLICATE_CENTROID_DELTA_PX = 22.0


# Directory structure for data processing
# Step 1 takes in photos of pieces on the bed and outputs binary BMPs of those photos
PHOTOS_DIR = '0_photos'
PHOTO_BMP_DIR = '1_photo_bmps'

# Step 2 takes in photo BMPs and outputs cleaned up individual pieces as bitmaps
SEGMENT_DIR = '2_segmented'

# Step 2b (phone mode): individual piece binary BMPs (intermediate, input to vectorization)
PIECE_BMP_DIR = '2_piece_bmps'

# Step 3 takes in piece BMPs and outputs SVGs and JSON
VECTOR_DIR = '3_vector'

# Step 4 goes through all the vector pieces and deletes duplicates
DEDUPED_DIR = '4_deduped'

# Step 5 takes in SVGs and outputs a graph of connectivity
CONNECTIVITY_DIR = '5_connectivity'

# Step 6 takes in the graph of connectivity and outputs a solution
SOLUTION_DIR = '6_solution'

# Step 7 adjusts the tightness of the solved puzzle: how much breathing room do pieces need to actually click together?
TIGHTNESS_DIR = '7_tightness'


# Phone mode parameters (used when MODE == 'phone')
if MODE == 'phone':
    # Preprocessing
    PHONE_BLUR_KERNEL = 5             # Gaussian blur kernel size for segmentation
    PHONE_MORPH_KERNEL = 5            # Morphological operations kernel size
    PHONE_ADAPTIVE_BLOCK_SIZE = 51    # Adaptive threshold block size (must be odd)
    PHONE_ADAPTIVE_C = 10             # Adaptive threshold constant subtracted from mean
    PHONE_MIN_PIECE_AREA_RATIO = 0.005  # Minimum piece area as ratio of image area
    PHONE_TARGET_PIECE_SIZE = 945     # Target size for piece normalization
    # Deduplication
    PHONE_HASH_THRESHOLD = 15         # Perceptual hash distance threshold
    PHONE_DUPLICATE_GEOMETRIC_THRESHOLD = 2.2  # Geometric similarity threshold for dedup
    # Segmentation
    PHONE_BG_BRIGHTNESS_THRESHOLD = 128  # Brightness threshold for background detection
    PHONE_BORDER_RATIO_FOR_BG_DETECT = 0.1  # Border ratio for background sampling


_config_overrides = {}


def load_config(config_path):
    """
    Load configuration overrides from a JSON file.
    Values from the file override the default constants.
    """
    global _config_overrides
    if not os.path.exists(config_path):
        return

    with open(config_path) as f:
        _config_overrides = json.load(f)

    globals().update(_config_overrides)
    print(f"Loaded config overrides from {config_path}")


def save_config(config_path):
    """
    Save current configuration to a JSON file.
    Only saves phone mode parameters.
    """
    phone_keys = [
        'PHONE_BLUR_KERNEL', 'PHONE_MORPH_KERNEL',
        'PHONE_ADAPTIVE_BLOCK_SIZE', 'PHONE_ADAPTIVE_C',
        'PHONE_MIN_PIECE_AREA_RATIO', 'PHONE_TARGET_PIECE_SIZE',
        'PHONE_HASH_THRESHOLD', 'PHONE_DUPLICATE_GEOMETRIC_THRESHOLD',
        'PHONE_BG_BRIGHTNESS_THRESHOLD', 'PHONE_BORDER_RATIO_FOR_BG_DETECT',
        'PUZZLE_WIDTH', 'PUZZLE_HEIGHT',
    ]
    config = {k: globals().get(k) for k in phone_keys}
    os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else '.', exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def auto_tune_for_image(image_size, image_brightness=None, image_contrast=None):
    """
    Auto-tune phone mode parameters based on image properties.

    Adjusts:
      - PHONE_ADAPTIVE_BLOCK_SIZE based on image resolution
      - PHONE_ADAPTIVE_C based on image brightness
      - PHONE_BLUR_KERNEL based on image resolution
      - PHONE_MIN_PIECE_AREA_RATIO based on expected piece count

    Args:
        image_size: (width, height) tuple
        image_brightness: mean brightness (0-255), computed if None
        image_contrast: std deviation (0-128), computed if None

    Returns:
        dict of adjusted parameter names and values
    """
    global PHONE_ADAPTIVE_BLOCK_SIZE, PHONE_ADAPTIVE_C
    global PHONE_BLUR_KERNEL, PHONE_MORPH_KERNEL
    global PHONE_MIN_PIECE_AREA_RATIO

    adjustments = {}
    w, h = image_size
    max_dim = max(w, h)

    ref_dim = 2000
    scale_factor = max_dim / ref_dim

    new_block_size = max(3, min(99, int(PHONE_ADAPTIVE_BLOCK_SIZE * scale_factor)))
    if new_block_size % 2 == 0:
        new_block_size += 1
    if new_block_size != PHONE_ADAPTIVE_BLOCK_SIZE:
        adjustments['PHONE_ADAPTIVE_BLOCK_SIZE'] = new_block_size
        PHONE_ADAPTIVE_BLOCK_SIZE = new_block_size

    new_blur = max(3, min(15, int(PHONE_BLUR_KERNEL * scale_factor)))
    if new_blur % 2 == 0:
        new_blur += 1
    if new_blur != PHONE_BLUR_KERNEL:
        adjustments['PHONE_BLUR_KERNEL'] = new_blur
        PHONE_BLUR_KERNEL = new_blur

    new_morph = max(3, min(15, int(PHONE_MORPH_KERNEL * scale_factor)))
    if new_morph % 2 == 0:
        new_morph += 1
    if new_morph != PHONE_MORPH_KERNEL:
        adjustments['PHONE_MORPH_KERNEL'] = new_morph
        PHONE_MORPH_KERNEL = new_morph

    if image_brightness is not None:
        if image_brightness < 60:
            new_c = max(5, PHONE_ADAPTIVE_C - 5)
        elif image_brightness > 200:
            new_c = PHONE_ADAPTIVE_C + 3
        else:
            new_c = PHONE_ADAPTIVE_C

        if new_c != PHONE_ADAPTIVE_C:
            adjustments['PHONE_ADAPTIVE_C'] = new_c
            PHONE_ADAPTIVE_C = new_c

    if image_contrast is not None:
        if image_contrast < 20:
            new_c = PHONE_ADAPTIVE_C - 3
            if new_c != PHONE_ADAPTIVE_C:
                adjustments['PHONE_ADAPTIVE_C'] = new_c
                PHONE_ADAPTIVE_C = new_c

    total_pieces = PUZZLE_WIDTH * PUZZLE_HEIGHT
    expected_piece_ratio = 1.0 / (total_pieces * 2)
    min_ratio = max(0.001, min(0.02, expected_piece_ratio))
    if abs(min_ratio - PHONE_MIN_PIECE_AREA_RATIO) > 0.001:
        adjustments['PHONE_MIN_PIECE_AREA_RATIO'] = min_ratio
        PHONE_MIN_PIECE_AREA_RATIO = min_ratio

    if adjustments:
        print(f"Auto-tuned parameters: {adjustments}")

    return adjustments


def validate_config():
    """
    Validate configuration parameters.
    Returns list of validation errors (empty = all valid).
    """
    errors = []

    if PUZZLE_WIDTH < 1 or PUZZLE_HEIGHT < 1:
        errors.append(f"Invalid puzzle dimensions: {PUZZLE_WIDTH}x{PUZZLE_HEIGHT}")

    if MODE == 'phone':
        if PHONE_ADAPTIVE_BLOCK_SIZE % 2 == 0:
            errors.append(
                f"PHONE_ADAPTIVE_BLOCK_SIZE must be odd, got {PHONE_ADAPTIVE_BLOCK_SIZE}"
            )
        if PHONE_BLUR_KERNEL % 2 == 0:
            errors.append(
                f"PHONE_BLUR_KERNEL must be odd, got {PHONE_BLUR_KERNEL}"
            )
        if PHONE_MORPH_KERNEL % 2 == 0:
            errors.append(
                f"PHONE_MORPH_KERNEL must be odd, got {PHONE_MORPH_KERNEL}"
            )
        if not (0 < PHONE_MIN_PIECE_AREA_RATIO < 1):
            errors.append(
                f"PHONE_MIN_PIECE_AREA_RATIO must be in (0, 1), got {PHONE_MIN_PIECE_AREA_RATIO}"
            )
        if PHONE_HASH_THRESHOLD < 0:
            errors.append(
                f"PHONE_HASH_THRESHOLD must be >= 0, got {PHONE_HASH_THRESHOLD}"
            )
        if PHONE_DUPLICATE_GEOMETRIC_THRESHOLD < 0:
            errors.append(
                f"PHONE_DUPLICATE_GEOMETRIC_THRESHOLD must be >= 0, "
                f"got {PHONE_DUPLICATE_GEOMETRIC_THRESHOLD}"
            )

    return errors
