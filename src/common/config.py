"""
Common configuration for the puzzle bot

Supports two modes:
  - 'robot': original robot-based puzzle solver
  - 'phone': mobile phone photo-based puzzle assistant
"""

import os


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

# Step 3 takes in piece BMPs and outputs SVGs
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
    PHONE_TARGET_PIECE_SIZE = 500     # Target size for piece normalization
    # Deduplication
    PHONE_HASH_THRESHOLD = 15         # Perceptual hash distance threshold
    PHONE_DUPLICATE_GEOMETRIC_THRESHOLD = 2.2  # Geometric similarity threshold for dedup
    # Segmentation
    PHONE_BG_BRIGHTNESS_THRESHOLD = 128  # Brightness threshold for background detection
    PHONE_BORDER_RATIO_FOR_BG_DETECT = 0.1  # Border ratio for background sampling
