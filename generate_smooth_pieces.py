#!/usr/bin/env python3
"""
Generate smooth, high-resolution piece BMPs from the puzzle image.

Key insight: pieces are only ~64x64 pixels in the source image.
We need to:
1. Extract piece contours from the source image
2. Smooth the contours
3. Render them at high resolution (1000px) for vectorization
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

INPUT_IMAGE = 'input/puzzles/1.png'
OUTPUT_DIR = 'output/smooth_pieces'

img = Image.open(INPUT_IMAGE)
rgba = np.array(img)
rgb = rgba[:, :, :3]
alpha = rgba[:, :, 3]
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# Use gray to find piece regions
binary_low = (gray > 50).astype(np.uint8)

# Clean up: morphological operations to smooth the binary
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary_clean = cv2.morphologyEx(binary_low, cv2.MORPH_CLOSE, kernel3)
binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel3)

labeled, num = ndlabel(binary_clean)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/raw', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/smooth', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/comparison', exist_ok=True)

TARGET_SIZE = 1000

pieces_data = []
for i in range(1, num + 1):
    area = np.sum(labeled == i)
    if area < 1500:
        continue
    
    piece_mask = (labeled == i).astype(np.uint8) * 255
    ys, xs = np.where(piece_mask > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    
    # Add padding
    pad = 5
    py0, py1 = max(0, y0 - pad), min(rgba.shape[0], y1 + pad)
    px0, px1 = max(0, x0 - pad), min(rgba.shape[1], x1 + pad)
    
    piece_crop = piece_mask[py0:py1, px0:px1]
    h, w = piece_crop.shape
    
    # Save raw (jagged) version
    raw_scale = TARGET_SIZE / max(w, h)
    raw_w, raw_h = int(w * raw_scale), int(h * raw_scale)
    raw_upscaled = cv2.resize(piece_crop, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
    raw_binary = (raw_upscaled > 127).astype(np.uint8)
    
    # Find contour on original small image
    contours, _ = cv2.findContours(piece_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        continue
    contour = max(contours, key=cv2.contourArea)
    
    # Smooth the contour using Chaikin's algorithm or simple averaging
    # Method 1: Gaussian blur on the upscaled image
    smooth_upscaled = cv2.resize(piece_crop, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)
    
    # Apply Gaussian blur to smooth jagged edges
    blur_size = max(3, int(raw_scale * 1.5))
    if blur_size % 2 == 0:
        blur_size += 1
    smooth_blurred = cv2.GaussianBlur(smooth_upscaled, (blur_size, blur_size), 0)
    smooth_binary = (smooth_blurred > 127).astype(np.uint8)
    
    # Method 2: Contour-based smoothing - draw smooth contour at high resolution
    # Scale contour points to target resolution
    contour_scaled = contour.astype(np.float64) * raw_scale
    contour_scaled = contour_scaled.astype(np.int32)
    
    smooth_contour_img = np.zeros((raw_h, raw_w), dtype=np.uint8)
    cv2.drawContours(smooth_contour_img, [contour_scaled], -1, 255, thickness=cv2.FILLED)
    
    # Apply slight blur to the contour-rendered image for anti-aliased edges
    smooth_contour_blurred = cv2.GaussianBlur(smooth_contour_img, (5, 5), 0)
    smooth_contour_binary = (smooth_contour_blurred > 127).astype(np.uint8)
    
    # Compare the three methods
    piece_id = len(pieces_data) + 1
    
    # Save comparison for first 5 pieces
    if piece_id <= 5:
        comparison = np.hstack([
            np.zeros((raw_h, 5), dtype=np.uint8) + 128,
            raw_binary * 255,
            np.zeros((raw_h, 5), dtype=np.uint8) + 128,
            smooth_binary * 255,
            np.zeros((raw_h, 5), dtype=np.uint8) + 128,
            smooth_contour_binary * 255,
        ])
        cv2.imwrite(f'{OUTPUT_DIR}/comparison/piece_{piece_id:03d}_comparison.png', comparison)
    
    # Save the smooth contour version (best quality)
    cv2.imwrite(f'{OUTPUT_DIR}/smooth/piece_{piece_id:03d}.bmp', smooth_contour_binary * 255)
    cv2.imwrite(f'{OUTPUT_DIR}/raw/piece_{piece_id:03d}.bmp', raw_binary * 255)
    
    pieces_data.append({
        'id': piece_id,
        'origin': (int(px0), int(py0)),
        'centroid': ((px0 + px1) / 2.0, (py0 + py1) / 2.0),
        'area': int(area),
        'source_size': (w, h),
        'target_size': (raw_w, raw_h),
    })

print(f"Generated {len(pieces_data)} pieces")
print(f"Source piece size: ~{pieces_data[0]['source_size'][0]}x{pieces_data[0]['source_size'][1]}")
print(f"Target size: ~{pieces_data[0]['target_size'][0]}x{pieces_data[0]['target_size'][1]}")
print(f"\nSaved to {OUTPUT_DIR}/")
print(f"  comparison/: side-by-side comparison (raw vs blur vs contour)")
print(f"  smooth/: contour-smoothed BMPs for vectorization")
print(f"  raw/: raw nearest-neighbor upscaled BMPs")

# Now test vectorization on a few pieces
print("\n--- Testing vectorization on smooth pieces ---")
from common import vector
success_smooth = 0
success_raw = 0
test_dir = 'output/smooth_vector_test'
os.makedirs(test_dir, exist_ok=True)

for pd in pieces_data[:10]:
    pid = pd['id']
    
    # Test smooth version
    bmp_path = f'{OUTPUT_DIR}/smooth/piece_{pid:03d}.bmp'
    h, w = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE).shape
    metadata = {
        'original_photo_name': '1.png',
        'photo_space_origin': pd['origin'],
        'photo_space_centroid': [w // 2, h // 2],
        'photo_width': w,
        'photo_height': h,
        'is_complete': True,
    }
    args = [bmp_path, pid, test_dir, metadata, (0, 0), 1.0, False]
    try:
        vector.load_and_vectorize(args)
        success_smooth += 1
        print(f"  Piece {pid} (smooth): OK")
    except Exception as e:
        print(f"  Piece {pid} (smooth): FAILED - {e}")
    
    # Test raw version for comparison
    bmp_path_raw = f'{OUTPUT_DIR}/raw/piece_{pid:03d}.bmp'
    h, w = cv2.imread(bmp_path_raw, cv2.IMREAD_GRAYSCALE).shape
    metadata_raw = metadata.copy()
    metadata_raw['photo_width'] = w
    metadata_raw['photo_height'] = h
    args_raw = [bmp_path_raw, pid + 100, test_dir, metadata_raw, (0, 0), 1.0, False]
    try:
        vector.load_and_vectorize(args_raw)
        success_raw += 1
        print(f"  Piece {pid} (raw): OK")
    except Exception as e:
        print(f"  Piece {pid} (raw): FAILED - {e}")

print(f"\nSmooth vectorization: {success_smooth}/10 success")
print(f"Raw vectorization: {success_raw}/10 success")
