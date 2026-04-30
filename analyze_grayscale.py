#!/usr/bin/env python3
"""
Analyze the puzzle image correctly - the background is BLACK, pieces are WHITE.
Use grayscale content (not alpha channel) to find separate pieces.
"""
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
rgb = rgba[:, :, :3]
alpha = rgba[:, :, 3]

gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
print(f"Image shape: {rgba.shape}")
print(f"Gray range: {gray.min()} - {gray.max()}")
print(f"Gray unique values count: {len(np.unique(gray))}")

# The background is black (0), pieces are white (255)
# Try different thresholds on the grayscale image
for thresh in [10, 20, 30, 50, 80, 100, 128, 145, 180, 200]:
    binary = (gray > thresh).astype(np.uint8)
    labeled, num = ndlabel(binary)
    valid = sum(1 for i in range(1, num + 1) if np.sum(labeled == i) > 100)
    print(f"  thresh={thresh}: {num} components, {valid} with area>100")

# Use the standard project threshold (SEG_THRESH=145) and also try adaptive
print("\n--- Using thresh=145 (project default) ---")
binary = (gray > 145).astype(np.uint8)
labeled, num = ndlabel(binary)
print(f"Components: {num}")
areas = []
for i in range(1, num + 1):
    area = np.sum(labeled == i)
    ys, xs = np.where(labeled == i)
    areas.append((i, area, xs.min(), ys.min(), xs.max(), ys.max()))

areas.sort(key=lambda x: x[1], reverse=True)
print(f"Top 15 pieces by area:")
for idx, (i, area, x0, y0, x1, y1) in enumerate(areas[:15]):
    print(f"  Piece {i}: area={area}, bbox=({x0},{y0})-({x1},{y1})")

print(f"\nTotal pieces: {len(areas)}")
if areas:
    just_areas = [a[1] for a in areas]
    print(f"Area range: {min(just_areas)} - {max(just_areas)}")
    print(f"Area mean: {np.mean(just_areas):.0f}, median: {np.median(just_areas):.0f}")

# Save visualization
vis = rgb.copy()
colors_arr = np.random.randint(50, 255, (num + 1, 3), dtype=np.uint8)
for i, area, x0, y0, x1, y1 in areas:
    if area < 100:
        continue
    region_mask = labeled == i
    blended = vis.copy()
    blended[region_mask] = colors_arr[i]
    vis = cv2.addWeighted(vis, 0.5, blended, 0.5, 0)

cv2.imwrite('output/grayscale_segmentation.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(f"\nSaved visualization to output/grayscale_segmentation.png")

# Save individual pieces
import os
os.makedirs('output/pieces_raw', exist_ok=True)
piece_num = 0
for i, area, x0, y0, x1, y1 in areas:
    if area < 100:
        continue
    piece_num += 1
    piece_mask = (labeled == i).astype(np.uint8)
    pad = 2
    py0, py1 = max(0, y0 - pad), min(rgba.shape[0], y1 + pad + 1)
    px0, px1 = max(0, x0 - pad), min(rgba.shape[1], x1 + pad + 1)

    piece_rgb = rgb[py0:py1, px0:px1].copy()
    piece_binary = piece_mask[py0:py1, px0:px1]

    # Black out background in the piece
    for ch in range(3):
        piece_rgb[:, :, ch][piece_binary == 0] = 0

    cv2.imwrite(f'output/pieces_raw/piece_{piece_num:03d}.png', cv2.cvtColor(piece_rgb, cv2.COLOR_RGB2BGR))

print(f"Saved {piece_num} raw pieces to output/pieces_raw/")
