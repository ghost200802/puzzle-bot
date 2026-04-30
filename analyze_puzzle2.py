"""Analyze puzzle image structure - find gaps between pieces."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel, distance_transform_edt, binary_dilation

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
bgr = cv2.cvtColor(rgba[:,:,:3], cv2.COLOR_RGB2BGR)
alpha = rgba[:,:,3]
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

mask = (alpha > 128).astype(np.uint8)

# Strategy: find the thin gaps between pieces
# Gaps appear as edges/ridges within the foreground

# 1. Canny edge detection within the foreground
edges = cv2.Canny(gray, 30, 80)
edges_masked = edges & mask
print(f"Canny edges in foreground: {np.sum(edges_masked > 0)} pixels")

# 2. Use gradient magnitude
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
grad_mag = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
grad_masked = grad_mag & mask

# 3. Threshold gradient to find strong edges
_, strong_edges = cv2.threshold(grad_masked, 40, 255, cv2.THRESH_BINARY)
print(f"Strong gradient edges: {np.sum(strong_edges > 0)} pixels")

# 4. Dilate the edges slightly to create separations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilated_edges = cv2.dilate(strong_edges, kernel, iterations=2)

# 5. Remove edges from foreground to create separated pieces
separated = mask & (~dilated_edges.astype(bool)).astype(np.uint8)
print(f"Separated foreground: {np.sum(separated)} pixels (was {np.sum(mask)})")

# 6. Connected components
labeled, num = ndlabel(separated)
print(f"Connected components after edge removal: {num}")

# Filter by size
valid = 0
sizes = []
for i in range(1, num + 1):
    s = np.sum(labeled == i)
    sizes.append((i, s))
sizes.sort(key=lambda x: -x[1])

min_size = np.sum(mask) * 0.005  # at least 0.5% of foreground
for i, (cid, s) in enumerate(sizes):
    if s >= min_size:
        valid += 1
        print(f"  Piece {valid}: component {cid}, {s} pixels")
    else:
        break
print(f"\nValid pieces: {valid}")

# Save debug image
debug = bgr.copy()
for i, (cid, s) in enumerate(sizes[:valid]):
    ys, xs = np.where(labeled == cid)
    color = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    c = color[i % len(color)]
    for y, x in zip(ys[::5], xs[::5]):
        debug[y, x] = c
cv2.imwrite('output/debug_pieces.png', debug)
print("Debug image saved to output/debug_pieces.png")
