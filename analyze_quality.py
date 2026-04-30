#!/usr/bin/env python3
"""
Analyze the quality loss in the segmentation pipeline.

Current (bad):  threshold at low res -> contour -> scale up
Better:         crop original -> scale up -> threshold at high res
"""
import os
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
rgb = rgba[:, :, :3]
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# Binary for labeling
binary = (gray > 50).astype(np.uint8)
labeled, num = ndlabel(binary)

# Find a sample piece
piece_id = 1
for i in range(1, num + 1):
    area = np.sum(labeled == i)
    if area > 2000:
        piece_id = i
        break

piece_mask = (labeled == piece_id).astype(np.uint8)
ys, xs = np.where(piece_mask > 0)
y0, y1 = ys.min(), ys.max() + 1
x0, x1 = xs.min(), xs.max() + 1

print(f"Sample piece: bbox=({x0},{y0})-({x1},{y1}), size={x1-x0}x{y1-y0}")

pad = 5
py0, py1 = max(0, y0 - pad), min(rgba.shape[0], y1 + pad)
px0, px1 = max(0, x0 - pad), min(rgba.shape[1], x1 + pad)

# Original grayscale crop
orig_gray_crop = gray[py0:py1, px0:px1]
h, w = orig_gray_crop.shape
scale = 1000 / max(w, h)
new_w, new_h = int(w * scale), int(h * scale)

print(f"Original crop: {w}x{h}, scale={scale:.1f}, target={new_w}x{new_h}")

# Method 1 (current bad): threshold first, then scale
binary_lowres = (orig_gray_crop > 50).astype(np.uint8) * 255
contours, _ = cv2.findContours(binary_lowres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = max(contours, key=cv2.contourArea)
contour_scaled = (contour.astype(np.float64) * scale).astype(np.int32)
bad_img = np.zeros((new_h, new_w), dtype=np.uint8)
cv2.drawContours(bad_img, [contour_scaled], -1, 255, thickness=cv2.FILLED)
bad_img = cv2.GaussianBlur(bad_img, (5, 5), 0)
bad_binary = (bad_img > 127).astype(np.uint8)

# Method 2 (better): scale original first, then threshold
scaled_gray = cv2.resize(orig_gray_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
# Use alpha-like threshold: at high res, the anti-aliased edges have gradient values
# Apply Gaussian blur for smoothness, then threshold
blurred = cv2.GaussianBlur(scaled_gray, (5, 5), 0)
good_binary = (blurred > 30).astype(np.uint8)

# Clean up noise
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
good_binary = cv2.morphologyEx(good_binary, cv2.MORPH_OPEN, kernel_small)
good_binary = cv2.morphologyEx(good_binary, cv2.MORPH_CLOSE, kernel_small)

# Method 3 (best): use original alpha channel + scale up
alpha_crop = rgba[py0:py1, px0:px1, 3]
scaled_alpha = cv2.resize(alpha_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
alpha_binary = (scaled_alpha > 30).astype(np.uint8)
alpha_binary = cv2.morphologyEx(alpha_binary, cv2.MORPH_OPEN, kernel_small)
alpha_binary = cv2.morphologyEx(alpha_binary, cv2.MORPH_CLOSE, kernel_small)

# Compare: count edge detail (perimeter length, convexity defects)
os.makedirs('output/quality_comparison', exist_ok=True)

for name, bmap in [("bad_contour_method", bad_binary), ("scale_first_method", good_binary), ("alpha_scale_method", alpha_binary)]:
    contour_img = bmap * 255
    contours2, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours2:
        c = max(contours2, key=cv2.contourArea)
        perimeter = cv2.arcLength(c, True)
        hull = cv2.convexHull(c, returnPoints=False)
        n_defects = 0
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(c, hull)
            n_defects = len(defects) if defects is not None else 0
        print(f"\n{name}:")
        print(f"  Area: {np.sum(bmap)}, Perimeter: {perimeter:.0f}, Defects: {n_defects}")
    cv2.imwrite(f'output/quality_comparison/{name}.png', contour_img)

# Side by side comparison
comparison = np.hstack([
    bad_binary * 255,
    np.zeros((new_h, 10), dtype=np.uint8) + 128,
    good_binary * 255,
    np.zeros((new_h, 10), dtype=np.uint8) + 128,
    alpha_binary * 255,
])
cv2.imwrite('output/quality_comparison/side_by_side.png', comparison)
print(f"\nSaved comparisons to output/quality_comparison/")
print(f"Left=bad(contour), Middle=scale_first, Right=alpha_scale")
