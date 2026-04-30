#!/usr/bin/env python3
"""
Analyze the alpha channel for anti-aliased edge information.
"""
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
rgb = rgba[:, :, :3]
alpha = rgba[:, :, 3].astype(np.float64)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# Step 1: Use gray to find piece regions
binary = (gray > 50).astype(np.uint8)
labeled, num = ndlabel(binary)

# Find a sample piece
sample_id = 1
for i in range(1, num + 1):
    area = np.sum(labeled == i)
    if area > 2000:
        sample_id = i
        break

piece_mask = (labeled == sample_id)
ys, xs = np.where(piece_mask)
y0, y1 = ys.min(), ys.max() + 1
x0, x1 = xs.min(), xs.max() + 1

print(f"Sample piece {sample_id}: bbox=({x0},{y0})-({x1},{y1}), size={x1-x0}x{y1-y0}")

# Look at alpha values at the piece boundary
# Find boundary pixels (piece pixels adjacent to non-piece pixels)
from scipy.ndimage import binary_dilation
dilated = binary_dilation(piece_mask)
boundary = dilated & ~piece_mask
boundary_alpha = alpha[boundary]
print(f"\nBoundary alpha values: {np.unique(boundary_alpha)}")

# Look at alpha values inside the piece
inside_alpha = alpha[piece_mask]
unique_inside = np.unique(inside_alpha)
print(f"Inside alpha unique values: {unique_inside}")
print(f"Inside alpha min={inside_alpha.min()}, max={inside_alpha.max()}")

# The key insight: alpha has anti-aliased values at edges
# We can use alpha as a smooth mask

# Strategy: combine gray-based separation with alpha smoothness
# 1. Use gray > 50 to determine which piece each pixel belongs to
# 2. Use alpha as the smooth edge mask for each piece
# 3. Smooth and resize properly

# Let's create a smooth mask for the sample piece using alpha
piece_alpha_crop = alpha[y0:y1, x0:x1]
piece_gray_crop = gray[y0:y1, x0:x1]
piece_labeled_crop = labeled[y0:y1, x0:x1]

# Method: alpha channel smoothed
smooth_mask = piece_alpha_crop.copy()
smooth_mask[piece_labeled_crop != sample_id] = 0

# Apply Gaussian blur to smooth the edges
smooth_blurred = cv2.GaussianBlur(smooth_mask, (3, 3), 0)
smooth_binary = (smooth_blurred > 30).astype(np.uint8)

# Compare with original binary
orig_binary = (piece_labeled_crop == sample_id).astype(np.uint8)

print(f"\nOriginal binary edge pixels: {np.sum(np.abs(np.diff(orig_binary, axis=ax)).sum() for ax in [0,1])}")
print(f"Smooth binary edge pixels: {np.sum(np.abs(np.diff(smooth_binary, axis=ax)).sum() for ax in [0,1])}")

# Save comparison
os.makedirs('output/smooth_test', exist_ok=True)
cv2.imwrite('output/smooth_test/orig_binary.png', orig_binary * 255)
cv2.imwrite('output/smooth_test/smooth_binary.png', smooth_binary * 255)

# Now upscale both to 500px and compare
h, w = orig_binary.shape
scale = 500 / max(w, h)
new_w, new_h = int(w * scale), int(h * scale)

orig_upscaled = cv2.resize(orig_binary * 255, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
orig_upscaled = (orig_upscaled > 127).astype(np.uint8)

smooth_upscaled = cv2.resize(smooth_binary * 255, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
smooth_upscaled = (smooth_upscaled > 127).astype(np.uint8)

cv2.imwrite('output/smooth_test/orig_upscaled.png', orig_upscaled * 255)
cv2.imwrite('output/smooth_test/smooth_upscaled.png', smooth_upscaled * 255)

print(f"\nUpscaled size: {new_w}x{new_h}")
print(f"Orig upscaled perimeter pixels: {np.sum(cv2.Canny(orig_upscaled*255, 100, 200) > 0)}")
print(f"Smooth upscaled perimeter pixels: {np.sum(cv2.Canny(smooth_upscaled*255, 100, 200) > 0)}")

import os
