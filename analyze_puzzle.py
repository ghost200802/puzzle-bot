"""Analyze the puzzle image to understand its structure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel, distance_transform_edt

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
bgr = cv2.cvtColor(rgba[:,:,:3], cv2.COLOR_RGB2BGR)
alpha = rgba[:,:,3]
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

print(f"Image size: {rgba.shape}")
print(f"Alpha: min={alpha.min()}, max={alpha.max()}")
print(f"Alpha=0: {np.sum(alpha==0)}, Alpha=255: {np.sum(alpha==255)}")

# 1. Get foreground mask
mask = (alpha > 128).astype(np.uint8)

# 2. Distance transform - find centers of pieces
dist = distance_transform_edt(mask)
print(f"Distance transform: max={dist.max():.1f}, mean={dist.mean():.1f}")

# 3. Find local maxima as seed points
from scipy.ndimage import maximum_filter
local_max = maximum_filter(dist, size=30)
peaks = (dist == local_max) & (dist > 15)
labeled_peaks, n_peaks = ndlabel(peaks.astype(np.uint8))
print(f"Seed points found: {n_peaks}")

# 4. Watershed
# Sure foreground = peaks
sure_fg = np.zeros_like(mask, dtype=np.int32)
for i in range(1, n_peaks + 1):
    ys, xs = np.where(labeled_peaks == i)
    if len(ys) > 0:
        cy, cx = ys[len(ys)//2], xs[len(xs)//2]
        sure_fg[cy, cx] = i

# Sure background = dilated mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
sure_bg = cv2.dilate(mask, kernel, iterations=3)

# Unknown region
unknown = cv2.subtract(sure_bg, (sure_fg > 0).astype(np.uint8))

# Markers for watershed
markers = cv2.watershed(bgr, sure_fg.copy())
# Mark unknown as 0
markers[unknown > 0] = 0

# Count unique labels (excluding -1 which is boundary)
unique_labels = set(np.unique(markers)) - {0, -1}
print(f"Watershed regions: {len(unique_labels)}")

# Count pieces with reasonable size
piece_count = 0
for label_id in unique_labels:
    size = np.sum(markers == label_id)
    if size > 1000:
        piece_count += 1
        print(f"  Region {label_id}: {size} pixels")

print(f"\nFinal piece count: {piece_count}")
print(f"Suggested grid: trying common sizes for {piece_count} pieces")
