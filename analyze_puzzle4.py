"""Analyze puzzle image - detect piece boundaries via edges."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel, distance_transform_edt

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
rgb = rgba[:,:,:3]
alpha = rgba[:,:,3]
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

mask = (alpha > 128).astype(np.uint8)

# Find foreground bounding box
ys, xs = np.where(mask > 0)
y0, y1 = ys.min(), ys.max()
x0, x1 = xs.min(), xs.max()
print(f"Foreground bbox: x=[{x0},{x1}], y=[{y0},{y1}], size={x1-x0+1}x{y1-y0+1}")

# Crop to foreground
gray_crop = gray[y0:y1+1, x0:x1+1]
bgr_crop = bgr[y0:y1+1, x0:x1+1]
mask_crop = mask[y0:y1+1, x0:x1+1]

print(f"Crop size: {gray_crop.shape}")

# Try Canny with different thresholds
for lo, hi in [(20,60), (30,80), (40,100), (50,120), (10,40)]:
    edges = cv2.Canny(gray_crop, lo, hi)
    edges_fg = edges & mask_crop
    print(f"  Canny({lo},{hi}): {np.sum(edges_fg>0)} edge pixels")

# Use the best Canny
edges = cv2.Canny(gray_crop, 30, 80)
edges_fg = edges & mask_crop

# Dilate edges to create barriers
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
edges_dilated = cv2.dilate(edges_fg, kernel3, iterations=2)

# Distance transform for watershed seeds
no_edges = mask_crop & (~edges_dilated.astype(bool)).astype(np.uint8)
dist = distance_transform_edt(no_edges)
print(f"Distance transform max: {dist.max():.1f}")

# Find seeds using h-minima suppression
from scipy.ndimage import maximum_filter, binary_dilation
local_max = maximum_filter(dist, size=40)
peaks = ((dist == local_max) & (dist > 20)).astype(np.uint8)
labeled_peaks, n_seeds = ndlabel(peaks)
print(f"Seed points: {n_seeds}")

# Show seed locations
seed_centers = []
for i in range(1, n_seeds + 1):
    ys_s, xs_s = np.where(labeled_peaks == i)
    size = len(ys_s)
    if size > 0:
        cy, cx = int(ys_s.mean()), int(xs_s.mean())
        seed_centers.append((cy, cx, i, size))

seed_centers.sort(key=lambda x: (x[0], x[1]))
print("Seed locations:")
for cy, cx, idx, sz in seed_centers[:30]:
    print(f"  Seed {idx}: ({cx},{cy}) size={sz}")

# Try watershed
markers = np.zeros(gray_crop.shape[:2], dtype=np.int32)
for cy, cx, idx, sz in seed_centers:
    markers[cy, cx] = idx

markers = cv2.watershed(bgr_crop, markers)

unique_labels = set(np.unique(markers)) - {0, -1}
print(f"\nWatershed result: {len(unique_labels)} regions")

valid = 0
for lbl in sorted(unique_labels):
    size = np.sum(markers == lbl)
    if size > 500:
        valid += 1
print(f"Valid regions (>500px): {valid}")

# Check grid structure
seed_rows = sorted(set(cy // 100 for cy, cx, _, _ in seed_centers))
seed_cols = sorted(set(cx // 100 for cx, cy, _, _ in seed_centers))
print(f"Seed row groups: {seed_rows}")
print(f"Seed col groups: {seed_cols}")
