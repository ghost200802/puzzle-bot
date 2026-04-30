#!/usr/bin/env python3
"""
Use projection analysis to find the grid structure of the assembled puzzle.
Then use grid-based cutting with watershed refinement.
"""
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel, distance_transform_edt
from scipy.signal import find_peaks

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
rgb = rgba[:, :, :3]
alpha = rgba[:, :, 3]
mask = (alpha > 128).astype(np.uint8)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# Compute gradient magnitude (Sobel)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = np.sqrt(sobelx**2 + sobely**2)
grad_mag[mask == 0] = 0

# Vertical projection (sum of gradients along each column)
v_proj = np.sum(grad_mag, axis=0)
v_proj = v_proj.astype(float)

# Horizontal projection (sum of gradients along each row)
h_proj = np.sum(grad_mag, axis=1)
h_proj = h_proj.astype(float)

# Smooth projections
kernel_size = 5
v_proj_smooth = np.convolve(v_proj, np.ones(kernel_size)/kernel_size, mode='same')
h_proj_smooth = np.convolve(h_proj, np.ones(kernel_size)/kernel_size, mode='same')

# Find peaks in projections (these are the grid lines)
# Grid lines have HIGH gradient because piece boundaries create edges
v_mean = np.mean(v_proj_smooth[mask.any(axis=0)])
h_mean = np.mean(h_proj_smooth[mask.any(axis=1)])

v_peaks, v_props = find_peaks(v_proj_smooth, height=v_mean * 1.5, distance=20, prominence=v_mean * 0.3)
h_peaks, h_props = find_peaks(h_proj_smooth, height=h_mean * 1.5, distance=20, prominence=h_mean * 0.3)

print(f"Vertical boundary peaks (columns): {v_peaks}")
print(f"  Count: {len(v_peaks)}")
print(f"Horizontal boundary peaks (rows): {h_peaks}")
print(f"  Count: {len(h_peaks)}")
print(f"Estimated grid: {len(v_peaks)+1} cols x {len(h_peaks)+1} rows = {(len(v_peaks)+1)*(len(h_peaks)+1)} pieces")

# Define grid cell boundaries
# Add edges of foreground as boundaries
fg_cols = np.where(mask.any(axis=0))[0]
fg_rows = np.where(mask.any(axis=1))[0]
col_start, col_end = fg_cols[0], fg_cols[-1]
row_start, row_end = fg_rows[0], fg_rows[-1]

v_bounds = [col_start] + list(v_peaks) + [col_end]
h_bounds = [row_start] + list(h_peaks) + [row_end]

print(f"\nVertical boundaries: {v_bounds}")
print(f"Horizontal boundaries: {h_bounds}")

# Create grid-based initial segmentation
n_cols = len(v_bounds) - 1
n_rows = len(h_bounds) - 1
print(f"\nGrid: {n_cols} x {n_rows} = {n_cols * n_rows} pieces")

# Create a labeled image where each grid cell has a unique label
grid_labels = np.zeros_like(mask, dtype=np.int32)
piece_id = 1
cell_info = []
for r in range(n_rows):
    for c in range(n_cols):
        r0 = h_bounds[r]
        r1 = h_bounds[r + 1]
        c0 = v_bounds[c]
        c1 = v_bounds[c + 1]
        grid_labels[r0:r1, c0:c1] = piece_id
        cell_info.append((piece_id, r, c, r0, r1, c0, c1))
        piece_id += 1

# Apply mask - only foreground
grid_labels[mask == 0] = 0

# Use grid labels as markers for watershed to refine boundaries
# The watershed will adjust boundaries based on actual edge positions
markers = grid_labels.copy()
markers[mask == 0] = 0

# Prepare edge image for watershed
edges = cv2.Canny(gray, 30, 80)
edges[mask == 0] = 0

# Create a BGR image for watershed
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# Dilate edges to create barriers
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
edge_barriers = cv2.dilate(edges, kernel3, iterations=2)

# Run watershed with grid markers
ws_result = cv2.watershed(bgr, markers)

# Count valid regions
unique_labels = np.unique(ws_result)
unique_labels = unique_labels[unique_labels > 0]
print(f"\nWatershed result: {len(unique_labels)} regions")

# Calculate area stats
areas = []
for lbl in unique_labels:
    area = np.sum(ws_result == lbl)
    if area > 200:
        areas.append((lbl, area))

areas.sort(key=lambda x: x[1], reverse=True)
print(f"Regions with area > 200: {len(areas)}")
if areas:
    just_areas = [a[1] for a in areas]
    print(f"Area stats: min={min(just_areas)}, max={max(just_areas)}, mean={np.mean(just_areas):.0f}")

# Save visualization
vis = rgb.copy()
colors_arr = np.random.randint(50, 255, (max(unique_labels) + 1, 3), dtype=np.uint8)
for lbl, area in areas:
    region_mask = ws_result == lbl
    blended = vis.copy()
    blended[region_mask] = colors_arr[lbl]
    vis = cv2.addWeighted(vis, 0.5, blended, 0.5, 0)

# Draw boundaries
boundary = np.zeros_like(gray, dtype=np.uint8)
boundary[ws_result == -1] = 255
vis[boundary > 0] = [255, 0, 0]

# Draw grid lines
for v in v_peaks:
    vis[:, v] = [0, 255, 0]
for h in h_peaks:
    vis[h, :] = [0, 255, 0]

cv2.imwrite('output/grid_watershed.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(f"\nSaved visualization to output/grid_watershed.png")

# Extract individual pieces
import os
os.makedirs('output/pieces', exist_ok=True)
piece_num = 0
for lbl, area in areas:
    piece_num += 1
    piece_mask = (ws_result == lbl).astype(np.uint8)
    ys, xs = np.where(piece_mask)
    if len(ys) == 0:
        continue
    y0, y1 = max(0, ys.min() - 2), min(rgba.shape[0], ys.max() + 3)
    x0, x1 = max(0, xs.min() - 2), min(rgba.shape[1], xs.max() + 3)

    piece_rgb = rgb[y0:y1, x0:x1].copy()
    piece_alpha = piece_mask[y0:y1, x0:x1] * 255

    piece_rgba = np.dstack([piece_rgb, piece_alpha])
    piece_img = Image.fromarray(piece_rgba, 'RGBA')
    piece_img.save(f'output/pieces/piece_{piece_num:03d}.png')

print(f"\nSaved {piece_num} pieces to output/pieces/")
