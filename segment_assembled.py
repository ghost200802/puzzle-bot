#!/usr/bin/env python3
"""
Segment an assembled puzzle image into individual pieces using edge detection + watershed.
"""
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel, distance_transform_edt

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
rgb = rgba[:, :, :3]
alpha = rgba[:, :, 3]
mask = (alpha > 128).astype(np.uint8)

print(f"Image: {rgba.shape}")
print(f"Foreground pixels: {mask.sum()}")

# Convert to grayscale for edge detection
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# Apply Canny with various thresholds to find piece boundary seams
print("\nCanny edge detection (foreground only):")
for lo, hi in [(20, 60), (30, 80), (40, 100), (50, 120), (60, 150)]:
    edges = cv2.Canny(gray, lo, hi)
    edges_fg = edges.copy()
    edges_fg[mask == 0] = 0
    n_edge_px = np.sum(edges_fg > 0)
    print(f"  Canny({lo},{hi}): {n_edge_px} edge pixels in foreground")

# Use the best Canny result
edges = cv2.Canny(gray, 30, 80)
edges_fg = edges.copy()
edges_fg[mask == 0] = 0

# Dilate edges slightly to close gaps
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
edges_dilated = cv2.dilate(edges_fg, kernel3, iterations=1)

# Create markers for watershed: edges = 0 (boundary), foreground = 1, background = 0
# Use distance transform on the inverse of edges to find piece centers
no_edge = mask.copy()
no_edge[edges_dilated > 0] = 0

# Distance transform to find centers of regions between edges
dist = distance_transform_edt(no_edge)
print(f"\nDistance transform: max={dist.max():.1f}, mean={dist[no_edge>0].mean():.1f}")

# Find local maxima as seed points
from scipy.ndimage import maximum_filter
local_max = maximum_filter(dist, size=15)
seeds = ((dist == local_max) & (dist > 3) & (no_edge > 0)).astype(np.uint8)
labeled_seeds, n_seeds = ndlabel(seeds)
print(f"Seed points found: {n_seeds}")

# Watershed
markers = labeled_seeds.copy().astype(np.int32)
# Mark background as 0 (will become -1 in watershed output)
bg_label = n_seeds + 1
markers[mask == 0] = bg_label

# Prepare 3-channel image for watershed
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
ws_result = cv2.watershed(bgr, markers)

# Count regions (excluding background)
ws_labels = ws_result.copy()
ws_labels[ws_labels < 1] = 0
ws_labels[ws_labels == bg_label] = 0
unique_labels = np.unique(ws_labels)
unique_labels = unique_labels[unique_labels > 0]
print(f"Watershed regions: {len(unique_labels)}")

# Count valid regions (area > 500)
valid_count = 0
region_areas = []
for lbl in unique_labels:
    area = np.sum(ws_labels == lbl)
    if area > 500:
        valid_count += 1
        region_areas.append(area)

print(f"Valid regions (area>500): {valid_count}")
if region_areas:
    region_areas.sort(reverse=True)
    print(f"Region areas (top 10): {region_areas[:10]}")
    print(f"Region areas (min/max/mean): {min(region_areas)}/{max(region_areas)}/{np.mean(region_areas):.0f}")

# Save visualization
vis = rgb.copy()
colors = np.random.randint(50, 255, (max(unique_labels) + 1, 3), dtype=np.uint8)
for lbl in unique_labels:
    area = np.sum(ws_labels == lbl)
    if area > 500:
        region_mask = ws_labels == lbl
        blended = vis.copy()
        blended[region_mask] = colors[lbl]
        vis = cv2.addWeighted(vis, 0.6, blended, 0.4, 0)

# Draw watershed boundaries
boundary = np.zeros_like(gray, dtype=np.uint8)
boundary[ws_result == -1] = 255
vis[boundary > 0] = [255, 0, 0]

cv2.imwrite('output/watershed_result.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(f"\nSaved watershed visualization to output/watershed_result.png")

# Also save individual piece crops if we got reasonable results
if 5 <= valid_count <= 200:
    print(f"\nExtracting {valid_count} pieces...")
    import os
    os.makedirs('output/pieces', exist_ok=True)
    piece_num = 0
    for lbl in unique_labels:
        area = np.sum(ws_labels == lbl)
        if area < 500:
            continue
        piece_num += 1
        piece_mask = (ws_labels == lbl).astype(np.uint8)
        ys, xs = np.where(piece_mask)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        
        piece_rgb = rgb[y0:y1, x0:x1].copy()
        piece_alpha = piece_mask[y0:y1, x0:x1] * 255
        
        piece_rgba = np.dstack([piece_rgb, piece_alpha])
        piece_img = Image.fromarray(piece_rgba, 'RGBA')
        piece_img.save(f'output/pieces/piece_{piece_num:03d}.png')
    print(f"Saved {piece_num} piece images to output/pieces/")
