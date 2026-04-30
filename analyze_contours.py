#!/usr/bin/env python3
"""
Analyze piece contours to understand their shapes.
Check if they have jigsaw tabs/blanks.
"""
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel
import os

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
rgb = rgba[:, :, :3]
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

os.makedirs('output/contour_analysis', exist_ok=True)

# Use a lower threshold to capture full piece shapes
for thresh in [20, 50, 100]:
    binary = (gray > thresh).astype(np.uint8)
    labeled, num = ndlabel(binary)
    
    # Get the 5 largest pieces
    pieces = []
    for i in range(1, num + 1):
        area = np.sum(labeled == i)
        if area > 500:
            pieces.append((i, area))
    pieces.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n=== thresh={thresh}: analyzing top 5 pieces ===")
    
    for idx, (i, area) in enumerate(pieces[:5]):
        piece_mask = (labeled == i).astype(np.uint8) * 255
        ys, xs = np.where(piece_mask > 0)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        
        piece_crop = piece_mask[y0:y1, x0:x1]
        
        # Find contours
        contours, _ = cv2.findContours(piece_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            
            # Approximate contour to polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convexity defects
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                n_defects = len(defects) if defects is not None else 0
            else:
                n_defects = 0
            
            h, w = piece_crop.shape
            print(f"  Piece {i}: {w}x{h}, area={area}, perimeter={perimeter:.0f}, "
                  f"approx_vertices={len(approx)}, convexity_defects={n_defects}")
            
            # Save visualization
            vis = cv2.cvtColor(piece_crop, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 1)
            cv2.drawContours(vis, [approx], -1, (0, 0, 255), 2)
            cv2.imwrite(f'output/contour_analysis/t{thresh}_piece{idx}.png', vis)

# Also check: are the pieces actually jigsaw-shaped?
# Use thresh=50 and check the convexity defects of ALL pieces
print("\n=== Convexity defect analysis (thresh=50) ===")
binary = (gray > 50).astype(np.uint8)
labeled, num = ndlabel(binary)

defect_counts = []
for i in range(1, num + 1):
    area = np.sum(labeled == i)
    if area < 500:
        continue
    piece_mask = (labeled == i).astype(np.uint8) * 255
    ys, xs = np.where(piece_mask > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    piece_crop = piece_mask[y0:y1, x0:x1]
    
    contours, _ = cv2.findContours(piece_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            n_defects = len(defects) if defects is not None else 0
        else:
            n_defects = 0
        defect_counts.append(n_defects)

if defect_counts:
    print(f"  Pieces: {len(defect_counts)}")
    print(f"  Defect count range: {min(defect_counts)} - {max(defect_counts)}")
    print(f"  Defect count mean: {np.mean(defect_counts):.1f}")
    print(f"  Pieces with 0 defects (convex/rectangular): {sum(1 for d in defect_counts if d == 0)}")
    print(f"  Pieces with 1-3 defects: {sum(1 for d in defect_counts if 1 <= d <= 3)}")
    print(f"  Pieces with 4+ defects (likely jigsaw): {sum(1 for d in defect_counts if d >= 4)}")
