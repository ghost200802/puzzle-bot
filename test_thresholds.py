#!/usr/bin/env python3
"""
Analyze what the pieces look like at different thresholds.
Find the threshold that gives the best jigsaw piece shapes.
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

os.makedirs('output/thresh_test', exist_ok=True)

for thresh in [10, 20, 30, 50, 80, 100, 128, 145]:
    binary = (gray > thresh).astype(np.uint8)
    labeled, num = ndlabel(binary)
    
    pieces_info = []
    for i in range(1, num + 1):
        area = np.sum(labeled == i)
        if area > 500:
            ys, xs = np.where(labeled == i)
            h = ys.max() - ys.min() + 1
            w = xs.max() - xs.min() + 1
            aspect = max(w, h) / max(min(w, h), 1)
            pieces_info.append((i, area, w, h, aspect))
    
    pieces_info.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nthresh={thresh}: {num} total, {len(pieces_info)} with area>500")
    if pieces_info:
        areas = [p[1] for p in pieces_info[:10]]
        aspects = [p[4] for p in pieces_info[:10]]
        print(f"  Top 10 areas: {areas}")
        print(f"  Top 10 aspect ratios: {[f'{a:.2f}' for a in aspects]}")
    
    # Save a few sample pieces for visual inspection
    sample_dir = f'output/thresh_test/t{thresh}'
    os.makedirs(sample_dir, exist_ok=True)
    for j, (i, area, w, h, aspect) in enumerate(pieces_info[:5]):
        piece_mask = (labeled == i).astype(np.uint8)
        ys, xs = np.where(piece_mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        
        piece_rgb = rgb[y0:y1, x0:x1].copy()
        for ch in range(3):
            piece_rgb[:, :, ch][piece_mask[y0:y1, x0:x1] == 0] = 0
        
        cv2.imwrite(f'{sample_dir}/sample_{j}.png', cv2.cvtColor(piece_rgb, cv2.COLOR_RGB2BGR))

print("\nSaved sample pieces to output/thresh_test/")
