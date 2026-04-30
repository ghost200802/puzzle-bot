#!/usr/bin/env python3
"""Try different alpha thresholds to find the one that separates pieces."""
from PIL import Image
import numpy as np
from scipy.ndimage import label as ndlabel

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
alpha = rgba[:, :, 3]

print("Alpha threshold -> connected components:")
for thresh in [128, 200, 240, 245, 248, 250, 252, 253, 254, 255]:
    binary = (alpha >= thresh).astype(np.uint8)
    labeled, num = ndlabel(binary)
    total_area = np.sum(binary)
    areas = []
    for i in range(1, num + 1):
        a = np.sum(labeled == i)
        if a > 100:
            areas.append(a)
    valid = len(areas)
    print(f'  thresh={thresh}: {num} components, {valid} with area>100, total_area={total_area}')
    if 10 <= valid <= 200:
        print(f'    *** PROMISING: {valid} pieces! ***')
        areas.sort(reverse=True)
        for j, a in enumerate(areas[:10]):
            ys, xs = np.where(labeled == (np.argsort([np.sum(labeled == i) for i in range(1, num + 1)])[::-1][j] + 1))
        # Just show area stats
        print(f'    Areas (top 10): {areas[:10]}')
        print(f'    Areas (min/max): {min(areas)} / {max(areas)}')
