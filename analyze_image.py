#!/usr/bin/env python3
"""Quick analysis of the puzzle image to understand its structure."""
from PIL import Image
import numpy as np
from scipy.ndimage import label as ndlabel

img = Image.open('input/puzzles/1.png')
print(f'Size: {img.size}')
print(f'Mode: {img.mode}')

rgba = np.array(img)
print(f'Shape: {rgba.shape}')
print(f'dtype: {rgba.dtype}')

if rgba.shape[2] == 4:
    alpha = rgba[:, :, 3]
    print(f'Alpha range: {alpha.min()} - {alpha.max()}')
    unique_alpha = np.unique(alpha)
    print(f'Alpha unique values: {unique_alpha}')
    print(f'Non-zero alpha pixels: {np.sum(alpha > 0)}')
    print(f'Total pixels: {alpha.size}')

    binary = (alpha > 128).astype(np.uint8)
    labeled, num = ndlabel(binary)
    print(f'\nConnected components (alpha>128): {num}')

    for i in range(1, min(num + 1, 31)):
        area = np.sum(labeled == i)
        ys, xs = np.where(labeled == i)
        print(f'  Component {i}: area={area}, bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})')

    if num > 30:
        print(f'  ... and {num - 30} more components')

    # Also check RGB channels for color discontinuity
    rgb = rgba[:, :, :3]
    gray = np.array(img.convert('L'))
    print(f'\nGray value range in foreground: {gray[binary > 0].min()} - {gray[binary > 0].max()}')
    print(f'Gray mean in foreground: {gray[binary > 0].mean():.1f}')
else:
    print('Not RGBA, checking grayscale...')
    gray = np.array(img.convert('L'))
    print(f'Gray range: {gray.min()} - {gray.max()}')
