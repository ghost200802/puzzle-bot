import sys
sys.path.insert(0, 'src')
from common.vector import Vector
from PIL import Image
import numpy as np
import cv2

INPUT_BMP = 'output/puzzle_run/3_vector/piece_1.bmp'
OUTPUT_DIR = 'output/test_normalized'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = Image.open(INPUT_BMP)
print(f"Original piece_1.bmp size: {img.size}")

binary = np.array(img)

if img.mode == '1':
    binary = binary.astype(np.uint8)
elif binary.ndim == 3:
    binary = (binary[:, :, 0] > 127).astype(np.uint8)
else:
    binary = (binary > 127).astype(np.uint8)

print(f"Binary unique: {np.unique(binary)}, shape: {binary.shape}")
print(f"Pixel 0 count: {np.sum(binary==0)}, Pixel 1 count: {np.sum(binary==1)}")

ys, xs = np.where(binary == 1)
min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()
binary_crop = binary[min_y:max_y+1, min_x:max_x+1].copy()
h, w = binary_crop.shape
max_dim = max(w, h)
print(f"Cropped to: {w}x{h}, max_dim={max_dim}")

target_size = 500
scale = target_size / max_dim
new_w = max(int(w * scale), 1)
new_h = max(int(h * scale), 1)
print(f"Scale factor: {scale:.3f}, target: {new_w}x{new_h}")

scaled = cv2.resize(binary_crop.astype(np.uint8) * 255, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
scaled = (scaled > 127).astype(np.uint8)
print(f"Scaled binary shape: {scaled.shape}, unique: {np.unique(scaled)}")

scaled_bmp_path = os.path.join(OUTPUT_DIR, 'piece_1_normalized.bmp')
from common.find_islands import save_island_as_bmp
save_island_as_bmp(scaled, scaled_bmp_path)
print(f"Saved: {scaled_bmp_path}")

# Verify the saved BMP can be loaded correctly
from common.util import load_bmp_as_binary_pixels
bp, bw, bh = load_bmp_as_binary_pixels(scaled_bmp_path)
print(f"Reloaded BMP: {bw}x{bh}, ones={np.sum(bp==1)}, zeros={np.sum(bp==0)}")

v = Vector.from_file(scaled_bmp_path, 1)
print(f"\nVector: dim={v.dim}, scalar={v.scalar:.2f}")
v.find_border_raster()
v.vectorize()
print(f"Total vertices: {len(v.vertices)}, Centroid: {v.centroid}")

v.find_four_corners()
print(f"\nSelected corners:")
for i, corner in enumerate(v.corners):
    print(f"  Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f})")

print(f"\nCorner indices: {v.corner_indices}")

try:
    v.extract_four_sides()
    print(f"\nExtracted sides:")
    for i, side in enumerate(v.sides):
        n_verts = len(side.vertices)
        first = side.vertices[0]
        last = side.vertices[-1]
        print(f"  Side {i}: {n_verts} vertices, ({first[0]:.1f},{first[1]:.1f}) -> ({last[0]:.1f},{last[1]:.1f}), is_edge={side.is_edge}, angle={round(side.angle*180/3.14159, 1)}°")
except Exception as e:
    print(f"extract_four_sides error: {e}")
