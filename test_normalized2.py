import sys
sys.path.insert(0, 'src')
from common.vector import Vector
from PIL import Image
import numpy as np
import cv2
from shapely.geometry import Polygon, Point

INPUT_BMP = 'output/puzzle_run/3_vector/piece_1.bmp'
OUTPUT_DIR = 'output/test_normalized'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = Image.open(INPUT_BMP)
binary = np.array(img).astype(np.uint8)

ys, xs = np.where(binary == 1)
min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()
binary_crop = binary[min_y:max_y+1, min_x:max_x+1].copy()
h, w = binary_crop.shape
max_dim = max(w, h)

target_size = 500
scale = target_size / max_dim
new_w = max(int(w * scale), 1)
new_h = max(int(h * scale), 1)

scaled = cv2.resize(binary_crop.astype(np.uint8) * 255, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
scaled = (scaled > 127).astype(np.uint8)

# Save using save_island_as_bmp  
scaled_bmp_path = os.path.join(OUTPUT_DIR, 'piece_1_normalized.bmp')
from common.find_islands import save_island_as_bmp
save_island_as_bmp(scaled, scaled_bmp_path)

# Verify
from common.util import load_bmp_as_binary_pixels
bp, bw, bh = load_bmp_as_binary_pixels(scaled_bmp_path)
print(f"Reloaded: {bw}x{bh}, ones={np.sum(bp==1)}, zeros={np.sum(bp==0)}")

# Load and vectorize
v = Vector.from_file(scaled_bmp_path, 1)
print(f"dim={v.dim}, scalar={v.scalar:.2f}")

v.find_border_raster()
print(f"Border pixels: {np.sum(v.border==1)}")

# Run vectorize step by step
indices = np.argwhere(v.border == 1)
sy, sx = tuple(indices[0])
v.vertices = [(sx, sy)]
cx, cy = sx, sy
p_angle = 0
import math
closed = False
while not closed:
    neighbors = [
        (cx, cy - 1), (cx + 1, cy - 1), (cx + 1, cy),
        (cx + 1, cy + 1), (cx, cy + 1), (cx - 1, cy + 1),
        (cx - 1, cy), (cx - 1, cy - 1),
    ]
    shift = int(round(p_angle * float(len(neighbors))/(2 * math.pi)))
    bx, by = cx, cy
    for i in range(0 + shift, 8 + shift):
        nx, ny = neighbors[i % len(neighbors)]
        n = v.border[ny][nx]
        if n == 1:
            dx, dy = nx - cx, ny - cy
            abs_angle = math.atan2(dy, dx)
            rel_angle = abs_angle - p_angle
            p_angle = abs_angle
            if rel_angle < 0:
                rel_angle += 2 * math.pi
            if v.vertices[-1] != (cx, cy):
                v.vertices.append((cx, cy))
            cx, cy = nx, ny
            if cx == sx and cy == sy:
                closed = True
            break
    if bx == cx and by == cy:
        print("STUCK IN LOOP!")
        break

v.centroid = v.centroid if hasattr(v, 'centroid') else None
from common import util
v.centroid = util.centroid(v.vertices)
print(f"Total vertices: {len(v.vertices)}, Centroid: {v.centroid}")

# Check if polygon is valid for incenter
poly = Polygon(v.vertices)
print(f"Polygon valid: {poly.is_valid}, area: {poly.area}")

# Compute incenter manually
try:
    v.incenter = util.incenter(v.vertices)
    print(f"Incenter: {v.incenter}")
except Exception as e:
    print(f"Incenter error: {e}")
    v.incenter = v.centroid

# Now find corners
v.find_four_corners()
print(f"\nCorners: {v.corner_indices}")
for i, c in enumerate(v.corners):
    print(f"  Corner {i}: ({c[0]:.1f}, {c[1]:.1f})")
