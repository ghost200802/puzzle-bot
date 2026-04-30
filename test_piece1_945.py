import sys
sys.path.insert(0, 'src')
from common.vector import Vector, load_and_vectorize
from common.find_islands import save_island_as_bmp
import os, json
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel

INPUT_IMAGE = 'input/puzzles/1.png'
OUTPUT_DIR = 'output/test_945'
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = Image.open(INPUT_IMAGE)
rgba = np.array(img)
rgb = rgba[:, :, :3]
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

binary_detect = (gray > 50).astype(np.uint8)
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary_detect = cv2.morphologyEx(binary_detect, cv2.MORPH_CLOSE, kernel3)
binary_detect = cv2.morphologyEx(binary_detect, cv2.MORPH_OPEN, kernel3)

labeled, num = ndlabel(binary_detect)
print(f"Detected {num} pieces")

raw_pieces = []
for i in range(1, num + 1):
    area = np.sum(labeled == i)
    if area < 1500:
        continue
    ys, xs = np.where(labeled == i)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    raw_pieces.append({
        'label': i, 'area': area,
        'bbox': (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
        'size': (w, h), 'max_dim': max(w, h),
    })

typical_max_dim = int(np.median([p['max_dim'] for p in raw_pieces]))
scale_factor = 945 / typical_max_dim
print(f"Typical piece size: {typical_max_dim}px, scale_factor: {scale_factor:.2f}x")

# Process piece_1 only
rp = raw_pieces[0]
x0, y0, x1, y1 = rp['bbox']
pad = max(3, int(typical_max_dim * 0.05))
py0, py1 = max(0, y0 - pad), min(gray.shape[0], y1 + pad)
px0, px1 = max(0, x0 - pad), min(gray.shape[1], x1 + pad)

orig_crop = gray[py0:py1, px0:px1]
h, w = orig_crop.shape
new_w = max(int(w * scale_factor), 10)
new_h = max(int(h * scale_factor), 10)
print(f"Crop: {w}x{h}, target: {new_w}x{new_h}")

scaled = cv2.resize(orig_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
blur_k = max(3, int(scale_factor * 0.8))
if blur_k % 2 == 0:
    blur_k += 1
blurred = cv2.GaussianBlur(scaled, (blur_k, blur_k), 0)
_, smooth_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
smooth_binary = (smooth_binary > 127).astype(np.uint8)

kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
smooth_binary = cv2.morphologyEx(smooth_binary, cv2.MORPH_OPEN, kern)
smooth_binary = cv2.morphologyEx(smooth_binary, cv2.MORPH_CLOSE, kern)

lbl, n = ndlabel(smooth_binary)
if n > 1:
    best = max(range(1, n + 1), key=lambda j: np.sum(lbl == j))
    smooth_binary = (lbl == best).astype(np.uint8)

bmp_path = os.path.join(OUTPUT_DIR, 'piece_1.bmp')
save_island_as_bmp(smooth_binary, bmp_path)
print(f"Saved: {bmp_path}")

# Check saved BMP
from common.util import load_bmp_as_binary_pixels
bp, bw, bh = load_bmp_as_binary_pixels(bmp_path)
print(f"BMP loaded: {bw}x{bh}, ones={np.sum(bp==1)}, zeros={np.sum(bp==0)}")

# Vectorize
v = Vector.from_file(bmp_path, 1)
print(f"\nVector: dim={v.dim}, scalar={v.scalar:.2f}")
v.find_border_raster()
print(f"Border pixels: {np.sum(v.border==1)}")
v.vectorize()
print(f"Vertices: {len(v.vertices)}, Centroid: {v.centroid}")

try:
    v.find_four_corners()
    print(f"\nCorners: {v.corner_indices}")
    for i, c in enumerate(v.corners):
        print(f"  Corner {i}: ({c[0]:.1f}, {c[1]:.1f})")
    
    v.extract_four_sides()
    print(f"\nSides:")
    for i, side in enumerate(v.sides):
        first = side.vertices[0]
        last = side.vertices[-1]
        print(f"  Side {i}: {len(side.vertices)} verts, ({first[0]:.0f},{first[1]:.0f})->({last[0]:.0f},{last[1]:.0f}), is_edge={side.is_edge}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
