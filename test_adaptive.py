#!/usr/bin/env python3
"""Quick test: verify the new adaptive pipeline produces quality pieces."""
import os, sys
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from common.config import PHONE_TARGET_PIECE_SIZE

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
rgb = rgba[:, :, :3]
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

binary_detect = (gray > 50).astype(np.uint8)
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary_detect = cv2.morphologyEx(binary_detect, cv2.MORPH_CLOSE, kernel3)
binary_detect = cv2.morphologyEx(binary_detect, cv2.MORPH_OPEN, kernel3)
labeled, num = ndlabel(binary_detect)

raw = []
for i in range(1, num + 1):
    area = np.sum(labeled == i)
    if area < 1500:
        continue
    ys, xs = np.where(labeled == i)
    raw.append({'label': i, 'area': area,
                'bbox': (xs.min(), ys.min(), xs.max()+1, ys.max()+1),
                'max_dim': max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)})

typical = int(np.median([r['max_dim'] for r in raw]))
scale = PHONE_TARGET_PIECE_SIZE / typical
print(f"Pieces: {len(raw)}, typical={typical}px, target={PHONE_TARGET_PIECE_SIZE}, scale={scale:.1f}x")

os.makedirs('output/adaptive_test', exist_ok=True)

from common import vector
test_dir = 'output/adaptive_test'
success = 0

for rp in raw[:10]:
    x0, y0, x1, y1 = rp['bbox']
    pad = max(3, int(typical * 0.05))
    py0, py1 = max(0, y0-pad), min(gray.shape[0], y1+pad)
    px0, px1 = max(0, x0-pad), min(gray.shape[1], x1+pad)
    
    crop = gray[py0:py1, px0:px1]
    h, w = crop.shape
    nw, nh = max(int(w*scale), 10), max(int(h*scale), 10)
    
    scaled = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
    bk = max(3, int(scale * 0.8))
    if bk % 2 == 0: bk += 1
    blurred = cv2.GaussianBlur(scaled, (bk, bk), 0)
    _, sbinary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sbinary = (sbinary > 127).astype(np.uint8)
    
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sbinary = cv2.morphologyEx(sbinary, cv2.MORPH_OPEN, kern)
    sbinary = cv2.morphologyEx(sbinary, cv2.MORPH_CLOSE, kern)
    
    lbl, n = ndlabel(sbinary)
    if n > 1:
        best = max(range(1, n+1), key=lambda j: np.sum(lbl == j))
        sbinary = (lbl == best).astype(np.uint8)
    
    pid = raw.index(rp) + 1
    bmp_path = f'{test_dir}/piece_{pid}.bmp'
    cv2.imwrite(bmp_path, sbinary * 255)
    
    h2, w2 = sbinary.shape
    metadata = {'original_photo_name': '1.png', 'photo_space_origin': (px0, py0),
                'photo_space_centroid': [w2//2, h2//2], 'photo_width': w2, 'photo_height': h2, 'is_complete': True}
    args = [bmp_path, pid, test_dir, metadata, (0,0), 1.0, False]
    try:
        vector.load_and_vectorize(args)
        success += 1
        print(f"  Piece {pid} ({w}x{h} -> {w2}x{h2}): OK")
    except Exception as e:
        print(f"  Piece {pid}: FAILED - {e}")

print(f"\nResult: {success}/10 vectorized successfully")
