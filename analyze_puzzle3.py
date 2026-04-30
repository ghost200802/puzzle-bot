"""Deep analysis of puzzle image - understand the actual content."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import label as ndlabel

img = Image.open('input/puzzles/1.png')
rgba = np.array(img)
print(f"Image: {rgba.shape}")

# Check if this is a jigsaw puzzle or something else
alpha = rgba[:,:,3]
rgb = rgba[:,:,:3]

# Sample some rows to understand the layout
print("\nRow analysis (every 50 rows):")
for y in range(0, rgba.shape[0], 50):
    row_alpha = alpha[y, :]
    transitions = 0
    in_fg = False
    for x in range(len(row_alpha)):
        fg = row_alpha[x] > 128
        if fg != in_fg:
            transitions += 1
            in_fg = fg
    fg_pixels = np.sum(row_alpha > 128)
    if fg_pixels > 0:
        print(f"  Row {y}: {fg_pixels} fg pixels, {transitions} alpha transitions")

# Check the color distribution at boundaries between regions
# Find vertical boundary lines by looking at columns with many alpha transitions
print("\nColumn alpha transition analysis:")
col_transitions = []
for x in range(rgba.shape[1]):
    col = alpha[:, x]
    trans = 0
    in_fg = False
    for y in range(len(col)):
        fg = col[y] > 128
        if fg != in_fg:
            trans += 1
            in_fg = fg
    col_transitions.append((x, trans))

# Find columns with high transitions (boundaries between pieces)
high_trans = [(x, t) for x, t in col_transitions if t > 4]
print(f"Columns with >4 alpha transitions: {len(high_trans)}")
for x, t in sorted(high_trans, key=lambda v: -v[1])[:20]:
    print(f"  Column {x}: {t} transitions")

# Let's try a completely different approach:
# Look at the RGB variance along horizontal/vertical lines
# Puzzle piece boundaries should have color discontinuities
print("\nColor discontinuity analysis:")
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)

# Horizontal gradient (column-wise discontinuities)
h_grad = np.abs(np.diff(gray, axis=1))
# Vertical gradient (row-wise discontinuities)  
v_grad = np.abs(np.diff(gray, axis=0))

# Find columns with high average gradient (vertical boundaries)
h_col_avg = np.mean(h_grad, axis=0)
v_row_avg = np.mean(v_grad, axis=0)

# Threshold
h_peaks = np.where(h_col_avg > np.percentile(h_col_avg, 98))[0]
v_peaks = np.where(v_row_avg > np.percentile(v_row_avg, 98))[0]

# Cluster nearby peaks
def cluster(peaks, gap=20):
    if len(peaks) == 0:
        return []
    clusters = [[peaks[0]]]
    for p in peaks[1:]:
        if p - clusters[-1][-1] < gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(np.mean(c)) for c in clusters]

h_boundaries = cluster(h_peaks)
v_boundaries = cluster(v_peaks)

print(f"Vertical boundaries (columns): {h_boundaries}")
print(f"Horizontal boundaries (rows): {v_boundaries}")
print(f"Grid estimate: {len(h_boundaries)+1} cols x {len(v_boundaries)+1} rows = {(len(h_boundaries)+1)*(len(v_boundaries)+1)} pieces")
