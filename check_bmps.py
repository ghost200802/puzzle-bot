import os, numpy as np
from PIL import Image

bmp_dir = r'f:\work_Puzzle_github\puzzle-bot\output\puzzle_new\3_vector'
files = sorted([f for f in os.listdir(bmp_dir) if f.endswith('.bmp')])
print(f'Total BMPs: {len(files)}')

for f in files[:5]:
    path = os.path.join(bmp_dir, f)
    img = Image.open(path)
    arr = np.array(img)
    print(f'{f}: mode={img.mode}, shape={arr.shape}, dtype={arr.dtype}, unique={np.unique(arr)}, sum_true={np.sum(arr)}, sum_255={np.sum(arr==255)}')

# Also check piece 6 (too large)
p6 = os.path.join(bmp_dir, 'piece_6.bmp')
if os.path.exists(p6):
    img6 = Image.open(p6)
    arr6 = np.array(img6)
    print(f'piece_6.bmp: mode={img6.mode}, shape={arr6.shape}, dtype={arr6.dtype}, unique={np.unique(arr6)}')

# Check how vectorizer loads BMPs
print('\n--- vectorizer load test ---')
sys_path = r'f:\work_Puzzle_github\puzzle-bot\src'
import sys
sys.path.insert(0, sys_path)
from common.find_islands import load_binary_bitmap
for f in files[:3]:
    path = os.path.join(bmp_dir, f)
    grid = load_binary_bitmap(path)
    print(f'{f}: loaded shape={grid.shape}, dtype={grid.dtype}, unique={np.unique(grid)}, sum={np.sum(grid)}')
