import sys
sys.path.insert(0, 'src')
from common.vector import Vector, load_and_vectorize
import os, json

OUTPUT_DIR = 'output/test_945'
bmp_path = os.path.join(OUTPUT_DIR, 'piece_1.bmp')

metadata = {
    'original_photo_name': '1.png',
    'photo_space_origin': [127, 20],
    'photo_space_centroid': [479, 546],
    'photo_width': 958,
    'photo_height': 958,
    'is_complete': True,
}

args = [bmp_path, 1, OUTPUT_DIR, metadata, (0, 0), 1.0, False]
load_and_vectorize(args)

for side_idx in range(4):
    json_path = os.path.join(OUTPUT_DIR, f'side_1_{side_idx}.json')
    with open(json_path) as f:
        data = json.load(f)
    verts = data['vertices']
    first = verts[0]
    last = verts[-1]
    is_edge = data['is_edge']
    print(f"Side {side_idx}: {len(verts)} vertices, ({first[0]:.0f},{first[1]:.0f})->({last[0]:.0f},{last[1]:.0f}), is_edge={is_edge}")
