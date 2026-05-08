import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

import importlib
import common.texture_verify as tv
importlib.reload(tv)
from common.texture_verify import verify_match

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '3_vector')
COLOR_PATH = os.path.join(OUTPUT_DIR, '2_piece_colors')

result = verify_match(COLOR_PATH, DEDUPED_PATH, 4, 3, 137, 3)
print("verify_match result for 4[3] <-> 137[3]:")
for k, v in result.items():
    print(f"  {k}: {v}")
