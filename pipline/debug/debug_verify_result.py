import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import importlib
import common.texture_verify as tv
importlib.reload(tv)
from common.texture_verify import verify_match

from pipline.config import get_output_dir, get_vector_path, get_color_path

OUTPUT_DIR = get_output_dir()
DEDUPED_PATH = get_vector_path()
COLOR_PATH = get_color_path()

result = verify_match(COLOR_PATH, DEDUPED_PATH, 4, 3, 137, 3)
print("verify_match result for 4[3] <-> 137[3]:")
for k, v in result.items():
    print(f"  {k}: {v}")
