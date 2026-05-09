import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from show_connectivity import show as show_connectivity
from config import get_output_dir, get_deduped_path, get_connectivity_path

OUTPUT_DIR = get_output_dir()
DEDUPED_PATH = get_deduped_path()
CONNECTIVITY_PATH = get_connectivity_path()
CHECK_PATH = os.path.join(get_output_dir(), 'check', 'connectivity')

print("Regenerating visualization (no flip)...")
show_connectivity(CONNECTIVITY_PATH, DEDUPED_PATH, CHECK_PATH)
print("Done!")
