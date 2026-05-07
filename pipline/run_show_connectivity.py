import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR
from show_connectivity import show as show_connectivity

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)
CHECK_PATH = os.path.join(OUTPUT_DIR, 'check', 'connectivity')

print("Regenerating visualization (no flip)...")
show_connectivity(CONNECTIVITY_PATH, DEDUPED_PATH, CHECK_PATH)
print("Done!")
