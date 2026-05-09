import os
import sys
import json

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from config import get_output_dir, get_connectivity_path

OUTPUT_DIR = get_output_dir()
CONNECTIVITY_PATH = get_connectivity_path()

connectivity_file = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
edge_info_file = os.path.join(CONNECTIVITY_PATH, 'piece_edge_info.json')

with open(connectivity_file, 'r') as f:
    conn = json.load(f)
with open(edge_info_file, 'r') as f:
    edge_info = {int(k): v for k, v in json.load(f).items()}

for pid in [70, 127]:
    ef = edge_info.get(pid, [False]*4)
    flat_sides = [i for i, f in enumerate(ef) if f]
    print(f"Piece {pid}: edge_flags={ef}, flat_sides={flat_sides}")
    for si in range(4):
        fits = conn[str(pid)][si]
        n_fits = len(fits)
        pids = [m['pid'] for m in fits[:5]]
        print(f"  side[{si}]: {n_fits} fits, first={pids}")

print("\n--- Orientation analysis ---")
print("TOP=0, RIGHT=1, BOTTOM=2, LEFT=3")
print("At top-right (9,0): need flat on TOP(0) and RIGHT(1)")
print("At bottom-left (0,9): need flat on BOTTOM(2) and LEFT(3)")
print()

for pid in [70, 127]:
    ef = edge_info.get(pid, [False]*4)
    flat_sides = [i for i, f in enumerate(ef) if f]
    print(f"Piece {pid}: physical flat sides = {flat_sides}")
    for ori in range(4):
        rotated_flat = [(s - ori) % 4 for s in flat_sides]
        rotated_flat.sort()
        print(f"  ori={ori} ('{'^>v<'[ori]}'): flat faces directions {rotated_flat}")
        if set(rotated_flat) == {0, 1}:
            print(f"    -> GOOD for top-right corner!")
        if set(rotated_flat) == {2, 3}:
            print(f"    -> GOOD for bottom-left corner!")
