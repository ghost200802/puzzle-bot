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

ORI_CHARS = ['^', '>', 'v', '<']
TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3

print("=== Piece 127 at top-right (9,0) ===")
print("  Must connect LEFT to piece 6 (top edge neighbor)")
print("  Must connect BOTTOM to piece 138 (right edge neighbor)")
print("  Must have flat on TOP and RIGHT")
print()

ef_127 = edge_info.get(127, [False]*4)
flat_127 = [i for i, f in enumerate(ef_127) if f]
print(f"  Physical flat sides: {flat_127}")

for ori in range(4):
    rotated_flat = sorted([(s - ori) % 4 for s in flat_127])
    is_tr = set(rotated_flat) == {TOP, RIGHT}
    marker = " <-- TOP-RIGHT" if is_tr else ""

    left_side = (LEFT - ori) % 4
    bottom_side = (BOTTOM - ori) % 4
    left_fits = [m['pid'] for m in conn['127'][left_side]]
    bottom_fits = [m['pid'] for m in conn['127'][bottom_side]]

    connects_6 = 6 in left_fits
    connects_138 = 138 in bottom_fits
    status = ""
    if connects_6 and connects_138:
        status = " <-- CONNECTS TO BOTH 6 AND 138!"
    elif connects_6:
        status = " (connects 6 only)"
    elif connects_138:
        status = " (connects 138 only)"

    print(f"  ori={ori} ('{ORI_CHARS[ori]}'): flat={rotated_flat}, "
          f"left_side[{left_side}] has {len(left_fits)} fits (6:{connects_6}), "
          f"bottom_side[{bottom_side}] has {len(bottom_fits)} fits (138:{connects_138})"
          f"{marker}{status}")

print()
print("=== Piece 70 at bottom-left (0,9) ===")
print("  Must connect TOP to piece 112 (left edge neighbor)")
print("  Must connect RIGHT to piece 20 (bottom edge neighbor)")
print("  Must have flat on BOTTOM and LEFT")
print()

ef_70 = edge_info.get(70, [False]*4)
flat_70 = [i for i, f in enumerate(ef_70) if f]
print(f"  Physical flat sides: {flat_70}")

for ori in range(4):
    rotated_flat = sorted([(s - ori) % 4 for s in flat_70])
    is_bl = set(rotated_flat) == {BOTTOM, LEFT}
    marker = " <-- BOTTOM-LEFT" if is_bl else ""

    top_side = (TOP - ori) % 4
    right_side = (RIGHT - ori) % 4
    top_fits = [m['pid'] for m in conn['70'][top_side]]
    right_fits = [m['pid'] for m in conn['70'][right_side]]

    connects_112 = 112 in top_fits
    connects_20 = 20 in right_fits
    status = ""
    if connects_112 and connects_20:
        status = " <-- CONNECTS TO BOTH 112 AND 20!"
    elif connects_112:
        status = " (connects 112 only)"
    elif connects_20:
        status = " (connects 20 only)"

    print(f"  ori={ori} ('{ORI_CHARS[ori]}'): flat={rotated_flat}, "
          f"top_side[{top_side}] has {len(top_fits)} fits (112:{connects_112}), "
          f"right_side[{right_side}] has {len(right_fits)} fits (20:{connects_20})"
          f"{marker}{status}")

print()
print("=== Also check: which neighbors does 6 connect to on its RIGHT side? ===")
fits_6_right = [m['pid'] for m in conn['6'][(RIGHT - 0) % 4]]
print(f"  6^ right side: {fits_6_right[:10]}")

print()
print("=== Which neighbors does 112 connect to on its BOTTOM side? ===")
fits_112_bottom = [m['pid'] for m in conn['112'][(BOTTOM - 1) % 4]]
print(f"  112> bottom side: {fits_112_bottom[:10]}")

print()
print("=== Summary of all 4 corners ===")
corners = {'TL': (134, (0,0)), 'TR': (127, (9,0)), 'BR': (71, (9,9)), 'BL': (70, (0,9))}
for label, (pid, (cx, cy)) in corners.items():
    ef = edge_info.get(pid, [False]*4)
    flat = [i for i, f in enumerate(ef) if f]
    print(f"  {label}: piece {pid} at ({cx},{cy}), physical flat={flat}")
    for ori in range(4):
        rotated_flat = sorted([(s - ori) % 4 for s in flat])
        print(f"    ori={ori} ('{ORI_CHARS[ori]}'): flat faces {rotated_flat}")
