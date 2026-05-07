import os
import sys
import json

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, '5_connectivity')

with open(os.path.join(CONNECTIVITY_PATH, 'connectivity.json')) as f:
    conn = json.load(f)
with open(os.path.join(CONNECTIVITY_PATH, 'piece_edge_info.json')) as f:
    edge_info = json.load(f)

print("=== Sides with best error > 1.5 (suspicious) ===\n")
for pid_str, fits_list in sorted(conn.items(), key=lambda x: int(x[0])):
    pid = int(pid_str)
    ef = edge_info[pid_str]
    flat_count = sum(1 for f in ef if f)
    ptype = "CORNER" if flat_count >= 2 else ("EDGE" if flat_count >= 1 else "INNER")

    for si, matches in enumerate(fits_list):
        if not matches:
            continue
        best_err = min(m[2] for m in matches) / 1000.0
        if best_err > 1.5:
            m = matches[0]
            print(f"  P{pid}[{si}] ({ptype}) -> P{m[0]}[{m[1]}]  error={best_err:.3f}")

print("\n=== Sides with best error > 0.5 (check quality) ===\n")
count = 0
for pid_str, fits_list in sorted(conn.items(), key=lambda x: int(x[0])):
    pid = int(pid_str)
    ef = edge_info[pid_str]
    flat_count = sum(1 for f in ef if f)
    ptype = "CORNER" if flat_count >= 2 else ("EDGE" if flat_count >= 1 else "INNER")

    for si, matches in enumerate(fits_list):
        if not matches:
            continue
        best_err = min(m[2] for m in matches) / 1000.0
        if best_err > 0.5:
            m = matches[0]
            print(f"  P{pid}[{si}] ({ptype}) -> P{m[0]}[{m[1]}]  error={best_err:.3f}  (top5: {', '.join(f'{mm[2]/1000:.3f}' for mm in matches[:5])})")
            count += 1

print(f"\nTotal: {count} sides with best error > 0.5")
