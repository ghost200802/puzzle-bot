import os
import sys
import json
import math

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, '5_connectivity')

with open(os.path.join(CONNECTIVITY_PATH, 'connectivity.json')) as f:
    conn = json.load(f)

all_errors = []
best_errors_per_side = []

for pid_str, fits_list in conn.items():
    for si, matches in enumerate(fits_list):
        for m in matches:
            other_pid, other_si, error_x1000 = m
            error = error_x1000 / 1000.0
            all_errors.append(error)
        if matches:
            best_err = min(m[2] for m in matches) / 1000.0
            best_errors_per_side.append(best_err)

all_errors.sort()
best_errors_per_side.sort()

print(f"Total match entries: {len(all_errors)}")
print(f"\n--- All errors distribution ---")
for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
    count = sum(1 for e in all_errors if e <= threshold)
    pct = 100.0 * count / len(all_errors)
    print(f"  <= {threshold:.1f}: {count:6d} ({pct:.1f}%)")

print(f"\n--- Best error per side distribution ---")
print(f"  Total sides with matches: {len(best_errors_per_side)}")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
    count = sum(1 for e in best_errors_per_side if e <= threshold)
    pct = 100.0 * count / len(best_errors_per_side)
    print(f"  <= {threshold:.1f}: {count:4d} ({pct:.1f}%)")

print(f"\n--- Best errors > 2.0 (suspicious) ---")
suspicious = [(i, e) for i, e in enumerate(best_errors_per_side) if e > 2.0]
print(f"  Count: {len(suspicious)}")

print(f"\n--- Percentiles of best errors ---")
for p in [5, 10, 25, 50, 75, 90, 95]:
    idx = int(len(best_errors_per_side) * p / 100)
    print(f"  P{p}: {best_errors_per_side[min(idx, len(best_errors_per_side)-1)]:.4f}")

print(f"\n--- Error histogram (all matches) ---")
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    count = sum(1 for e in all_errors if lo <= e < hi)
    bar = '#' * (count // 50)
    print(f"  [{lo:.1f}, {hi:.1f}): {count:5d} {bar}")
