import os
import json
import numpy as np

from config import get_connectivity_path

with open(os.path.join(get_connectivity_path(), 'connectivity.json')) as f:
    data = json.load(f)

counts = []
for pid_str in sorted(data.keys(), key=int):
    sides = data[pid_str]
    for si, matches in enumerate(sides):
        if matches:
            counts.append(len(matches))

arr = np.array(counts)
print(f"Total piece-sides with matches: {len(arr)}")
print(f"Candidates per side: mean={np.mean(arr):.1f} P50={np.median(arr):.0f} P90={np.percentile(arr,90):.0f} max={np.max(arr)} min={np.min(arr)}")
for n in [1, 2, 3, 5, 10, 20, 30, 50]:
    cnt = sum(1 for x in arr if x <= n)
    print(f"  sides with <= {n:2d} candidates: {cnt}/{len(arr)} ({cnt/len(arr)*100:.1f}%)")

print(f"\nTotal match pairs: {sum(arr)}")
print(f"\nSample (first 5 pieces):")
for pid_str in sorted(data.keys(), key=int)[:5]:
    sides = data[pid_str]
    for si, matches in enumerate(sides):
        if matches:
            print(f"  Piece {pid_str}, side {si}: {len(matches)} candidates")
