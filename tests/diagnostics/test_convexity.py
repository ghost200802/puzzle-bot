import os
import sys
import math

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..'))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))

from common import pieces, sides

from pipline.config import get_output_dir, get_deduped_path

OUTPUT_DIR = get_output_dir()
DEDUPED_PATH = get_deduped_path()

ps_raw = pieces.Piece.load_all(DEDUPED_PATH, resample=False)

test_pairs = [
    (125, 3, 78, 1),
    (1, 1, 117, 2),
    (1, 0, 70, 0),
]

for pid_a, si_a, pid_b, si_b in test_pairs:
    sa = ps_raw[pid_a].sides[si_a]
    sb = ps_raw[pid_b].sides[si_b]
    print(f"P{pid_a}[{si_a}] is_convex={sa.is_convex}  vs  P{pid_b}[{si_b}] is_convex={sb.is_convex}")
    if sa.is_convex is not None and sb.is_convex is not None:
        if sa.is_convex == sb.is_convex:
            print(f"  -> REJECTED: same convexity (both {'convex' if sa.is_convex else 'concave'})")
        else:
            print(f"  -> PASS: one convex, one concave")
    else:
        print(f"  -> SKIP: convexity unknown")

print(f"\n--- Convexity distribution across all sides ---")
convex_count = 0
concave_count = 0
none_count = 0
for pid, piece in ps_raw.items():
    for i in range(4):
        s = piece.sides[i]
        if s.is_convex is None:
            none_count += 1
        elif s.is_convex:
            convex_count += 1
        else:
            concave_count += 1
print(f"  Convex: {convex_count}  Concave: {concave_count}  Unknown/Edge: {none_count}")
