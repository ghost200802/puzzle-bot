import os, sys, math
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))
from common import pieces

DEDUPED_DIR = os.path.join(_here, '..', 'output', 'puzzle_new', '4_deduped')
ps = pieces.Piece.load_all(DEDUPED_DIR, resample=False)

for pid in [137, 141]:
    p = ps[pid]
    print(f"Piece {pid}:")
    for si in range(4):
        s = p.sides[si]
        print(f"  Side {si}: is_convex={s.is_convex}, is_edge={s.is_edge}, len={s.original_length:.1f}")
    print()
