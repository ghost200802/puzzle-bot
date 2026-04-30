#!/usr/bin/env python3
"""Analyze the actual grid layout of pieces."""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Read deduped data to get piece positions
deduped_dir = 'output/puzzle_run/4_deduped'
from common import pieces

ps = pieces.Piece.load_all(deduped_dir, resample=True)
print(f"Loaded {len(ps)} pieces")

# Count edges vs interior
edge_count = 0
corner_count = 0
for pid, piece in ps.items():
    edge_sides = sum(1 for s in piece.sides if s.is_edge)
    if edge_sides >= 2:
        corner_count += 1
    elif edge_sides == 1:
        edge_count += 1

interior = len(ps) - corner_count - edge_count
print(f"\nPiece classification:")
print(f"  Corners (2+ edges): {corner_count}")
print(f"  Edges (1 edge): {edge_count}")
print(f"  Interior (0 edges): {interior}")

# For a rectangular puzzle with W x H grid:
# corners = 4, edges = 2*(W+H)-4, interior = (W-2)*(H-2)
# Total = W * H
# From our data: total = 95, corners = 4 (if correct)
# Need to find W, H such that W*H = 95
# But 95 = 5 * 19 ... not a nice grid
# Let's check: maybe some pieces are missing or it's not rectangular

print(f"\nTotal pieces: {len(ps)}")
print(f"Factors of 95: 1x95, 5x19")

# Read connectivity to see edge assignments
conn_file = 'output/puzzle_run/5_connectivity/connectivity.json'
with open(conn_file) as f:
    conn = json.load(f)

print(f"\nConnectivity entries: {len(conn)}")
for pid_str, fits in list(conn.items())[:5]:
    edge_sides = [i for i, side in enumerate(fits) if len(side) == 0]
    print(f"  Piece {pid_str}: edge_sides={edge_sides}")

# Try to figure out the actual grid from piece positions
# Read origins from side files
import glob
positions = []
for sf in glob.glob(os.path.join(deduped_dir, 'side_*_0.json')):
    with open(sf) as f:
        data = json.load(f)
    origin = data.get('photo_space_origin', [0, 0])
    pid = data.get('piece_id', 0)
    positions.append((pid, origin[0], origin[1]))

positions.sort(key=lambda p: (p[2], p[1]))
print(f"\nPiece positions (sorted by y, x):")
for pid, x, y in positions:
    print(f"  Piece {pid}: ({x}, {y})")
