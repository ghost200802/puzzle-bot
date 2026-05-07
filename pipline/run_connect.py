import os
import sys
import json
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR
from common import connect
from show_connectivity import show as show_connectivity

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)
CHECK_PATH = os.path.join(OUTPUT_DIR, 'check', 'connectivity')


def build_piece_edge_info(deduped_path):
    piece_edge_info = {}
    vp = Path(deduped_path)
    for path in sorted(vp.glob("side_*_0.json")):
        pid = int(path.parts[-1].split('_')[1])
        edge_flags = []
        for j in range(4):
            jpath = vp / f'side_{pid}_{j}.json'
            with open(jpath) as f:
                data = json.load(f)
            edge_flags.append(data.get('is_edge', False))
        piece_edge_info[pid] = edge_flags
    return piece_edge_info


def main():
    print("=" * 60)
    print("Connectivity Building Pipeline (v2)")
    print(f"Input:  {DEDUPED_PATH}")
    print(f"Output: {CONNECTIVITY_PATH}")
    print("=" * 60)

    if not os.path.exists(DEDUPED_PATH):
        print(f"Error: deduped directory not found: {DEDUPED_PATH}")
        return

    os.makedirs(CONNECTIVITY_PATH, exist_ok=True)

    piece_edge_info = build_piece_edge_info(DEDUPED_PATH)

    corners = [pid for pid, flags in piece_edge_info.items()
               if sum(1 for f in flags if f) >= 2]
    edges = [pid for pid, flags in piece_edge_info.items()
             if sum(1 for f in flags if f) >= 1]
    inner = [pid for pid, flags in piece_edge_info.items()
             if sum(1 for f in flags if f) == 0]

    print(f"\nPiece classification:")
    print(f"  Total pieces: {len(piece_edge_info)}")
    print(f"  Corners (>=2 flat edges): {len(corners)} -> {corners}")
    print(f"  Edge pieces (>=1 flat edge): {len(edges)}")
    print(f"  Inner pieces (0 flat edges): {len(inner)}")

    edge_info_path = os.path.join(CONNECTIVITY_PATH, 'piece_edge_info.json')
    with open(edge_info_path, 'w') as f:
        json.dump({str(k): v for k, v in piece_edge_info.items()}, f)
    print(f"\nSaved piece_edge_info.json ({len(piece_edge_info)} pieces)")

    print("\nBuilding connectivity graph...")
    connectivity = connect.build(DEDUPED_PATH, CONNECTIVITY_PATH)
    print(f"Connectivity graph saved to {CONNECTIVITY_PATH}/connectivity.json")

    summary = {}
    for pid_str, fits_list in connectivity.items():
        pid = int(pid_str)
        total_matches = sum(len(f) for f in fits_list)
        sides_with_matches = sum(1 for f in fits_list if len(f) > 0)
        best_errors = []
        for f in fits_list:
            if f:
                best_errors.append(min(m['error'] for m in f))
        summary[pid] = {
            'total_matches': total_matches,
            'sides_with_matches': sides_with_matches,
            'best_error': min(best_errors) if best_errors else None,
            'is_corner': pid in corners,
            'is_edge': pid in edges and pid not in corners,
            'is_inner': pid in inner,
        }

    summary_path = os.path.join(CONNECTIVITY_PATH, 'connectivity_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved connectivity_summary.json")

    total_matches = sum(s['total_matches'] for s in summary.values())
    pieces_with_matches = sum(1 for s in summary.values() if s['total_matches'] > 0)
    print(f"\n{'=' * 60}")
    print(f"Connectivity building complete!")
    print(f"  Pieces processed: {len(connectivity)}")
    print(f"  Pieces with matches: {pieces_with_matches}")
    print(f"  Total match entries: {total_matches}")
    print(f"  Output: {CONNECTIVITY_PATH}/")
    print(f"{'=' * 60}")

    print("\nGenerating connectivity visualization...")
    show_connectivity(CONNECTIVITY_PATH, DEDUPED_PATH, CHECK_PATH)


if __name__ == '__main__':
    main()
