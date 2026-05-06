#!/usr/bin/env python3
"""
Realtime phase entry point for puzzle solving.

Usage:
  # Identify pieces from a single photo
  python src/run_realtime.py --database ./my_puzzle/output/database --photo ./current_piece.jpg

  # Interactive camera mode
  python src/run_realtime.py --database ./my_puzzle/output/database --camera

  # Specify output directory for updated database
  python src/run_realtime.py --database ./my_puzzle/output/database --photo ./piece.jpg --output-dir ./my_puzzle/output/database
"""

import argparse
import os
import sys
import time

from common.database import PieceDatabase
from common.real_time import RealTimeIdentifier, _describe_rotation


def main():
    parser = argparse.ArgumentParser(
        description='Puzzle piece realtime identification'
    )

    parser.add_argument('--database', required=True, type=str,
                        help='Path to the puzzle database directory')
    parser.add_argument('--photo', default=None, type=str,
                        help='Path to a photo of the piece to identify')
    parser.add_argument('--camera', default=False, action='store_true',
                        help='Use camera for interactive identification')
    parser.add_argument('--camera-id', default=0, type=int,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--output-dir', default=None, type=str,
                        help='Output directory for updated database (default: same as --database)')
    parser.add_argument('--save', default=False, action='store_true',
                        help='Save the updated database after identification')

    args = parser.parse_args()

    if not args.photo and not args.camera:
        parser.error("Either --photo or --camera must be specified")

    print(f"Loading database from {args.database}...")
    db = PieceDatabase.load(args.database)
    print(f"  Pieces: {len(db.pieces)}")
    missing, is_complete = db.check_completeness()
    if missing > 0:
        print(f"  Missing: {missing} pieces")
    if db.solution is not None:
        print(f"  Solution: {db.solution.placed_count}/{db.width * db.height} pieces placed")
    else:
        print(f"  Solution: not yet solved")

    identifier = RealTimeIdentifier(db, camera_id=args.camera_id)

    start_time = time.time()

    if args.photo:
        print(f"\nIdentifying pieces from: {args.photo}")
        result = identifier.identify(args.photo)

        print("\n" + "=" * 60)
        if not result.results:
            print("No pieces detected in the photo.")
        else:
            for r in result.results:
                print(f"  {r}")
                if r.status in ('known', 'new_solved') and r.solution_position:
                    pos = r.solution_position
                    print(f"    -> Place at row {pos['y']+1}, col {pos['x']+1}")
                    print(f"    -> {_describe_rotation(pos['rotation'])}")
                elif r.status == 'new_unsolved':
                    print(f"    -> Position unknown, added to database")
        print("=" * 60)

    elif args.camera:
        identifier.run_interactive(display=True)

    duration = time.time() - start_time
    print(f"\nIdentification completed in {round(duration, 2)} sec")

    if args.save:
        output_dir = args.output_dir or args.database
        print(f"\nSaving updated database to {output_dir}...")
        db.save(output_dir)


if __name__ == '__main__':
    main()
