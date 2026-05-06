#!/usr/bin/env python3
"""
Entrypoint from the command line to find a puzzle solution from a batch of input photos

Supports two modes:
  - phone mode (default): --photos-dir, --width, --height
  - robot mode: --path (legacy)
"""

import cProfile
import argparse
import posixpath
import os
import time
import json

import process, solve
from common.config import (
    MODE,
    PHOTOS_DIR, PHOTO_BMP_DIR, SEGMENT_DIR, DEDUPED_DIR,
    VECTOR_DIR, CONNECTIVITY_DIR, SOLUTION_DIR, TIGHTNESS_DIR,
)
from common import util


def _prepare_new_run(path, start_at_step, stop_before_step):
    # set up all the directories we'll need
    dirs = [
        PHOTOS_DIR, PHOTO_BMP_DIR, SEGMENT_DIR, DEDUPED_DIR,
        VECTOR_DIR, CONNECTIVITY_DIR, SOLUTION_DIR, TIGHTNESS_DIR,
    ]
    for i, d in enumerate(dirs):
        os.makedirs(os.path.join(path, d), exist_ok=True)
        # remove any files left over from a previous run
        if os.path.exists(os.path.join(path, d, '.DS_Store')):
            os.remove(os.path.join(path, d, '.DS_Store'))
        if i != 0 and i > start_at_step and i <= stop_before_step:
            for f in os.listdir(os.path.join(path, d)):
                os.remove(os.path.join(path, d, f))


def main():
    parser = argparse.ArgumentParser(
        description='Puzzle solver: process photos and find solution'
    )

    if MODE == 'phone':
        # Phone mode arguments
        parser.add_argument('--photos-dir', required=True,
                            help='Directory containing phone photos (JPG/PNG)',
                            type=str)
        parser.add_argument('--target', required=False, default=None,
                            help='Path to target image (box photo)',
                            type=str)
        parser.add_argument('--width', required=False, default=None, type=int,
                            help='Puzzle width (number of columns). Auto-detected if omitted.')
        parser.add_argument('--height', required=False, default=None, type=int,
                            help='Puzzle height (number of rows). Auto-detected if omitted.')
        parser.add_argument('--output-dir', default='output', type=str,
                            help='Output directory')
        parser.add_argument('--segmentation', default='adaptive',
                            choices=['adaptive', 'otsu', 'grabcut'],
                            help='Segmentation method')
        parser.add_argument('--serialize', default=False,
                            action='store_true',
                            help='Single-thread processing')
        parser.add_argument('--start-at-step', default=0, type=int,
                            help='Start processing at this step')
        parser.add_argument('--stop-before-step', default=10, type=int,
                            help='Stop processing before this step')
    else:
        # Robot mode arguments (legacy)
        parser.add_argument('--path', required=True, type=str,
                            help='Path to the base directory')
        parser.add_argument('--only-process-id', default=None, type=str,
                            help='Only process the photo with this ID')
        parser.add_argument('--start-at-step', default=0, type=int,
                            help='Start processing at this step')
        parser.add_argument('--stop-before-step', default=10, type=int,
                            help='Stop before this step')
        parser.add_argument('--serialize', default=False,
                            action='store_true',
                            help='Single-thread processing')

    args = parser.parse_args()
    start_time = time.time()

    if MODE == 'phone':
        _run_phone_mode(args)
    else:
        _run_robot_mode(args)

    duration = time.time() - start_time
    print(f"\n\n### Ran in {round(duration, 2)} sec ###\n")


def _run_phone_mode(args):
    """Run the phone mode pipeline."""
    output_dir = args.output_dir
    _prepare_new_run(output_dir, args.start_at_step, args.stop_before_step)

    # Copy photos to the standard photos directory
    photos_src = args.photos_dir
    photos_dst = os.path.join(output_dir, PHOTOS_DIR)
    os.makedirs(photos_dst, exist_ok=True)

    for f in os.listdir(photos_src):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(photos_src, f)
            dst = os.path.join(photos_dst, f)
            with open(src, 'rb') as sf:
                data = sf.read()
            with open(dst, 'wb') as df:
                df.write(data)

    # Process photos
    process.batch_process_photos(
        path=output_dir,
        serialize=args.serialize,
        start_at_step=args.start_at_step,
        stop_before_step=args.stop_before_step,
        puzzle_width=args.width,
        puzzle_height=args.height,
        segmentation_method=args.segmentation,
    )

    # Solve
    if args.stop_before_step >= 5:
        solve.solve(
            path=output_dir,
            start_at=args.start_at_step,
            puzzle_width=args.width,
            puzzle_height=args.height,
        )


def _run_robot_mode(args):
    """Run the robot mode pipeline (legacy)."""
    _prepare_new_run(args.path, args.start_at_step, args.stop_before_step)

    # Load batch info
    batch_info_file = posixpath.join(args.path, PHOTOS_DIR, "batch.json")
    with open(batch_info_file, "r") as jsonfile:
        batch_info = json.load(jsonfile)["photos"]
    robot_states = {}
    for d in batch_info:
        robot_states[d["file_name"]] = d["position"]

    # Process
    process.batch_process_photos(
        path=args.path,
        serialize=args.serialize,
        robot_states=robot_states,
        id=args.only_process_id,
        start_at_step=args.start_at_step,
        stop_before_step=args.stop_before_step,
    )

    # Solve
    if args.stop_before_step >= 3 and args.only_process_id is None:
        solve.solve(path=args.path, start_at=args.start_at_step)


if __name__ == '__main__':
    PROFILE = False
    if PROFILE:
        cProfile.run('main()', 'profile_results.prof')
    else:
        main()
