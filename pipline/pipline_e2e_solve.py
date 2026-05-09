#!/usr/bin/env python3
import os
import sys
import time
import argparse
import subprocess

_here = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_here, '..'))

SCRIPTS = [
    ('Step 1: Split Pieces',    'pipline/run_splitpieces.py', ['-i', '-o']),
    ('Step 2: Vectorize',       'pipline/run_vectorize.py',   ['-o']),
    ('Step 3: Deduplicate',     'pipline/run_dedup.py',       ['-o']),
    ('Step 4: Connectivity',    'pipline/run_connect.py',     ['-o']),
    ('Step 5: Solve',           'pipline/run_solve.py',       ['-o']),
]


def main():
    parser = argparse.ArgumentParser(description='End-to-end puzzle solving pipeline')
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing piece images (PNG/JPG)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output root directory')
    parser.add_argument('-s', '--start-step', type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help='Start from this step (default: 1)')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("End-to-End Puzzle Pipeline")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Steps:  {args.start_step}-{len(SCRIPTS)}")
    print("=" * 60)

    t_total = time.time()

    for i, (label, script, flags) in enumerate(SCRIPTS, start=1):
        if i < args.start_step:
            print(f"\n  [SKIP] {label} (step {i})")
            continue

        script_path = os.path.join(ROOT_DIR, script)
        if not os.path.isfile(script_path):
            print(f"\n  [ERROR] Script not found: {script_path}")
            sys.exit(1)

        cmd = [sys.executable, script_path]
        if '-i' in flags:
            cmd += ['-i', input_dir]
        cmd += ['-o', output_dir]

        print(f"\n{'#' * 60}")
        print(f"# {label}")
        print(f"{'#' * 60}")

        t0 = time.time()
        result = subprocess.run(cmd, cwd=ROOT_DIR)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\n  [FAILED] {label} (exit code {result.returncode}, {elapsed:.1f}s)")
            sys.exit(result.returncode)

        print(f"\n  [OK] {label} ({elapsed:.1f}s)")

    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete! Total: {total_elapsed:.1f}s")
    print(f"Output: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
