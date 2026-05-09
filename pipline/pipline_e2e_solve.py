#!/usr/bin/env python3
import os
import sys
import time
import argparse
import subprocess

_here = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_here, '..'))

SOLUTION_DIR_NAME = '6_solution'


def build_cmd(script_path, step, input_dir, output_dir, target_image):
    cmd = [sys.executable, script_path]
    if step == 1:
        cmd += ['-i', input_dir, '-o', output_dir]
    elif step in (2, 3, 4, 5):
        cmd += ['-o', output_dir]
    elif step == 6:
        cmd += ['-o', output_dir]
        if target_image:
            cmd += ['--target', target_image]
    elif step == 7:
        solution_dir = os.path.join(output_dir, SOLUTION_DIR_NAME)
        cmd += ['--solution', solution_dir, '-o', output_dir]
        if target_image:
            cmd += ['--target', target_image]
    return cmd


STEPS = [
    (1, 'Step 1: Split Pieces',     'pipline/run_splitpieces.py'),
    (2, 'Step 2: Vectorize',        'pipline/run_vectorize.py'),
    (3, 'Step 3: Deduplicate',      'pipline/run_dedup.py'),
    (4, 'Step 4: Connectivity',     'pipline/run_connect.py'),
    (5, 'Step 5: Solve',            'pipline/run_solve.py'),
    (6, 'Step 6: Match Target',     'pipline/run_matchtarget.py'),
    (7, 'Step 7: Targeted Solve',   'pipline/run_targetedsolve.py'),
]


def main():
    parser = argparse.ArgumentParser(description='End-to-end puzzle solving pipeline')
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing piece images (PNG/JPG)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output root directory')
    parser.add_argument('-t', '--target', default=None,
                        help='Target image for match target / targeted solve (steps 6-7)')
    parser.add_argument('-s', '--start-step', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7],
                        help='Start from this step (default: 1)')
    parser.add_argument('-e', '--end-step', type=int, default=7, choices=[1, 2, 3, 4, 5, 6, 7],
                        help='End at this step (default: 7)')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    active_steps = [(n, label, script) for n, label, script in STEPS
                    if args.start_step <= n <= args.end_step]

    print("=" * 60)
    print("End-to-End Puzzle Pipeline")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Steps:  {active_steps[0][0]}-{active_steps[-1][0]}")
    if args.target:
        print(f"  Target: {args.target}")
    print("=" * 60)

    t_total = time.time()

    for step_num, label, script in active_steps:
        script_path = os.path.join(ROOT_DIR, script)
        if not os.path.isfile(script_path):
            print(f"\n  [ERROR] Script not found: {script_path}")
            sys.exit(1)

        cmd = build_cmd(script_path, step_num, input_dir, output_dir, args.target)

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
