import os
import sys
import json
import time
import multiprocessing

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.texture_verify import verify_match
from config import get_output_dir, get_deduped_path, get_connectivity_path, get_color_path, get_vector_path

OUTPUT_DIR = get_output_dir()
DEDUPED_PATH = get_deduped_path()
CONNECTIVITY_PATH = get_connectivity_path()
COLOR_PATH = get_color_path()
if not os.path.isdir(COLOR_PATH):
    COLOR_PATH = get_vector_path()

ERROR_RATIO_THRESHOLD = 2.0
NCC_FILTER_RATIO = 0.5


def _verify_single(args):
    pid_a, si, pid_b, sj = args
    result = verify_match(COLOR_PATH, DEDUPED_PATH, pid_a, si, pid_b, sj)
    return (pid_a, si, pid_b, sj, result)


def _compute_score(ncc, color_diff):
    ncc_pos = max(ncc, 0.001)
    return ncc_pos / (1.0 + color_diff / 50.0)


def main():
    conn_path = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
    if not os.path.exists(conn_path):
        print(f"Error: connectivity.json not found: {conn_path}")
        return

    with open(conn_path) as f:
        connectivity = json.load(f)

    print("=" * 60)
    print("Texture Verification Pipeline")
    print(f"  Connectivity: {conn_path}")
    print(f"  Color images: {COLOR_PATH}")
    print(f"  Side data:    {DEDUPED_PATH}")
    print(f"  Pieces: {len(connectivity)}")
    print(f"  Error ratio threshold: {ERROR_RATIO_THRESHOLD}x")
    print(f"  NCC filter ratio: <{NCC_FILTER_RATIO} of best -> reject")
    print("=" * 60)

    # ================================================================
    # Phase 0: Pre-filter by error ratio
    # ================================================================
    print(f"\n--- Phase 0: Pre-filter by error (ratio > {ERROR_RATIO_THRESHOLD}x) ---")

    original_total = 0
    filtered_total = 0
    filtered_connectivity = {}

    for pid_str, fits_list in connectivity.items():
        filtered_fits = []
        for si, matches in enumerate(fits_list):
            original_total += len(matches)
            if not matches:
                filtered_fits.append([])
                continue

            best_error = matches[0]['error']
            kept = []
            for m in matches:
                if m['error'] <= best_error * ERROR_RATIO_THRESHOLD:
                    kept.append(m)
            filtered_fits.append(kept)
            filtered_total += len(kept)

        filtered_connectivity[pid_str] = filtered_fits

    removed_by_error = original_total - filtered_total
    print(f"  Original candidates:  {original_total}")
    print(f"  After error filter:   {filtered_total}")
    print(f"  Removed by error:     {removed_by_error} ({removed_by_error / max(original_total, 1) * 100:.1f}%)")

    filtered_conn_path = os.path.join(CONNECTIVITY_PATH, 'connectivity_filtered.json')
    with open(filtered_conn_path, 'w') as f:
        json.dump(filtered_connectivity, f, indent=2)
    print(f"  Saved to: {filtered_conn_path}")

    # ================================================================
    # Phase 1: Compute texture scores for filtered pairs
    # ================================================================
    tasks = []
    for pid_str, fits_list in filtered_connectivity.items():
        pid_a = int(pid_str)
        for si, matches in enumerate(fits_list):
            for m in matches:
                pid_b = m['pid']
                sj = m['si']
                tasks.append((pid_a, si, pid_b, sj, m))

    total = len(tasks)
    print(f"\n--- Phase 1: Computing texture scores ({total} pairs) ---")

    n_workers = min(os.cpu_count() or 1, 8)
    print(f"  Using {n_workers} workers")

    t0 = time.time()
    done = 0

    task_args = [(t[0], t[1], t[2], t[3]) for t in tasks]
    task_meta = {(t[0], t[1], t[2], t[3]): t[4] for t in tasks}

    results = {}
    skip_counts = {'no_side_data': 0, 'is_edge': 0, 'no_color_data': 0, 'too_few_samples': 0}

    with multiprocessing.Pool(processes=n_workers) as pool:
        for pid_a, si, pid_b, sj, result in pool.imap_unordered(_verify_single, task_args):
            done += 1
            pid_str = str(pid_a)
            if pid_str not in results:
                results[pid_str] = [[], [], [], []]

            m = task_meta[(pid_a, si, pid_b, sj)]
            reason = result['reason']

            if reason in skip_counts:
                skip_counts[reason] += 1

            results[pid_str][si].append({
                'pid': pid_b,
                'si': sj,
                'error': m['error'],
                'reject': reason in skip_counts,
                'ncc': round(result['ncc'], 4),
                'color_diff': round(result['color_diff_mean'], 2),
                'grad_score': round(result['grad_score'], 3) if result['grad_score'] is not None else None,
                'texture_level': result['texture_level'],
                'reason': reason if reason in skip_counts else 'ok',
                'n_samples': result['n_samples'],
            })

            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  Progress: {done}/{total} ({done/total*100:.1f}%) "
                      f"[{elapsed:.1f}s elapsed, ETA {eta:.0f}s]")

    phase1_time = time.time() - t0

    # ================================================================
    # Phase 2: Filter by NCC - reject if ncc < best_ncc * NCC_FILTER_RATIO
    # ================================================================
    print(f"\n--- Phase 2: NCC relative filter (< {NCC_FILTER_RATIO} of best) ---")
    rejected_by_ncc = 0
    total_piece_sides = 0

    for pid_str in results:
        for si in range(4):
            valid = [m for m in results[pid_str][si] if m['reason'] == 'ok']
            if len(valid) < 2:
                continue

            total_piece_sides += 1

            best_ncc = max(m['ncc'] for m in valid)
            threshold_ncc = best_ncc * NCC_FILTER_RATIO

            for m in valid:
                if m['ncc'] < threshold_ncc:
                    m['reject'] = True
                    m['reason'] = 'ncc_relative_reject'
                    rejected_by_ncc += 1

    for pid_str in results:
        for si in range(4):
            results[pid_str][si].sort(key=lambda x: (x['reject'], -_compute_score(x['ncc'], x['color_diff'])))

    report_path = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    total_valid = sum(
        1 for pid_str in results for si in range(4)
        for m in results[pid_str][si] if not m['reject']
    )
    total_verified = sum(
        1 for pid_str in results for si in range(4)
        for m in results[pid_str][si] if m['reason'] == 'ok'
    )

    kept_per_side = []
    for pid_str in results:
        for si in range(4):
            valid = [m for m in results[pid_str][si] if not m['reject']]
            if valid:
                kept_per_side.append(len(valid))

    import numpy as np
    kps = np.array(kept_per_side) if kept_per_side else np.array([0])

    total_skipped = sum(skip_counts.values())

    print(f"\n{'=' * 60}")
    print(f"Texture Verification Report")
    print(f"{'=' * 60}")
    print(f"  Phase 0 - Error pre-filter:")
    print(f"    Original candidates:     {original_total}")
    print(f"    After error filter:      {filtered_total}")
    print(f"    Removed:                 {removed_by_error} ({removed_by_error / max(original_total, 1) * 100:.1f}%)")
    print(f"  Phase 1 - Texture scoring:")
    print(f"    Pairs scored:            {total}")
    print(f"    Skipped (no side data):  {skip_counts['no_side_data']}")
    print(f"    Skipped (is edge):       {skip_counts['is_edge']}")
    print(f"    Skipped (no color data): {skip_counts['no_color_data']}")
    print(f"    Skipped (too few samps): {skip_counts['too_few_samples']}")
    print(f"    Valid for filtering:     {total_verified}")
    print(f"  Phase 2 - NCC relative filter:")
    print(f"    Piece-sides with >=2:    {total_piece_sides}")
    print(f"    Rejected by NCC:         {rejected_by_ncc} ({rejected_by_ncc / max(total_verified, 1) * 100:.1f}%)")
    print(f"  Summary:")
    print(f"    Total kept:              {total_valid}")
    print(f"    Candidates per side:     mean={np.mean(kps):.1f} P50={np.median(kps):.0f} max={np.max(kps)}")
    print(f"    Phase 1 time:            {phase1_time:.1f}s")
    print(f"  Files:")
    print(f"    Filtered connectivity:   {filtered_conn_path}")
    print(f"    Verification report:     {report_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
