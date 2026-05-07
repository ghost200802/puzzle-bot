import os
import sys
import json
import time
import multiprocessing

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR, VECTOR_DIR
from common.texture_verify import verify_match

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, CONNECTIVITY_DIR)
COLOR_PATH = os.path.join(OUTPUT_DIR, '2_piece_colors')
if not os.path.isdir(COLOR_PATH):
    COLOR_PATH = os.path.join(OUTPUT_DIR, VECTOR_DIR)


def _verify_single(args):
    pid_a, si, pid_b, sj = args
    result = verify_match(COLOR_PATH, DEDUPED_PATH, pid_a, si, pid_b, sj)
    return (pid_a, si, pid_b, sj, result)


def main():
    conn_path = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
    if not os.path.exists(conn_path):
        print(f"Error: connectivity.json not found: {conn_path}")
        return

    with open(conn_path) as f:
        connectivity = json.load(f)

    print("=" * 60)
    print("Texture Continuity Verification")
    print(f"  Connectivity: {conn_path}")
    print(f"  Color images: {COLOR_PATH}")
    print(f"  Side data:    {DEDUPED_PATH}")
    print(f"  Pieces: {len(connectivity)}")
    print("=" * 60)

    tasks = []
    for pid_str, fits_list in connectivity.items():
        pid_a = int(pid_str)
        for si, matches in enumerate(fits_list):
            for m in matches:
                pid_b = m['pid']
                sj = m['si']
                tasks.append((pid_a, si, pid_b, sj, m))

    total = len(tasks)
    print(f"\n  Total match pairs to verify: {total}")

    n_workers = min(os.cpu_count() or 1, 8)
    print(f"  Using {n_workers} workers")

    t0 = time.time()
    done = 0

    task_args = [(t[0], t[1], t[2], t[3]) for t in tasks]
    task_meta = {(t[0], t[1], t[2], t[3]): t[4] for t in tasks}

    results = {}
    verified = 0
    rejected = 0
    no_color = 0
    no_side = 0
    is_edge_count = 0
    too_few = 0

    with multiprocessing.Pool(processes=n_workers) as pool:
        for pid_a, si, pid_b, sj, result in pool.imap_unordered(_verify_single, task_args):
            done += 1
            pid_str = str(pid_a)
            if pid_str not in results:
                results[pid_str] = [[], [], [], []]

            m = task_meta[(pid_a, si, pid_b, sj)]

            results[pid_str][si].append({
                'pid': pid_b,
                'si': sj,
                'error': m['error'],
                'reject': result['reject'],
                'color_diff': round(result['color_diff_mean'], 2),
                'ncc': round(result['ncc'], 4),
                'grad_score': round(result['grad_score'], 3) if result['grad_score'] is not None else None,
                'texture_level': result['texture_level'],
                'reason': result['reason'],
                'n_samples': result['n_samples'],
            })

            reason = result['reason']
            if reason == 'no_color_data':
                no_color += 1
            elif reason == 'no_side_data':
                no_side += 1
            elif reason == 'is_edge':
                is_edge_count += 1
            elif reason == 'too_few_samples':
                too_few += 1
            else:
                verified += 1
                if result['reject']:
                    rejected += 1

            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  Progress: {done}/{total} ({done/total*100:.1f}%) "
                      f"[{elapsed:.1f}s elapsed, ETA {eta:.0f}s] "
                      f"rejected={rejected}")

    elapsed = time.time() - t0

    for pid_str in results:
        for si in range(4):
            results[pid_str][si].sort(key=lambda x: x['error'])

    report_path = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    kept = verified - rejected
    print(f"\n{'=' * 60}")
    print(f"Texture Verification Report")
    print(f"{'=' * 60}")
    print(f"  Total matches:     {total}")
    print(f"  No side data:      {no_side}")
    print(f"  Is edge (skipped): {is_edge_count}")
    print(f"  No color data:     {no_color}")
    print(f"  Too few samples:   {too_few}")
    print(f"  Verified:          {verified}")
    print(f"  Rejected:          {rejected} ({rejected / max(verified, 1) * 100:.1f}%)")
    print(f"  Kept:              {kept}")
    print(f"  Time:              {elapsed:.1f}s")
    print(f"  Report saved to:   {report_path}")
    print(f"{'=' * 60}")

    if rejected > 0:
        print(f"\nRejected matches:")
        for pid_str, fits_list in sorted(results.items(), key=lambda x: int(x[0])):
            for si, matches in enumerate(fits_list):
                for m in matches:
                    if m['reject']:
                        grad_str = f"{m['grad_score']:.3f}" if m['grad_score'] is not None else "N/A"
                        print(f"  {pid_str}[{si}] -> {m['pid']}[{m['si']}] "
                              f"err={m['error']} ΔE={m['color_diff']:.1f} "
                              f"ncc={m['ncc']:.3f} grad={grad_str} "
                              f"tex={m['texture_level']} reason={m['reason']}")


if __name__ == '__main__':
    main()
