import os, sys, glob
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'check'))
from common.config import VECTOR_DIR
from common import vector
from config import set_output_root, get_output_dir, get_vector_path

NUM_WORKERS = min(max(1, multiprocessing.cpu_count() - 2), 14)


def _process_one(args):
    try:
        vector.load_and_vectorize(args)
        return args[1], True, None
    except Exception as e:
        return args[1], False, str(e)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Vectorize piece outlines')
    parser.add_argument('-o', '--output', default=None,
                        help='Output root directory')
    args = parser.parse_args()

    if args.output:
        set_output_root(args.output)

    BMP_DIR = os.path.join(get_output_dir(), '2_piece_bmps')
    VECTOR_OUT = get_vector_path()

    os.makedirs(VECTOR_OUT, exist_ok=True)

    bmp_files = sorted(glob.glob(os.path.join(BMP_DIR, 'piece_*.bmp')))
    if not bmp_files:
        print(f"No BMP files found in {BMP_DIR}")
        return

    print("=" * 60)
    print("Vectorization Pipeline (multiprocess)")
    print(f"Input:  {BMP_DIR} ({len(bmp_files)} files)")
    print(f"Output: {VECTOR_OUT}")
    print(f"Workers: {NUM_WORKERS}")
    print("=" * 60)

    all_args = []
    for bmp_path in bmp_files:
        basename = os.path.basename(bmp_path)
        pid = int(basename.replace('piece_', '').replace('.bmp', ''))
        with Image.open(bmp_path) as img:
            w, h = img.size
        metadata = {
            'original_photo_name': 'puzzle_new',
            'photo_space_origin': (0, 0),
            'photo_space_centroid': [w // 2, h // 2],
            'photo_width': w,
            'photo_height': h,
            'is_complete': True,
        }
        all_args.append([bmp_path, pid, VECTOR_OUT, metadata, (0, 0), 1.0, False])

    success = 0
    failed = 0
    failed_ids = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_process_one, a): a[1] for a in all_args}
        for future in as_completed(futures):
            pid, ok, err = future.result()
            if ok:
                success += 1
            else:
                failed += 1
                failed_ids.append(pid)
                print(f"  ERROR piece {pid}: {err}")
            done = success + failed
            if done % 20 == 0 or done == len(all_args):
                print(f"  Progress: {done}/{len(all_args)}")

    print(f"\n{'=' * 60}")
    print(f"Vectorization complete: {success} success, {failed} failed")
    if failed_ids:
        print(f"Failed pieces: {failed_ids}")
    print(f"Output: {VECTOR_OUT}/")
    print(f"{'=' * 60}")

    from check_squareness import run_check
    run_check(get_output_dir())


if __name__ == '__main__':
    main()
