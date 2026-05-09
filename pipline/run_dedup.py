import os, sys, json, math, shutil, itertools
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'check'))
from common.config import VECTOR_DIR, DEDUPED_DIR, CHECK_DIR

OUTPUT_ROOT = os.environ.get('PUZZLE_OUTPUT_ROOT', '')
VECTOR_PATH = os.path.join(OUTPUT_ROOT, VECTOR_DIR)
DEDUPED_PATH = os.path.join(OUTPUT_ROOT, DEDUPED_DIR)
CHECK_PATH = os.path.join(OUTPUT_ROOT, CHECK_DIR)
BMP_DIR = os.path.join(OUTPUT_ROOT, '2_piece_bmps')
COLOR_DIR = os.path.join(OUTPUT_ROOT, '2_piece_colors')

PROFILE_N = 50

SIDE_RMSE = 0.08
LENGTH_RATIO = 0.65
NCC_MIN = 0.65

META_PATH = os.path.join(CHECK_PATH, 'dedup_match_meta.json')
NUM_WORKERS = min(max(1, multiprocessing.cpu_count() - 2), 14)


def compute_contrast(pid, color_dir):
    path = os.path.join(color_dir, f'piece_{pid}.png')
    if not os.path.exists(path):
        return 0.0
    img = np.array(Image.open(path).convert('RGBA'))
    mask = img[:, :, 3] > 128
    if np.sum(mask) < 100:
        return 0.0
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    vals = gray[mask].astype(np.float64)
    return float(np.std(vals))


def load_pieces(vector_path):
    pieces = {}
    vp = Path(vector_path)
    for path in sorted(vp.glob("side_*_0.json")):
        pid = int(path.parts[-1].split('_')[1])
        sides_data = []
        valid = True
        for j in range(4):
            jpath = vp / f'side_{pid}_{j}.json'
            try:
                with open(jpath) as f:
                    data = json.load(f)
                if not data.get('vertices') or len(data['vertices']) < 2:
                    valid = False
                    break
                sides_data.append(data)
            except Exception as e:
                print(f"  Warning: piece {pid} side {j}: {e}")
                valid = False
                break
        if valid and len(sides_data) == 4:
            pieces[pid] = sides_data
    return pieces


def classify_side(side_data):
    if side_data['is_edge']:
        return 'F'
    vertices = np.array(side_data['vertices'], dtype=float)
    piece_center = np.array(side_data.get('piece_center', [0, 0]), dtype=float)
    if len(vertices) < 3:
        return 'F'
    p1, p2 = vertices[0], vertices[-1]
    edge_vec = p2 - p1
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1:
        return 'F'
    perp = np.array([-edge_vec[1], edge_vec[0]])
    center_side = np.dot(piece_center - p1, perp)
    mid_verts = vertices[len(vertices) // 4:3 * len(vertices) // 4]
    mid_offsets = np.dot(mid_verts - p1, perp)
    avg_offset = np.mean(mid_offsets)
    if center_side * avg_offset < 0:
        return 'T'
    else:
        return 'B'


def get_side_length(side_data):
    vertices = np.array(side_data['vertices'], dtype=float)
    if len(vertices) < 2:
        return 0
    return np.linalg.norm(vertices[-1] - vertices[0])


def resample_polyline(vertices, n):
    pts = np.array(vertices, dtype=float)
    if len(pts) < 2:
        return np.tile(pts[0] if len(pts) > 0 else np.array([0, 0]), (n, 1))
    dists = [0.0]
    for i in range(1, len(pts)):
        dists.append(dists[-1] + np.linalg.norm(pts[i] - pts[i - 1]))
    total = dists[-1]
    if total < 1e-6:
        return np.tile(pts[0], (n, 1))
    target = np.linspace(0, total, n)
    result = np.zeros((n, 2))
    for i, td in enumerate(target):
        idx = np.searchsorted(dists, td, side='right') - 1
        idx = max(0, min(idx, len(pts) - 2))
        seg = dists[idx + 1] - dists[idx]
        if seg < 1e-6:
            result[i] = pts[idx]
        else:
            t = (td - dists[idx]) / seg
            result[i] = pts[idx] + t * (pts[idx + 1] - pts[idx])
    return result


def get_side_profile(side_data, n=PROFILE_N):
    vertices = np.array(side_data['vertices'], dtype=float)
    if len(vertices) < 2:
        return np.zeros(n)
    p1, p2 = vertices[0], vertices[-1]
    length = np.linalg.norm(p2 - p1)
    if length < 1:
        return np.zeros(n)
    direction = (p2 - p1) / length
    normal = np.array([-direction[1], direction[0]])
    resampled = resample_polyline(vertices, n)
    profile = np.dot(resampled - p1, normal) / length
    return profile


def get_piece_signature(sides_data):
    types = [classify_side(s) for s in sides_data]
    edge_count = sum(1 for t in types if t == 'F')
    return edge_count, types


def normalize_types(types):
    n = len(types)
    rotations = [tuple(types[i:] + types[:i]) for i in range(n)]
    return min(rotations)


def compare_side_profiles(prof_a, prof_b):
    if len(prof_a) != len(prof_b):
        return 999.0
    return np.sqrt(np.mean((prof_a - prof_b) ** 2))


def match_two_pieces(sides_a, sides_b, rmse_thresh, ratio_min):
    best_error = float('inf')
    best_rot = -1
    n_matching = 0
    for rot in range(4):
        rot_b = sides_b[rot:] + sides_b[:rot]
        total_err = 0.0
        matched = 0
        ok = True
        for i in range(4):
            if sides_a[i]['is_edge'] and rot_b[i]['is_edge']:
                continue
            len_a = get_side_length(sides_a[i])
            len_b = get_side_length(rot_b[i])
            if len_a > 0 and len_b > 0:
                ratio = min(len_a, len_b) / max(len_a, len_b)
                if ratio < ratio_min:
                    ok = False
                    break
            prof_a = get_side_profile(sides_a[i])
            prof_b = get_side_profile(rot_b[i])
            err = compare_side_profiles(prof_a, prof_b)
            if err > rmse_thresh:
                ok = False
                break
            total_err += err
            matched += 1
        non_edge_count = sum(1 for i in range(4)
                            if not (sides_a[i]['is_edge'] and rot_b[i]['is_edge']))
        if ok and matched >= non_edge_count and non_edge_count > 0:
            if total_err < best_error:
                best_error = total_err
                best_rot = rot
                n_matching = matched
    return best_rot >= 0, best_rot, best_error, n_matching


def _geo_match_task(args):
    pa, pb, sides_a_serializable, sides_b_serializable, rmse_thresh, ratio_min = args
    is_match, rot, err, n_match = match_two_pieces(
        sides_a_serializable, sides_b_serializable, rmse_thresh, ratio_min)
    return pa, pb, is_match, rot, err, n_match


def _compute_ncc_task(args):
    pid_a, pid_b, rot = args
    path_a = os.path.join(COLOR_DIR, f'piece_{pid_a}.png')
    path_b = os.path.join(COLOR_DIR, f'piece_{pid_b}.png')
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        return pid_a, pid_b, rot, -1.0

    img_a = np.array(Image.open(path_a).convert('RGBA'))
    img_b = np.array(Image.open(path_b).convert('RGBA'))

    vp = Path(VECTOR_PATH)
    corners_a = []
    corners_b = []
    for j in range(4):
        with open(vp / f'side_{pid_a}_{j}.json') as f:
            corners_a.append(np.array(json.load(f)['vertices'][0], dtype=np.float64))
        with open(vp / f'side_{pid_b}_{j}.json') as f:
            corners_b.append(np.array(json.load(f)['vertices'][0], dtype=np.float64))

    src_pts = np.array([corners_b[(i + rot) % 4] for i in range(4)], dtype=np.float32)
    dst_pts = np.array(corners_a, dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        return pid_a, pid_b, rot, -1.0

    h, w = img_a.shape[:2]
    warped_b = cv2.warpPerspective(img_b, H, (w, h))

    mask_a = img_a[:, :, 3] > 128
    mask_b = warped_b[:, :, 3] > 128
    overlap = mask_a & mask_b

    if np.sum(overlap) < 500:
        return pid_a, pid_b, rot, -1.0

    gray_a = cv2.cvtColor(img_a[:, :, :3], cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(warped_b[:, :, :3], cv2.COLOR_RGB2GRAY)

    ga = gray_a[overlap].astype(np.float64)
    gb = gray_b[overlap].astype(np.float64)

    src_hist, _ = np.histogram(gb.astype(np.uint8), bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(ga.astype(np.uint8), bins=256, range=(0, 256))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1
    ref_cdf /= ref_cdf[-1] if ref_cdf[-1] > 0 else 1
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapping[i] = int(np.argmin(np.abs(ref_cdf - src_cdf[i])))
    gb_matched = mapping[gb.astype(np.uint8)].astype(np.float64)

    ga_norm = ga - ga.mean()
    gb_norm = gb_matched - gb_matched.mean()
    denom = np.sqrt(np.sum(ga_norm ** 2) * np.sum(gb_norm ** 2))
    if denom < 1e-6:
        return pid_a, pid_b, rot, -1.0
    ncc_gray = np.sum(ga_norm * gb_norm) / denom

    edges_a = cv2.Canny(gray_a, 50, 150)
    gray_b_matched_full = mapping[gray_b]
    edges_b = cv2.Canny(gray_b_matched_full, 50, 150)
    ea = edges_a[overlap].astype(np.float64)
    eb = edges_b[overlap].astype(np.float64)
    ea_norm = ea - ea.mean()
    eb_norm = eb - eb.mean()
    denom_e = np.sqrt(np.sum(ea_norm ** 2) * np.sum(eb_norm ** 2))
    if denom_e < 1e-6:
        ncc_edge = 0.0
    else:
        ncc_edge = np.sum(ea_norm * eb_norm) / denom_e

    ncc = max(ncc_gray, ncc_edge)
    return pid_a, pid_b, rot, round(ncc, 4)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Deduplicate puzzle pieces')
    parser.add_argument('-o', '--output', default=None,
                        help='Output root directory')
    args = parser.parse_args()

    if args.output:
        set_output_root(args.output)

    VECTOR_PATH = get_vector_path()
    DEDUPED_PATH = get_deduped_path()
    CHECK_PATH = get_check_path()
    BMP_DIR = os.path.join(get_output_dir(), '2_piece_bmps')
    COLOR_DIR = get_color_path()
    META_PATH = os.path.join(CHECK_PATH, 'dedup_match_meta.json')

    if os.path.exists(DEDUPED_PATH):
        shutil.rmtree(DEDUPED_PATH)
    os.makedirs(DEDUPED_PATH, exist_ok=True)
    os.makedirs(CHECK_PATH, exist_ok=True)

    print("=" * 60)
    print("Deduplication Pipeline (Geometric + HistMatch NCC)")
    print(f"RMSE<={SIDE_RMSE}  ratio>={LENGTH_RATIO}  NCC>={NCC_MIN}")
    print(f"Workers: {NUM_WORKERS}")
    print("=" * 60)

    pieces = load_pieces(VECTOR_PATH)
    print(f"Loaded {len(pieces)} pieces")

    signatures = {}
    for pid, sd in pieces.items():
        signatures[pid] = get_piece_signature(sd)

    sig_groups = {}
    for pid, (ec, types) in signatures.items():
        norm_sig = (ec, normalize_types(types))
        sig_groups.setdefault(norm_sig, []).append(pid)

    print(f"\nSignature groups: {len(sig_groups)}")
    for sig, pids in sorted(sig_groups.items(), key=lambda x: -len(x[1])):
        ec, types = sig
        print(f"  edges={ec} {types}: {len(pids)} pieces")

    parent = {pid: pid for pid in pieces}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    all_pairs_meta = {}

    # ---- Geometric matching (multiprocess) ----
    print(f"\n--- Geometric matching (RMSE<={SIDE_RMSE}, ratio>={LENGTH_RATIO}) ---")
    geo_tasks = []
    for sig, pids in sig_groups.items():
        if len(pids) < 2:
            continue
        for i, j in itertools.combinations(range(len(pids)), 2):
            pa, pb = pids[i], pids[j]
            geo_tasks.append((pa, pb, pieces[pa], pieces[pb], SIDE_RMSE, LENGTH_RATIO))

    print(f"  Total pairs to check: {len(geo_tasks)}")
    print(f"  Running geometric matching (multiprocess)...")

    geo_matches = []
    n_done = 0
    batch_size = max(1, len(geo_tasks) // 20)
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_geo_match_task, t): (t[0], t[1]) for t in geo_tasks}
        for future in as_completed(futures):
            pa, pb, is_match, rot, err, n_match = future.result()
            n_done += 1
            if n_done % batch_size == 0:
                print(f"    Progress: {n_done}/{len(geo_tasks)} ({len(geo_matches)} matches so far)")
            if is_match:
                geo_matches.append((pa, pb, err, n_match, rot))

    print(f"  Geometric matches: {len(geo_matches)}")

    # ---- NCC texture verification (multiprocess) ----
    print(f"  Computing NCC with histogram matching (multiprocess)...")
    ncc_tasks = [(pa, pb, rot) for pa, pb, err, n_match, rot in geo_matches]
    ncc_results = {}
    n_ncc_done = 0
    ncc_batch = max(1, len(ncc_tasks) // 10)
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_compute_ncc_task, t): (t[0], t[1]) for t in ncc_tasks}
        for future in as_completed(futures):
            pid_a, pid_b, rot, ncc = future.result()
            ncc_results[(min(pid_a, pid_b), max(pid_a, pid_b))] = (rot, ncc)
            n_ncc_done += 1
            if n_ncc_done % ncc_batch == 0:
                print(f"    NCC progress: {n_ncc_done}/{len(ncc_tasks)}")

    confirmed_count = 0
    rejected_count = 0
    for pa, pb, err, n_match, rot in geo_matches:
        key = (min(pa, pb), max(pa, pb))
        _, ncc = ncc_results.get(key, (rot, 0.0))
        confirmed = ncc >= NCC_MIN
        meta = {
            'rmse_thresh': SIDE_RMSE, 'length_ratio_thresh': LENGTH_RATIO,
            'ncc_thresh': NCC_MIN, 'rot': rot, 'total_rmse': round(err, 4),
            'n_matching_sides': n_match, 'ncc': ncc, 'confirmed': confirmed,
        }
        all_pairs_meta[key] = meta
        if confirmed:
            union(pa, pb)
            confirmed_count += 1
        else:
            rejected_count += 1

    print(f"  Confirmed: {confirmed_count}, rejected by texture: {rejected_count}")

    groups = {}
    for pid in pieces:
        root = find(pid)
        groups.setdefault(root, []).append(pid)

    uniques = set()
    contrast_cache = {}
    for pid in pieces:
        contrast_cache[pid] = compute_contrast(pid, COLOR_DIR)
    for root, members in groups.items():
        best = max(members, key=lambda pid: contrast_cache[pid])
        uniques.add(best)
        if len(members) > 1:
            print(f"  Group {sorted(members)} -> best=#{best} (contrast={contrast_cache[best]:.1f})")

    print(f"\n{'=' * 60}")
    print(f"RESULT: {len(uniques)} unique pieces (from {len(pieces)} total)")
    print(f"Duplicates removed: {len(pieces) - len(uniques)}")
    print(f"{'=' * 60}")

    dup_groups = {root: members for root, members in groups.items() if len(members) > 1}
    if dup_groups:
        print(f"\nDuplicate groups ({len(dup_groups)}):")
        for root, members in sorted(dup_groups.items()):
            print(f"  {members}")

    for pid in uniques:
        for i in range(4):
            sf = f'side_{pid}_{i}.json'
            src = os.path.join(VECTOR_PATH, sf)
            dst = os.path.join(DEDUPED_PATH, sf)
            if os.path.exists(src):
                shutil.copyfile(src, dst)
        svg_files = [f for f in os.listdir(VECTOR_PATH)
                     if f.startswith(f'{pid}_') and f.endswith('.svg')]
        for svg in svg_files:
            shutil.copyfile(os.path.join(VECTOR_PATH, svg),
                            os.path.join(DEDUPED_PATH, svg))

    meta_serializable = {}
    for k, v in all_pairs_meta.items():
        meta_serializable[f"{k[0]}_{k[1]}"] = {
            key: (bool(val) if isinstance(val, (np.bool_,)) else val)
            for key, val in v.items()
        }
    with open(META_PATH, 'w') as f:
        json.dump(meta_serializable, f, indent=2)

    print(f"\nSaved to {DEDUPED_PATH}/")
    print(f"Match metadata saved to {META_PATH}")

    print(f"\n{'='*60}")
    print("Generating duplicate groups visualization...")
    print(f"{'='*60}")
    from show_dup_groups import main as show_dup_main
    show_dup_main()

    print(f"\n{'='*60}")
    print("Generating signature groups visualization...")
    print(f"{'='*60}")
    from show_sig_groups import main as show_sig_main
    show_sig_main()

    return len(uniques)


if __name__ == '__main__':
    main()
