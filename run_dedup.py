import os, sys, json, math, shutil, itertools
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from common.config import VECTOR_DIR, DEDUPED_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'puzzle_new')
VECTOR_PATH = os.path.join(OUTPUT_DIR, VECTOR_DIR)
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)
BMP_DIR = os.path.join(OUTPUT_DIR, '2_piece_bmps')
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input', 'puzzles')

PROFILE_N = 50
SIDE_RMSE_THRESHOLD = 0.025
SIDE_LENGTH_RATIO_MIN = 0.85
COLOR_HIST_BINS = 32
COLOR_CORR_THRESHOLD = 0.85


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
    rmse = np.sqrt(np.mean((prof_a - prof_b) ** 2))
    return rmse


def match_two_pieces(sides_a, sides_b, threshold=SIDE_RMSE_THRESHOLD):
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
                if ratio < SIDE_LENGTH_RATIO_MIN:
                    ok = False
                    break
            prof_a = get_side_profile(sides_a[i])
            prof_b = get_side_profile(rot_b[i])
            err = compare_side_profiles(prof_a, prof_b)
            if err > threshold:
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


def extract_piece_color_histogram(pid, source_info):
    if source_info is None:
        return None
    src_file = source_info.get('source_file')
    bbox = source_info.get('bbox')
    if not src_file or not bbox:
        return None
    img_path = os.path.join(INPUT_DIR, src_file)
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path)
    rgba = np.array(img)
    if rgba.shape[2] != 4:
        return None
    alpha = rgba[:, :, 3]
    x0, y0, x1, y1 = bbox
    pad = 5
    x0p, y0p = max(0, x0 - pad), max(0, y0 - pad)
    x1p, y1p = min(alpha.shape[1], x1 + pad), min(alpha.shape[0], y1 + pad)
    alpha_crop = alpha[y0p:y1p, x0p:x1p]
    rgb_crop = rgba[y0p:y1p, x0p:x1p, :3]
    mask = alpha_crop > 128
    pixels = rgb_crop[mask]
    if len(pixels) < 100:
        return None
    pixels_bgr = pixels[:, ::-1].copy()
    hsv = cv2.cvtColor(pixels_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h_hist = np.histogram(hsv[:, 0], bins=COLOR_HIST_BINS, range=(0, 180))[0].astype(float)
    s_hist = np.histogram(hsv[:, 1], bins=COLOR_HIST_BINS, range=(0, 256))[0].astype(float)
    v_hist = np.histogram(hsv[:, 2], bins=COLOR_HIST_BINS, range=(0, 256))[0].astype(float)
    h_hist /= (h_hist.sum() + 1e-8)
    s_hist /= (s_hist.sum() + 1e-8)
    v_hist /= (v_hist.sum() + 1e-8)
    return np.concatenate([h_hist, s_hist, v_hist])


def color_histogram_correlation(hist_a, hist_b):
    if hist_a is None or hist_b is None:
        return 0.0
    corr_h = np.corrcoef(hist_a[:COLOR_HIST_BINS], hist_b[:COLOR_HIST_BINS])[0, 1]
    corr_s = np.corrcoef(hist_a[COLOR_HIST_BINS:2*COLOR_HIST_BINS], hist_b[COLOR_HIST_BINS:2*COLOR_HIST_BINS])[0, 1]
    corr_v = np.corrcoef(hist_a[2*COLOR_HIST_BINS:], hist_b[2*COLOR_HIST_BINS:])[0, 1]
    return (corr_h + corr_s + corr_v) / 3.0


def load_source_info():
    info_path = os.path.join(OUTPUT_DIR, 'piece_source_info.json')
    if os.path.exists(info_path):
        with open(info_path) as f:
            return json.load(f)
    return {}


def main():
    if os.path.exists(DEDUPED_PATH):
        shutil.rmtree(DEDUPED_PATH)
    os.makedirs(DEDUPED_PATH, exist_ok=True)

    print("=" * 60)
    print("Deduplication Pipeline (Geometric + Color)")
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

    dup_pairs = []
    for sig, pids in sig_groups.items():
        if len(pids) < 2:
            continue
        for i, j in itertools.combinations(range(len(pids)), 2):
            pa, pb = pids[i], pids[j]
            is_match, rot, err, n_match = match_two_pieces(pieces[pa], pieces[pb])
            if is_match:
                dup_pairs.append((pa, pb, err, n_match, rot))
                union(pa, pb)

    print(f"\nGeometric duplicate pairs: {len(dup_pairs)}")

    source_info = load_source_info()
    color_hists = {}
    if source_info:
        print(f"\nExtracting color features for {len(source_info)} pieces...")
        for pid_str, info in source_info.items():
            pid = int(pid_str)
            if pid in pieces:
                color_hists[pid] = extract_piece_color_histogram(pid, info)
        print(f"  Extracted {sum(1 for v in color_hists.values() if v is not None)} color histograms")

    color_verified = 0
    color_rejected = 0
    if color_hists:
        for pa, pb, err, n_match, rot in dup_pairs:
            ha, hb = color_hists.get(pa), color_hists.get(pb)
            if ha is not None and hb is not None:
                corr = color_histogram_correlation(ha, hb)
                if corr >= COLOR_CORR_THRESHOLD:
                    color_verified += 1
                else:
                    color_rejected += 1

    groups = {}
    for pid in pieces:
        root = find(pid)
        groups.setdefault(root, []).append(pid)

    uniques = set()
    for root, members in groups.items():
        best = max(members, key=lambda pid: sum(len(pieces[pid][i]['vertices']) for i in range(4)))
        uniques.add(best)

    print(f"\n{'=' * 60}")
    print(f"RESULT: {len(uniques)} unique pieces (from {len(pieces)} total)")
    print(f"Duplicates removed: {len(pieces) - len(uniques)}")
    if color_verified:
        print(f"Color verification: {color_verified} confirmed, {color_rejected} rejected")
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

    print(f"\nSaved to {DEDUPED_PATH}/")
    return len(uniques)


if __name__ == '__main__':
    main()
