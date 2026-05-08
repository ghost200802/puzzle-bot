import os
import sys
import json
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, '5_connectivity')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, '3_vector')
COLOR_PATH = os.path.join(OUTPUT_DIR, '2_piece_colors')

TARGET_PIECES = [4, 137]

print("=" * 70)
print("DEBUG: Piece 4 and 137 Connectivity Analysis")
print("=" * 70)

print("\n[1] Checking piece_edge_info.json")
edge_info_path = os.path.join(CONNECTIVITY_PATH, 'piece_edge_info.json')
with open(edge_info_path) as f:
    edge_info = json.load(f)
for pid in TARGET_PIECES:
    pid_s = str(pid)
    if pid_s in edge_info:
        flags = edge_info[pid_s]
        print(f"  Piece {pid}: edge flags = {flags}, is_edge sides = {[i for i, f in enumerate(flags) if f]}")
    else:
        print(f"  Piece {pid}: NOT FOUND in piece_edge_info")

print("\n[2] Checking side data for each piece")
for pid in TARGET_PIECES:
    print(f"\n  --- Piece {pid} ---")
    for si in range(4):
        side_path = os.path.join(DEDUPED_PATH, f'side_{pid}_{si}.json')
        if os.path.exists(side_path):
            with open(side_path) as f:
                data = json.load(f)
            is_edge = data.get('is_edge', False)
            verts = data.get('vertices', [])
            piece_center = data.get('piece_center', [])
            print(f"    Side {si}: is_edge={is_edge}, vertices={len(verts)}, piece_center={piece_center}")
            if verts:
                p1 = np.array(verts[0])
                p2 = np.array(verts[-1])
                length = np.linalg.norm(p2 - p1)
                print(f"      length={length:.2f}, p1={p1.tolist()}, p2={p2.tolist()}")
        else:
            print(f"    Side {si}: FILE NOT FOUND")

print("\n[3] Checking connectivity.json")
conn_path = os.path.join(CONNECTIVITY_PATH, 'connectivity.json')
with open(conn_path) as f:
    connectivity = json.load(f)

for pid in TARGET_PIECES:
    pid_s = str(pid)
    print(f"\n  --- Piece {pid} connectivity ---")
    if pid_s not in connectivity:
        print(f"    NOT FOUND in connectivity!")
        continue

    fits_list = connectivity[pid_s]

    for si in range(4):
        matches = fits_list[si] if si < len(fits_list) else []
        if matches:
            print(f"    Side {si}: {len(matches)} matches")
            for m in matches[:5]:
                print(f"      -> pid={m['pid']}, si={m['si']}, error={m['error']} (real={m['error']/1000:.4f}), "
                      f"len_diff={m.get('len_diff', 'N/A')}, "
                      f"convex_a={m.get('convex_a')}, convex_b={m.get('convex_b')}")
        else:
            side_path = os.path.join(DEDUPED_PATH, f'side_{pid}_{si}.json')
            if os.path.exists(side_path):
                with open(side_path) as f2:
                    data = json.load(f2)
                print(f"    Side {si}: NO MATCHES (is_edge={data.get('is_edge', False)})")
            else:
                print(f"    Side {si}: NO MATCHES (no side data file)")

print("\n[4] Checking if 4<->137 are mutual matches")
for pid_a in TARGET_PIECES:
    pid_b = [p for p in TARGET_PIECES if p != pid_a][0]
    pid_a_s = str(pid_a)
    if pid_a_s in connectivity:
        fits_list = connectivity[pid_a_s]
        for si in range(4):
            matches = fits_list[si] if si < len(fits_list) else []
            for m in matches:
                if m['pid'] == pid_b:
                    print(f"  Piece {pid_a}[{si}] -> Piece {pid_b}[{m['si']}]: "
                          f"error={m['error']} (real={m['error']/1000:.4f}), "
                          f"len_diff={m.get('len_diff', 'N/A')}")

print("\n[5] Checking connectivity_filtered.json")
filtered_path = os.path.join(CONNECTIVITY_PATH, 'connectivity_filtered.json')
if os.path.exists(filtered_path):
    with open(filtered_path) as f:
        filtered = json.load(f)

    for pid in TARGET_PIECES:
        pid_s = str(pid)
        print(f"\n  --- Piece {pid} filtered connectivity ---")
        if pid_s not in filtered:
            print(f"    NOT FOUND!")
            continue
        fits_list = filtered[pid_s]
        for si in range(4):
            matches = fits_list[si] if si < len(fits_list) else []
            if matches:
                print(f"    Side {si}: {len(matches)} matches (after error ratio filter)")
                for m in matches[:5]:
                    print(f"      -> pid={m['pid']}, si={m['si']}, error={m['error']:.4f}")
            else:
                print(f"    Side {si}: NO MATCHES after filter")
else:
    print("  File not found: connectivity_filtered.json")

print("\n[6] Checking texture_verify_report.json")
report_path = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')
if os.path.exists(report_path):
    with open(report_path) as f:
        report = json.load(f)

    for pid in TARGET_PIECES:
        pid_s = str(pid)
        print(f"\n  --- Piece {pid} texture verify report ---")
        if pid_s not in report:
            print(f"    NOT FOUND!")
            continue
        fits_list = report[pid_s]
        for si in range(4):
            matches = fits_list[si] if si < len(fits_list) else []
            if matches:
                print(f"    Side {si}: {len(matches)} entries")
                for m in matches[:5]:
                    print(f"      -> pid={m['pid']}, si={m['si']}, error={m['error']:.4f}, "
                          f"ncc={m['ncc']}, color_diff={m['color_diff']}, "
                          f"grad_score={m['grad_score']}, texture={m['texture_level']}, "
                          f"reject={m['reject']}, reason={m['reason']}")
            else:
                print(f"    Side {si}: NO entries in report")
else:
    print("  File not found: texture_verify_report.json")

print("\n[7] Detailed NCC analysis for 4<->137 pair")
from common.texture_verify import (
    load_side_data, load_color_image, extract_inner_band,
    compute_pattern_ncc, compute_seam_color_diff,
    compute_texture_richness, compute_gradient_consistency
)

for pid_a, pid_b in [(4, 137), (137, 4)]:
    print(f"\n  === NCC: Piece {pid_a} vs Piece {pid_b} ===")
    conn_a = connectivity.get(str(pid_a), [])
    for si in range(4):
        matches = conn_a[si] if si < len(conn_a) else []
        for m in matches:
            if m['pid'] == pid_b:
                sj = m['si']
                print(f"\n  Pair: {pid_a}[{si}] <-> {pid_b}[{sj}]")
                print(f"    Shape error: {m['error']:.4f}")

                side_a = load_side_data(DEDUPED_PATH, pid_a, si)
                side_b = load_side_data(DEDUPED_PATH, pid_b, sj)

                if side_a is None or side_b is None:
                    print("    Cannot load side data!")
                    continue

                print(f"    Side A: is_edge={side_a['is_edge']}, vertices={len(side_a['vertices'])}")
                print(f"    Side B: is_edge={side_b['is_edge']}, vertices={len(side_b['vertices'])}")

                color_a, mask_a = load_color_image(COLOR_PATH, pid_a)
                color_b, mask_b = load_color_image(COLOR_PATH, pid_b)

                if color_a is None:
                    print(f"    No color image for piece {pid_a}")
                    continue
                if color_b is None:
                    print(f"    No color image for piece {pid_b}")
                    continue

                print(f"    Color A: {color_a.shape}, Mask A: {mask_a.shape}")
                print(f"    Color B: {color_b.shape}, Mask B: {mask_b.shape}")

                band_a_colors, band_a_gray = extract_inner_band(
                    color_a, side_a['vertices'], side_a['piece_center'], mask_a
                )
                vertices_b_flipped = side_b['vertices'][::-1].copy()
                band_b_colors, band_b_gray = extract_inner_band(
                    color_b, vertices_b_flipped, side_b['piece_center'], mask_b
                )

                n = min(len(band_a_gray), len(band_b_gray))
                print(f"    Band A samples: {len(band_a_gray)}, Band B samples: {len(band_b_gray)}, common: {n}")

                if n < 5:
                    print(f"    Too few samples ({n}) for NCC!")
                    continue

                band_a_gray_c = band_a_gray[:n]
                band_b_gray_c = band_b_gray[:n]
                band_a_colors_c = band_a_colors[:n]
                band_b_colors_c = band_b_colors[:n]

                ncc = compute_pattern_ncc(band_a_gray_c, band_b_gray_c)
                color_diff_mean, color_diff_median = compute_seam_color_diff(band_a_colors_c, band_b_colors_c)
                tex_a = compute_texture_richness(band_a_gray_c)
                tex_b = compute_texture_richness(band_b_gray_c)
                grad_score = compute_gradient_consistency(band_a_gray_c, band_b_gray_c)

                print(f"    NCC: {ncc:.4f}")
                print(f"    Color diff: mean={color_diff_mean:.2f}, median={color_diff_median:.2f}")
                print(f"    Texture richness: A={tex_a:.4f}, B={tex_b:.4f}, min={min(tex_a, tex_b):.4f}")
                print(f"    Gradient consistency: {grad_score}")
                print(f"    Band A gray values: min={band_a_gray_c.min():.1f}, max={band_a_gray_c.max():.1f}, "
                      f"mean={band_a_gray_c.mean():.1f}, std={band_a_gray_c.std():.1f}")
                print(f"    Band B gray values: min={band_b_gray_c.min():.1f}, max={band_b_gray_c.max():.1f}, "
                      f"mean={band_b_gray_c.mean():.1f}, std={band_b_gray_c.std():.1f}")
                print(f"    Band A gray (first 10): {band_a_gray_c[:10].tolist()}")
                print(f"    Band B gray (first 10): {band_b_gray_c[:10].tolist()}")

                min_tex = min(tex_a, tex_b)
                if min_tex < 0.1:
                    reject = color_diff_mean > 80.0
                    print(f"    => LOW texture, reject by color_diff > 80: {reject}")
                else:
                    if grad_score is not None:
                        reject = (color_diff_mean > 60.0 and grad_score < 0.15)
                        print(f"    => RICH texture, reject by (color_diff>60 AND grad<0.15): {reject}")
                    else:
                        reject = color_diff_mean > 80.0
                        print(f"    => RICH texture (no grad), reject by color_diff > 80: {reject}")

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
