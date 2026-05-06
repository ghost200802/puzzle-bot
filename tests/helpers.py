import math
import os

import numpy as np


def create_synthetic_piece_bmp(tmp_path, piece_id=1, size=500, half=100, tab_w=30, tab_h=40):
    from common.find_islands import save_island_as_bmp

    binary = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    binary[cy - half:cy + half, cx - half:cx + half] = 1

    for dy in range(-tab_h // 2, tab_h // 2 + 1):
        for dx in range(-tab_w // 2, tab_w // 2 + 1):
            if dx * dx + dy * dy <= (tab_h // 2) ** 2:
                ny, nx = cy - half - tab_h // 2 + dy, cx + dx
                if 0 <= ny < size and 0 <= nx < size:
                    binary[ny, nx] = 1
                ny, nx = cy + half + tab_h // 2 + dy, cx + dx
                if 0 <= ny < size and 0 <= nx < size:
                    binary[ny, nx] = 1

    for dy in range(-tab_w // 2, tab_w // 2 + 1):
        for dx in range(-tab_h // 2, tab_h // 2 + 1):
            if dx * dx + dy * dy <= (tab_h // 2) ** 2:
                ny, nx = cy + dy, cx + half + tab_h // 2 + dx
                if 0 <= ny < size and 0 <= nx < size:
                    binary[ny, nx] = 1
                ny, nx = cy + dy, cx - half - tab_h // 2 + dx
                if 0 <= ny < size and 0 <= nx < size:
                    binary[ny, nx] = 1

    path = str(tmp_path / f'piece_{piece_id}.bmp')
    save_island_as_bmp(binary, path)
    return path, binary


def match_corners_nearest_neighbor(detected, true, max_dist=50):
    results = []
    used = set()
    for dt in detected:
        best_d = 999
        best_j = -1
        for j, tc in enumerate(true):
            if j in used:
                continue
            d = math.sqrt((dt[0] - tc[0]) ** 2 + (dt[1] - tc[1]) ** 2)
            if d < best_d:
                best_d = d
                best_j = j
        if best_d < max_dist:
            results.append(f"{best_d:.0f}px")
            used.add(best_j)
        else:
            results.append(f"MISS({best_d:.0f})")
    passed = all("MISS" not in r for r in results) and all(
        int(r.replace('px', '')) < 25 for r in results
    )
    return results, passed


def print_corner_comparison_table(results):
    print(f"{'Piece':>8} | {'Detected':^50} | {'True':^50} | Match")
    print("-" * 140)
    all_pass = True
    for item in results:
        pid = item['piece_id']
        det = item['detected']
        true = item['true']
        matches = item['matches']
        passed = item['passed']
        if not passed:
            all_pass = False
        det_str = "  ".join([f"({c[0]},{c[1]})" for c in det])
        true_str = "  ".join([f"({c[0]},{c[1]})" for c in true])
        match_str = "  ".join(matches)
        status = "OK" if passed else "FAIL"
        print(f"  Piece_{pid:2d} | {det_str:50s} | {true_str:50s} | {match_str:30s} [{status}]")
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


def generate_corner_comparison_svg(vector_obj, true_corners, output_path, max_dist=30):
    import math

    p = vector_obj
    det_c = [(int(c[0]), int(c[1])) for c in p.corners]
    d = p.scalar / 2.0

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append(
        f'<svg width="{3 * p.width / d}" height="{3 * p.height / d}" '
        f'viewBox="-10 -10 {20 + p.width / d} {20 + p.height / d}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    )

    border_sampled = p.vertices[::3]
    border_pts = ' '.join([f'{float(v[0]) / d:.2f},{float(v[1]) / d:.2f}' for v in border_sampled])
    lines.append(f'<polyline points="{border_pts}" style="fill:none; stroke:#cccccc; stroke-width:1.5" />')

    for ci, tc_list in enumerate(true_corners):
        for tc in tc_list:
            tx, ty = float(tc[0]) / d, float(tc[1]) / d
            lines.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="5" style="fill:none; stroke:#00cc00; stroke-width:2" />')

    for ci, dc in enumerate(det_c):
        dx, dy = float(dc[0]) / d, float(dc[1]) / d
        lines.append(f'<circle cx="{dx:.2f}" cy="{dy:.2f}" r="3" style="fill:#ff4444; stroke:#000; stroke-width:0.5" />')
        lines.append(f'<text x="{dx + 6:.2f}" y="{dy - 4:.2f}" font-size="9" font-family="monospace" fill="#ff4444">C{ci}</text>')

    lines.append('<text x="10" y="20" font-size="10" font-family="monospace" fill="#00cc00">O circle = true corners</text>')
    lines.append('<text x="10" y="34" font-size="10" font-family="monospace" fill="#ff4444">Red dot = detected corners</text>')

    results = []
    all_ok = True
    for dc in det_c:
        best_d = 999
        best_ci = -1
        for ci, tc_list in enumerate(true_corners):
            for tc in tc_list:
                dist = math.sqrt((dc[0] - tc[0]) ** 2 + (dc[1] - tc[1]) ** 2)
                if dist < best_d:
                    best_d = dist
                    best_ci = ci
        ok = best_d < max_dist
        if not ok:
            all_ok = False
        results.append(f"C{best_ci}(d={best_d:.0f})" if ok else f"MISS(d={best_d:.0f})")

    status = "OK" if all_ok else "FAIL"
    pid = p.id if hasattr(p, 'id') and p.id else "?"
    lines.append(f'<text x="10" y="48" font-size="10" font-family="monospace" fill="#000">Piece_{pid}: {status} {" ".join(results)}</text>')
    lines.append('</svg>')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return status, det_c, results


def generate_piece_outline_svg(vector_obj, output_path, piece_label=None):
    p = vector_obj
    corners = p.corners
    pid = piece_label or (p.id if hasattr(p, 'id') and p.id else "?")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000">\n')
        f.write('<rect width="1000" height="1000" fill="#1a1a2e"/>\n')
        pts = " ".join([f"{v[0]},{v[1]}" for v in p.vertices])
        f.write(f'<polygon points="{pts}" fill="#2d2d44" stroke="#888" stroke-width="1"/>\n')
        colors = ['#ff3333', '#33ff33', '#3399ff', '#ffff33']
        labels = ['C0', 'C1', 'C2', 'C3']
        for ci, (cx, cy) in enumerate(corners):
            cx, cy = int(cx), int(cy)
            f.write(f'<circle cx="{cx}" cy="{cy}" r="8" fill="{colors[ci]}" stroke="white" stroke-width="2"/>\n')
            f.write(f'<text x="{cx + 12}" y="{cy - 8}" fill="{colors[ci]}" font-size="16" font-family="monospace">{labels[ci]}({cx},{cy})</text>\n')
        f.write(f'<text x="10" y="25" fill="white" font-size="18" font-family="monospace">Piece {pid}</text>\n')
        f.write('</svg>\n')


def print_vector_details(vector_obj, label=None):
    v = vector_obj
    prefix = f"\n[{label}] " if label else "\n"
    print(f"{prefix}Vector: dim={v.dim}, scalar={v.scalar:.2f}")
    print(f"  Total vertices: {len(v.vertices)}, Centroid: {v.centroid}")

    print(f"  Selected corners:")
    for i, corner in enumerate(v.corners):
        print(f"    Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f})")

    print(f"  Corner indices: {v.corner_indices}")

    try:
        v.extract_four_sides()
        print(f"  Extracted sides:")
        for i, side in enumerate(v.sides):
            n_verts = len(side.vertices)
            first = side.vertices[0]
            last = side.vertices[-1]
            angle_deg = round(side.angle * 180 / 3.14159, 1) if hasattr(side, 'angle') and side.angle is not None else 'N/A'
            print(
                f"    Side {i}: {n_verts} vertices, "
                f"({first[0]:.1f},{first[1]:.1f}) -> ({last[0]:.1f},{last[1]:.1f}), "
                f"is_edge={side.is_edge}, angle={angle_deg}°"
            )
    except Exception as e:
        print(f"  extract_four_sides error: {e}")
