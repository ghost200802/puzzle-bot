import os
import json
import math
import numpy as np
import cv2

INNER_OFFSET = 6
BAND_WIDTH = 15
N_SAMPLES = 30

GRADIENT_SIGNIFICANCE_THRESHOLD = 8
TEXTURE_LOW_THRESHOLD = 0.1

COLOR_DIFF_REJECT_STRICT = 60.0
COLOR_DIFF_REJECT_LOOSE = 80.0

GRAD_REJECT_THRESHOLD = 0.15


def load_side_data(deduped_dir, piece_id, side_index):
    path = os.path.join(deduped_dir, f"side_{piece_id}_{side_index}.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return {
        'vertices': np.array(data['vertices'], dtype=np.float64),
        'piece_center': np.array(data['piece_center'], dtype=np.float64),
        'is_edge': data.get('is_edge', False),
    }


def load_color_image(color_dir, piece_id):
    path_bgr = os.path.join(color_dir, f"color_{piece_id}.png")
    if os.path.exists(path_bgr):
        img = cv2.imread(path_bgr, cv2.IMREAD_COLOR)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            return img, mask.astype(np.uint8)

    path_rgba = os.path.join(color_dir, f"piece_{piece_id}.png")
    if os.path.exists(path_rgba):
        img_rgba = cv2.imread(path_rgba, cv2.IMREAD_UNCHANGED)
        if img_rgba is not None and img_rgba.shape[2] == 4:
            alpha = img_rgba[:, :, 3]
            mask = (alpha > 128).astype(np.uint8) * 255
            img_bgr = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            return img_bgr, mask

    return None, None


def _resample_polyline(vertices, n):
    diffs = np.diff(vertices, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]
    if total_length < 1e-6:
        return vertices[:1].copy()
    sample_dists = np.linspace(0, total_length, n)
    resampled = np.zeros((n, 2))
    for i, d in enumerate(sample_dists):
        idx = np.searchsorted(cum_lengths, d, side='right') - 1
        idx = max(0, min(idx, len(seg_lengths) - 1))
        seg_start = cum_lengths[idx]
        seg_len = seg_lengths[idx]
        t = (d - seg_start) / seg_len if seg_len > 1e-6 else 0.0
        resampled[i] = vertices[idx] * (1 - t) + vertices[idx + 1] * t
    return resampled


def extract_inner_band(color_image, side_vertices, piece_center, binary_mask,
                       inner_offset=INNER_OFFSET, band_width=BAND_WIDTH,
                       n_samples=N_SAMPLES):
    h, w = color_image.shape[:2]
    resampled = _resample_polyline(side_vertices, n_samples)

    band_colors = []
    band_gray_values = []

    for i in range(n_samples):
        if i == 0:
            tangent = resampled[1] - resampled[0]
        elif i == n_samples - 1:
            tangent = resampled[-1] - resampled[-2]
        else:
            tangent = resampled[i + 1] - resampled[i - 1]

        tlen = np.sqrt(tangent[0] ** 2 + tangent[1] ** 2)
        if tlen < 1e-6:
            continue
        tangent = tangent / tlen

        normal = np.array([-tangent[1], tangent[0]])

        to_center = piece_center - resampled[i]
        if np.dot(normal, to_center) < 0:
            normal = -normal

        colors = []
        for d in range(inner_offset, inner_offset + band_width):
            pt = resampled[i] + normal * d
            px, py = int(round(pt[0])), int(round(pt[1]))
            if 0 <= py < h and 0 <= px < w and binary_mask[py, px] > 0:
                colors.append(color_image[py, px].astype(np.float64))

        if colors:
            avg_color = np.mean(colors, axis=0)
            gray = 0.114 * avg_color[0] + 0.587 * avg_color[1] + 0.299 * avg_color[2]
            band_colors.append(avg_color)
            band_gray_values.append(gray)

    if not band_colors:
        return np.array([]).reshape(0, 3), np.array([])

    return np.array(band_colors), np.array(band_gray_values)


def compute_texture_richness(band_gray):
    if len(band_gray) < 3:
        return 0.0
    diffs = np.abs(np.diff(band_gray))
    significant = np.sum(diffs > GRADIENT_SIGNIFICANCE_THRESHOLD)
    ratio = significant / (len(band_gray) - 1)
    return float(ratio)


def compute_seam_color_diff(band_a_colors, band_b_colors):
    if len(band_a_colors) == 0 or len(band_b_colors) == 0:
        return 999.0, 999.0

    a_bgr = band_a_colors.reshape(1, -1, 3).astype(np.uint8)
    b_bgr = band_b_colors.reshape(1, -1, 3).astype(np.uint8)

    a_lab = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
    b_lab = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)

    a_lab = a_lab.reshape(-1, 3)
    b_lab = b_lab.reshape(-1, 3)

    delta_e = np.sqrt(np.sum((a_lab - b_lab) ** 2, axis=1))
    return float(np.mean(delta_e)), float(np.median(delta_e))


def compute_pattern_ncc(band_a_gray, band_b_gray, max_shift=5):
    n = min(len(band_a_gray), len(band_b_gray))
    if n < max_shift * 2 + 3:
        return 0.0

    a = band_a_gray[:n].astype(np.float64)
    b = band_b_gray[:n].astype(np.float64)

    best_ncc = -2.0
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a_sub = a[s:]
            b_sub = b[:n - s]
        else:
            a_sub = a[:n + s]
            b_sub = b[-s:]

        m = len(a_sub)
        if m < 5:
            continue

        a_mean = np.mean(a_sub)
        b_mean = np.mean(b_sub)
        a_c = a_sub - a_mean
        b_c = b_sub - b_mean
        a_norm = np.sqrt(np.sum(a_c ** 2))
        b_norm = np.sqrt(np.sum(b_c ** 2))

        if a_norm < 1e-6 or b_norm < 1e-6:
            continue

        ncc = float(np.dot(a_c, b_c) / (a_norm * b_norm))
        if ncc > best_ncc:
            best_ncc = ncc

    return max(best_ncc, -1.0)


def compute_gradient_consistency(band_a_gray, band_b_gray):
    if len(band_a_gray) < 3 or len(band_b_gray) < 3:
        return None

    grad_a = np.diff(band_a_gray)
    grad_b = np.diff(band_b_gray)

    n = min(len(grad_a), len(grad_b))
    grad_a = grad_a[:n]
    grad_b = grad_b[:n]

    sign_agree = (np.sign(grad_a) == np.sign(grad_b)).astype(np.float64)
    magnitude = np.maximum(np.abs(grad_a), np.abs(grad_b))
    total_mag = np.sum(magnitude)

    if total_mag < 1e-6:
        return None

    return float(np.sum(sign_agree * magnitude) / total_mag)


def verify_match(color_dir, deduped_dir, pid_a, si_a, pid_b, si_b, shift=None):
    result_template = {
        'reject': False,
        'color_diff_mean': 0.0,
        'color_diff_median': 0.0,
        'ncc': 0.0,
        'grad_score': None,
        'texture_a': 0.0,
        'texture_b': 0.0,
        'texture_level': 'unknown',
        'reason': 'ok',
        'n_samples': 0,
    }

    side_a = load_side_data(deduped_dir, pid_a, si_a)
    side_b = load_side_data(deduped_dir, pid_b, si_b)

    if side_a is None or side_b is None:
        result_template['reason'] = 'no_side_data'
        return result_template

    if side_a['is_edge'] or side_b['is_edge']:
        result_template['reason'] = 'is_edge'
        return result_template

    color_a, mask_a = load_color_image(color_dir, pid_a)
    color_b, mask_b = load_color_image(color_dir, pid_b)

    if color_a is None or color_b is None:
        result_template['reason'] = 'no_color_data'
        return result_template

    band_a_colors, band_a_gray = extract_inner_band(
        color_a, side_a['vertices'], side_a['piece_center'], mask_a
    )

    vertices_b_flipped = side_b['vertices'][::-1].copy()
    band_b_colors, band_b_gray = extract_inner_band(
        color_b, vertices_b_flipped, side_b['piece_center'], mask_b
    )

    n = min(len(band_a_colors), len(band_b_colors))
    if n < 5:
        result_template['reason'] = 'too_few_samples'
        result_template['n_samples'] = n
        return result_template

    band_a_colors = band_a_colors[:n]
    band_b_colors = band_b_colors[:n]
    band_a_gray = band_a_gray[:n]
    band_b_gray = band_b_gray[:n]

    color_diff_mean, color_diff_median = compute_seam_color_diff(band_a_colors, band_b_colors)
    ncc = compute_pattern_ncc(band_a_gray, band_b_gray)

    tex_a = compute_texture_richness(band_a_gray)
    tex_b = compute_texture_richness(band_b_gray)
    min_tex = min(tex_a, tex_b)

    result_template['color_diff_mean'] = color_diff_mean
    result_template['color_diff_median'] = color_diff_median
    result_template['ncc'] = round(ncc, 4)
    result_template['texture_a'] = round(tex_a, 4)
    result_template['texture_b'] = round(tex_b, 4)
    result_template['n_samples'] = n

    if min_tex < TEXTURE_LOW_THRESHOLD:
        result_template['texture_level'] = 'low'
        reject = color_diff_mean > COLOR_DIFF_REJECT_LOOSE
        result_template['reject'] = reject
        result_template['reason'] = 'color_reject' if reject else 'ok'
        return result_template

    result_template['texture_level'] = 'rich'
    grad_score = compute_gradient_consistency(band_a_gray, band_b_gray)

    if grad_score is None:
        reject = color_diff_mean > COLOR_DIFF_REJECT_LOOSE
        result_template['reject'] = reject
        result_template['reason'] = 'color_reject' if reject else 'ok'
        return result_template

    result_template['grad_score'] = round(grad_score, 4)

    reject = (color_diff_mean > COLOR_DIFF_REJECT_STRICT
              and grad_score < GRAD_REJECT_THRESHOLD)
    result_template['reject'] = reject
    result_template['reason'] = 'color_gradient_reject' if reject else 'ok'
    return result_template
