"""
Adaptive background segmentation for mobile phone puzzle photos

Provides three segmentation strategies:
  - Adaptive threshold (recommended default)
  - Otsu global threshold
  - GrabCut (fallback for complex backgrounds)

Also provides background brightness detection to choose the correct
polarity of the binary mask (pieces = 1, background = 0).
"""

import numpy as np
import cv2

from common.config import (
    PHONE_BLUR_KERNEL,
    PHONE_MORPH_KERNEL,
    PHONE_ADAPTIVE_BLOCK_SIZE,
    PHONE_ADAPTIVE_C,
    PHONE_BG_BRIGHTNESS_THRESHOLD,
    PHONE_BORDER_RATIO_FOR_BG_DETECT,
)


def detect_bg_brightness(gray):
    """
    Detect whether the background is light or dark by sampling the
    border region (outermost ratio of the image).
    Returns 'light' or 'dark'.
    """
    h, w = gray.shape
    ratio = PHONE_BORDER_RATIO_FOR_BG_DETECT
    border_h = max(int(h * ratio), 1)
    border_w = max(int(w * ratio), 1)
    top = gray[:border_h, :]
    bottom = gray[-border_h:, :]
    left = gray[:, :border_w]
    right = gray[:, -border_w:]
    border_pixels = np.concatenate([
        top.ravel(), bottom.ravel(), left.ravel(), right.ravel()
    ])
    median_val = np.median(border_pixels)
    return 'light' if median_val > PHONE_BG_BRIGHTNESS_THRESHOLD else 'dark'


def _adaptive_params(gray):
    """
    Compute adaptive threshold parameters based on image size.
    For smaller images, use a smaller block size to better separate pieces.
    """
    h, w = gray.shape
    max_dim = max(h, w)
    ref_dim = 3000
    scale = max_dim / ref_dim

    block_size = max(3, int(PHONE_ADAPTIVE_BLOCK_SIZE * scale))
    if block_size % 2 == 0:
        block_size += 1

    blur = max(3, int(PHONE_BLUR_KERNEL * scale))
    if blur % 2 == 0:
        blur += 1

    morph = max(3, int(PHONE_MORPH_KERNEL * scale))
    if morph % 2 == 0:
        morph += 1

    return block_size, blur, morph


def segment_adaptive(gray, bg_brightness=None):
    """
    Segment using OpenCV adaptive threshold.
    Returns binary mask where pieces = 1, background = 0.
    """
    if bg_brightness is None:
        bg_brightness = detect_bg_brightness(gray)

    block_size, blur_size, morph_size = _adaptive_params(gray)

    blurred = cv2.GaussianBlur(
        gray, (blur_size, blur_size), 0
    )

    if bg_brightness == 'light':
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=PHONE_ADAPTIVE_C
        )
    else:
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=PHONE_ADAPTIVE_C
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morph_size, morph_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return (binary > 127).astype(np.uint8)


def segment_otsu(gray, bg_brightness=None):
    """
    Segment using Otsu's global threshold.
    Returns binary mask where pieces = 1, background = 0.
    """
    if bg_brightness is None:
        bg_brightness = detect_bg_brightness(gray)

    _, blur_size, morph_size = _adaptive_params(gray)

    blurred = cv2.GaussianBlur(
        gray, (blur_size, blur_size), 0
    )

    if bg_brightness == 'light':
        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morph_size, morph_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return (binary > 127).astype(np.uint8)


def segment_grabcut(bgr, bg_brightness=None):
    """
    Segment using GrabCut algorithm (slower but handles complex backgrounds).
    Assumes pieces are roughly in the center region of the image.
    Returns binary mask where pieces = 1, background = 0.
    """
    h, w = bgr.shape[:2]
    margin = int(min(h, w) * 0.05)
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)
    return binary


def segment_photo(gray, bgr=None, method='adaptive'):
    """
    High-level segmentation entry point.

    method: 'adaptive', 'otsu', or 'grabcut'
    Returns binary mask (uint8, 0 or 1).
    """
    bg = detect_bg_brightness(gray)
    if method == 'adaptive':
        return segment_adaptive(gray, bg)
    elif method == 'otsu':
        return segment_otsu(gray, bg)
    elif method == 'grabcut':
        if bgr is None:
            raise ValueError("grabcut method requires bgr image")
        return segment_grabcut(bgr, bg)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def segment_with_fallback(gray, bgr=None, min_piece_pixels=100):
    """
    Try multiple segmentation methods and return the best result.
    Evaluates each result's quality and picks the one with the best score.
    Falls back from adaptive -> otsu -> grabcut.
    """
    results = []

    for method in ['adaptive', 'otsu']:
        try:
            binary = segment_photo(gray, bgr=bgr, method=method)
            n_pixels = np.sum(binary == 1)
            if n_pixels < min_piece_pixels:
                continue
            quality = evaluate_segmentation_quality(binary, gray)
            results.append((method, binary, n_pixels, quality))
        except Exception:
            continue

    if bgr is not None:
        try:
            binary = segment_grabcut(bgr)
            n_pixels = np.sum(binary == 1)
            if n_pixels >= min_piece_pixels:
                quality = evaluate_segmentation_quality(binary, gray)
                results.append(('grabcut', binary, n_pixels, quality))
        except Exception:
            pass

    if not results:
        return np.zeros_like(gray, dtype=np.uint8)

    results.sort(key=lambda x: x[3]['overall_score'], reverse=True)
    best = results[0]
    print(f"  Segmentation: method={best[0]}, pieces={best[3]['piece_count']}, "
          f"score={best[3]['overall_score']:.2f}, "
          f"issues={best[3]['issues']}")
    return best[1]


def evaluate_segmentation_quality(binary, gray, expected_piece_area_ratio=0.005):
    """
    Evaluate the quality of a segmentation result.

    Metrics:
      - piece_ratio: fraction of image that is piece pixels
      - piece_count_estimate: estimated number of connected components
      - edge_sharpness: how well-defined piece boundaries are
      - overall_score: combined quality score (0-1, higher is better)

    Args:
        binary: binary mask (uint8, 0 or 1)
        gray: original grayscale image
        expected_piece_area_ratio: expected area ratio per piece

    Returns:
        dict with quality metrics
    """
    h, w = binary.shape
    total_pixels = h * w
    piece_pixels = np.sum(binary == 1)
    piece_ratio = piece_pixels / total_pixels

    contours, _ = cv2.findContours(
        binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    piece_count = len(contours)

    if piece_count == 0:
        return {
            'piece_ratio': 0,
            'piece_count': 0,
            'edge_sharpness': 0,
            'overall_score': 0,
            'issues': ['no_pieces_found'],
        }

    areas = [cv2.contourArea(c) for c in contours]
    min_area = min(areas)
    max_area = max(areas)
    mean_area = np.mean(areas)
    std_area = np.std(areas) if len(areas) > 1 else 0
    area_cv = std_area / mean_area if mean_area > 0 else 0

    edge_region = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(binary * 255,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                         iterations=1)
    eroded = cv2.erode(binary * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                        iterations=1)
    boundary = dilated - eroded
    overlap = np.sum((boundary > 0) & (edge_region > 0))
    boundary_total = np.sum(boundary > 0)
    edge_sharpness = overlap / boundary_total if boundary_total > 0 else 0

    issues = []
    if piece_ratio < 0.05:
        issues.append('too_few_piece_pixels')
    elif piece_ratio > 0.95:
        issues.append('too_many_piece_pixels')

    if piece_count < 3:
        issues.append('too_few_pieces')
    elif piece_count > 500:
        issues.append('too_many_pieces')

    if area_cv > 2.0:
        issues.append('high_area_variance')

    if edge_sharpness < 0.1:
        issues.append('blurry_boundaries')

    score = 0.0
    score += min(piece_ratio * 2, 1.0) * 0.3
    score += min(edge_sharpness, 1.0) * 0.3
    if 0.1 < piece_ratio < 0.8:
        score += 0.2
    if area_cv < 1.5:
        score += 0.2

    return {
        'piece_ratio': piece_ratio,
        'piece_count': piece_count,
        'edge_sharpness': edge_sharpness,
        'overall_score': score,
        'issues': issues,
        'is_good': len(issues) == 0,
    }


def segment_three_tier(gray, bgr=None, min_piece_pixels=100,
                        min_quality_score=0.3):
    """
    Three-tier segmentation strategy with quality evaluation.

    Tier 1 (fast): Adaptive threshold - check quality
    Tier 2 (medium): Otsu global threshold - check quality
    Tier 3 (slow): GrabCut - best effort

    At each tier, the segmentation quality is evaluated. If the quality
    is acceptable, the result is returned immediately. Otherwise, the
    next tier is tried.

    Args:
        gray: grayscale image
        bgr: BGR color image (required for GrabCut tier)
        min_piece_pixels: minimum acceptable piece pixel count
        min_quality_score: minimum quality score to accept a result

    Returns:
        (binary_mask, method_used, quality_dict)
    """
    bg = detect_bg_brightness(gray)

    for method_name, seg_func in [
        ('adaptive', lambda: segment_adaptive(gray, bg)),
        ('otsu', lambda: segment_otsu(gray, bg)),
    ]:
        try:
            binary = seg_func()
            n_pixels = np.sum(binary == 1)

            if n_pixels < min_piece_pixels:
                continue

            quality = evaluate_segmentation_quality(binary, gray)

            if quality['overall_score'] >= min_quality_score and quality['is_good']:
                return binary, method_name, quality

        except Exception:
            continue

    if bgr is not None:
        try:
            binary = segment_grabcut(bgr)
            n_pixels = np.sum(binary == 1)

            if n_pixels >= min_piece_pixels:
                quality = evaluate_segmentation_quality(binary, gray)
                return binary, 'grabcut', quality

        except Exception:
            pass

    for method_name in ['adaptive', 'otsu']:
        try:
            binary = segment_photo(gray, bgr=bgr, method=method_name)
            n_pixels = np.sum(binary == 1)
            if n_pixels >= min_piece_pixels:
                quality = evaluate_segmentation_quality(binary, gray)
                return binary, method_name, quality
        except Exception:
            continue

    quality = {
        'piece_ratio': 0, 'piece_count': 0, 'edge_sharpness': 0,
        'overall_score': 0, 'issues': ['all_methods_failed'],
        'is_good': False,
    }
    return np.zeros_like(gray, dtype=np.uint8), 'none', quality
