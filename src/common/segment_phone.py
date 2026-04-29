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


def segment_adaptive(gray, bg_brightness=None):
    """
    Segment using OpenCV adaptive threshold.
    Returns binary mask where pieces = 1, background = 0.
    """
    if bg_brightness is None:
        bg_brightness = detect_bg_brightness(gray)

    blurred = cv2.GaussianBlur(
        gray, (PHONE_BLUR_KERNEL, PHONE_BLUR_KERNEL), 0
    )

    if bg_brightness == 'light':
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=PHONE_ADAPTIVE_BLOCK_SIZE,
            C=PHONE_ADAPTIVE_C
        )
    else:
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=PHONE_ADAPTIVE_BLOCK_SIZE,
            C=PHONE_ADAPTIVE_C
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (PHONE_MORPH_KERNEL, PHONE_MORPH_KERNEL)
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

    blurred = cv2.GaussianBlur(
        gray, (PHONE_BLUR_KERNEL, PHONE_BLUR_KERNEL), 0
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
        (PHONE_MORPH_KERNEL, PHONE_MORPH_KERNEL)
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
    Falls back from adaptive -> otsu -> grabcut.
    Selects the result with the most reasonable number of piece pixels.
    """
    results = []

    for method in ['adaptive', 'otsu']:
        try:
            binary = segment_photo(gray, bgr=bgr, method=method)
            n_pixels = np.sum(binary == 1)
            if n_pixels >= min_piece_pixels:
                results.append((method, binary, n_pixels))
        except Exception:
            continue

    if bgr is not None and not results:
        try:
            binary = segment_grabcut(bgr)
            n_pixels = np.sum(binary == 1)
            if n_pixels >= min_piece_pixels:
                results.append(('grabcut', binary, n_pixels))
        except Exception:
            pass

    if not results:
        # Return empty mask as last resort
        return np.zeros_like(gray, dtype=np.uint8)

    results.sort(key=lambda x: x[2], reverse=True)
    return results[0][1]
