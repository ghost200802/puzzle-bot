"""
Photo preprocessing for mobile phone puzzle photos

Handles:
  - EXIF auto-rotation
  - Color normalization (white balance, CLAHE)
  - Piece size normalization
  - Image quality detection and enhancement
"""

import numpy as np
import cv2
from PIL import Image, ImageOps


def auto_rotate(image_path):
    """
    Read an image and apply EXIF rotation automatically.
    Returns a numpy BGR image (OpenCV format).
    """
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def normalize_color(image):
    """
    Normalize color using Gray World white balance + optional CLAHE.
    Returns the normalized BGR image.
    """
    result = gray_world_white_balance(image)
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def gray_world_white_balance(image):
    """
    Gray World white balance: scale each channel so their means are equal.
    """
    result = image.astype(np.float64)
    avg_b = result[:, :, 0].mean()
    avg_g = result[:, :, 1].mean()
    avg_r = result[:, :, 2].mean()
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    if avg_gray < 1e-6:
        return image.copy()
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return result.astype(np.uint8)


def preprocess_phone_photo(photo_path):
    """
    Full preprocessing pipeline for a phone photo.
    Returns (bgr_image, gray_image).
    """
    bgr = auto_rotate(photo_path)
    bgr = normalize_color(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def normalize_piece_size(binary, gray_full, origin, color=None, target_size=500):
    """
    Crop from original grayscale image at origin, scale up with INTER_CUBIC,
    then Gaussian blur + Otsu re-thresholding for clean smooth edges.
    This matches run_pipeline.py's high-quality two-pass approach.

    Args:
        binary: uint8 0/1 mask of the piece (from segmentation)
        gray_full: full-resolution grayscale image (original, unmasked)
        origin: (col, row) top-left of the island in the full image
        color: optional BGR crop for color rescaling (same shape as binary)
        target_size: target max dimension after scaling

    Returns: (scaled_binary, scaled_color_or_None, scale_factor)
    """
    from scipy.ndimage import label as ndlabel

    ys, xs = np.where(binary == 1)
    if len(xs) == 0:
        return binary, color, 1.0

    local_min_x, local_max_x = int(xs.min()), int(xs.max())
    local_min_y, local_max_y = int(ys.min()), int(ys.max())
    origin_col, origin_row = origin

    bh, bw = binary.shape
    img_x0 = origin_col + local_min_x
    img_y0 = origin_row + local_min_y
    img_x1 = origin_col + local_max_x + 1
    img_y1 = origin_row + local_max_y + 1

    piece_w = local_max_x - local_min_x + 1
    piece_h = local_max_y - local_min_y + 1
    pad = max(3, int(max(piece_w, piece_h) * 0.05))

    img_x0 = max(0, img_x0 - pad)
    img_y0 = max(0, img_y0 - pad)
    img_x1 = min(gray_full.shape[1], img_x1 + pad)
    img_y1 = min(gray_full.shape[0], img_y1 + pad)

    orig_crop = gray_full[img_y0:img_y1, img_x0:img_x1]
    crop_h, crop_w = orig_crop.shape
    max_dim = max(crop_w, crop_h)
    if max_dim == 0:
        return binary, color, 1.0

    scale = target_size / max_dim
    if abs(scale - 1.0) < 0.01:
        _, smooth_binary = cv2.threshold(
            orig_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        smooth_binary = (smooth_binary > 127).astype(np.uint8)
        return smooth_binary, color, 1.0

    new_w = max(int(crop_w * scale), 10)
    new_h = max(int(crop_h * scale), 10)

    scaled = cv2.resize(orig_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    blur_k = max(3, int(scale * 0.8))
    if blur_k % 2 == 0:
        blur_k += 1
    blurred = cv2.GaussianBlur(scaled, (blur_k, blur_k), 0)

    _, smooth_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    smooth_binary = (smooth_binary > 127).astype(np.uint8)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smooth_binary = cv2.morphologyEx(smooth_binary, cv2.MORPH_OPEN, kern)
    smooth_binary = cv2.morphologyEx(smooth_binary, cv2.MORPH_CLOSE, kern)

    lbl, n = ndlabel(smooth_binary)
    if n > 1:
        best = max(range(1, n + 1), key=lambda j: np.sum(lbl == j))
        smooth_binary = (lbl == best).astype(np.uint8)

    scaled_color = None
    if color is not None:
        scaled_color = cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return smooth_binary, scaled_color, scale


def normalize_piece_color(color_image, mask):
    """
    Normalize color of piece pixels (mask==1) only.
    """
    piece_pixels = color_image[mask == 1].astype(np.float64)
    if len(piece_pixels) == 0:
        return color_image.copy()
    avg_b = piece_pixels[:, 0].mean()
    avg_g = piece_pixels[:, 1].mean()
    avg_r = piece_pixels[:, 2].mean()
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    if avg_gray < 1e-6:
        return color_image.copy()
    result = color_image.copy().astype(np.float64)
    mask_bool = mask == 1
    result[mask_bool, 0] = np.clip(result[mask_bool, 0] * (avg_gray / max(avg_b, 1e-6)), 0, 255)
    result[mask_bool, 1] = np.clip(result[mask_bool, 1] * (avg_gray / max(avg_g, 1e-6)), 0, 255)
    result[mask_bool, 2] = np.clip(result[mask_bool, 2] * (avg_gray / max(avg_r, 1e-6)), 0, 255)
    return result.astype(np.uint8)


def auto_resize_for_processing(image, max_dimension=2000):
    """
    Resize image if it's too large for processing.
    Returns (resized_image, scale_factor).
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_dimension:
        return image, 1.0
    scale = max_dimension / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_image_quality(gray):
    """
    Detect image quality issues.
    Returns dict with quality metrics.
    """
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = max(0, min(100, laplacian_var / 10))
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    contrast_score = min(100, std_brightness / 1.28)

    issues = []
    if blur_score < 10:
        issues.append('blurry')
    if mean_brightness < 30:
        issues.append('too_dark')
    elif mean_brightness > 225:
        issues.append('too_bright')
    if contrast_score < 15:
        issues.append('low_contrast')

    return {
        'blur_score': blur_score,
        'brightness': mean_brightness,
        'contrast_score': contrast_score,
        'issues': issues,
        'is_good': len(issues) == 0,
    }


def enhance_image(gray, bgr=None):
    """
    Apply image enhancement based on detected quality issues.
    Returns (enhanced_gray, enhanced_bgr_or_None).
    """
    quality = detect_image_quality(gray)

    enhanced_gray = gray.copy()
    enhanced_bgr = bgr.copy() if bgr is not None else None

    if 'low_contrast' in quality['issues']:
        enhanced_gray = cv2.equalizeHist(enhanced_gray)
        if enhanced_bgr is not None:
            for i in range(3):
                enhanced_bgr[:, :, i] = cv2.equalizeHist(enhanced_bgr[:, :, i])

    if 'too_dark' in quality['issues']:
        enhanced_gray = np.clip(
            enhanced_gray.astype(np.float32) * 1.5, 0, 255
        ).astype(np.uint8)
        if enhanced_bgr is not None:
            enhanced_bgr = np.clip(
                enhanced_bgr.astype(np.float32) * 1.5, 0, 255
            ).astype(np.uint8)

    if 'too_bright' in quality['issues']:
        enhanced_gray = np.clip(
            enhanced_gray.astype(np.float32) * 0.7, 0, 255
        ).astype(np.uint8)
        if enhanced_bgr is not None:
            enhanced_bgr = np.clip(
                enhanced_bgr.astype(np.float32) * 0.7, 0, 255
            ).astype(np.uint8)

    return enhanced_gray, enhanced_bgr


def validate_piece(piece_binary, min_pixels=50, max_pixels=None):
    """
    Validate a piece binary mask.
    Returns (is_valid, reason).
    """
    pixel_count = np.sum(piece_binary == 1)
    if pixel_count < min_pixels:
        return False, f'too_few_pixels ({pixel_count} < {min_pixels})'
    if max_pixels and pixel_count > max_pixels:
        return False, f'too_many_pixels ({pixel_count} > {max_pixels})'

    ys, xs = np.where(piece_binary == 1)
    if len(xs) == 0:
        return False, 'empty'
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    aspect_ratio = max(w, h) / max(min(w, h), 1)
    if aspect_ratio > 10:
        return False, f'bad_aspect_ratio ({aspect_ratio:.1f})'

    return True, 'ok'


def detect_document_contour(image, edge_area_ratio=0.85):
    """
    Detect the four corners of the main document/puzzle region in an image.
    Uses edge detection + contour finding + convex hull approximation.

    Args:
        image: BGR input image
        edge_area_ratio: minimum area ratio for accepting a contour

    Returns:
        List of 4 corner points [[x, y], ...] in order: TL, TR, BR, BL,
        or None if no suitable contour found.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    min_area = w * h * edge_area_ratio * 0.3
    best_contour = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)

        if len(approx) == 4:
            score = area
            if score > best_score:
                best_score = score
                best_contour = approx
        elif len(approx) > 4:
            hull_approx = cv2.convexHull(approx)
            perimeter2 = cv2.arcLength(hull_approx, True)
            approx2 = cv2.approxPolyDP(hull_approx, 0.02 * perimeter2, True)
            if len(approx2) == 4:
                score = area
                if score > best_score:
                    best_score = score
                    best_contour = approx2

    if best_contour is None:
        return None

    pts = best_contour.reshape(4, 2)

    def _dist(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    center = pts.mean(axis=0)
    top_points = pts[pts[:, 1] < center[1]]
    bottom_points = pts[pts[:, 1] >= center[1]]

    if len(top_points) == 2 and len(bottom_points) == 2:
        tl = top_points[top_points[:, 0] == top_points[:, 0].min()][0]
        tr = top_points[top_points[:, 0] == top_points[:, 0].max()][0]
        bl = bottom_points[bottom_points[:, 0] == bottom_points[:, 0].min()][0]
        br = bottom_points[bottom_points[:, 0] == bottom_points[:, 0].max()][0]
    else:
        dists = [_dist(p, [0, 0]) for p in pts]
        order = np.argsort(dists)
        tl = pts[order[0]]
        br = pts[order[-1]]
        remaining = [pts[i] for i in range(4) if i not in [order[0], order[-1]]]
        if remaining[0][0] < remaining[1][0]:
            bl, tr = remaining
        else:
            tr, bl = remaining

    return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]


def perspective_correct(image, corners=None, output_size=None):
    """
    Apply perspective correction to an image.

    Args:
        image: BGR input image
        corners: Optional list of 4 corners [[x, y], ...] in TL, TR, BR, BL order.
                 If None, automatically detects the document contour.
        output_size: Optional (width, height) for the output. If None, uses
                     the maximum width and height from the detected corners.

    Returns:
        (corrected_bgr, transform_matrix) or (None, None) if correction failed.
    """
    if corners is None:
        corners = detect_document_contour(image)
        if corners is None:
            return None, None

    corners = np.array(corners, dtype=np.float32)

    if output_size is None:
        top_width = np.sqrt(
            (corners[1][0] - corners[0][0]) ** 2 +
            (corners[1][1] - corners[0][1]) ** 2
        )
        bottom_width = np.sqrt(
            (corners[2][0] - corners[3][0]) ** 2 +
            (corners[2][1] - corners[3][1]) ** 2
        )
        left_height = np.sqrt(
            (corners[3][0] - corners[0][0]) ** 2 +
            (corners[3][1] - corners[0][1]) ** 2
        )
        right_height = np.sqrt(
            (corners[2][0] - corners[1][0]) ** 2 +
            (corners[2][1] - corners[1][1]) ** 2
        )
        out_w = int(max(top_width, bottom_width))
        out_h = int(max(left_height, right_height))
    else:
        out_w, out_h = output_size

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    corrected = cv2.warpPerspective(image, M, (out_w, out_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

    return corrected, M


def auto_crop_to_content(image, border_ratio=0.02):
    """
    Automatically crop image to the main content area.
    Removes uniform borders.

    Args:
        image: BGR input image
        border_ratio: ratio of border to keep (0 = tight crop)

    Returns:
        Cropped BGR image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    margin_x = int(w * border_ratio)
    margin_y = int(h * border_ratio)

    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(image.shape[1] - x, w + 2 * margin_x)
    h = min(image.shape[0] - y, h + 2 * margin_y)

    return image[y:y + h, x:x + w].copy()
