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


def normalize_piece_size(binary, color, target_size=500):
    """
    Crop to bounding box and scale a piece so its larger dimension equals target_size.
    Returns (scaled_binary, scaled_color, scale_factor).
    """
    ys, xs = np.where(binary == 1)
    if len(xs) == 0:
        return binary, color, 1.0
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    binary_crop = binary[min_y:max_y + 1, min_x:max_x + 1].copy()
    if color.shape[:2] == binary.shape:
        color_crop = color[min_y:max_y + 1, min_x:max_x + 1].copy()
    else:
        color_crop = color.copy()
    h, w = binary_crop.shape
    max_dim = max(w, h)
    if max_dim == 0:
        return binary_crop, color_crop, 1.0
    scale = target_size / max_dim
    if abs(scale - 1.0) < 0.01:
        return binary_crop, color_crop, 1.0
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    scaled_binary = cv2.resize(binary_crop.astype(np.uint8) * 255, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
    scaled_binary = (scaled_binary > 127).astype(np.uint8)
    scaled_color = cv2.resize(color_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return scaled_binary, scaled_color, scale


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
