"""
Image matching utilities for puzzle pieces.

Provides:
  - Color histogram comparison
  - Color continuity scoring between adjacent edges
  - ORB feature-based matching
  - Combined geometric + image matching score
  - Piece-to-target-grid matching
"""

import numpy as np
import cv2


def compute_color_histogram(image_bgr, mask=None):
    """
    Compute HSV color histogram for an image.
    Returns dict with 'h', 's', 'v' channels.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if mask is None:
        mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255
    elif mask.dtype != np.uint8 or mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    hist = {
        'h': cv2.calcHist([hsv], [0], mask, [36], [0, 180]),
        's': cv2.calcHist([hsv], [1], mask, [32], [0, 256]),
        'v': cv2.calcHist([hsv], [2], mask, [32], [0, 256]),
    }
    return hist


def histogram_similarity(hist_a, hist_b):
    """
    Compare two color histograms using correlation.
    Returns a score from -1 to 1 (higher = more similar).
    """
    score = 0.0
    count = 0
    for channel in ['h', 's', 'v']:
        ha = hist_a.get(channel)
        hb = hist_b.get(channel)
        if ha is not None and hb is not None:
            ha_f = ha.astype(np.float32)
            hb_f = hb.astype(np.float32)
            s = cv2.compareHist(ha_f, hb_f, cv2.HISTCMP_CORREL)
            score += s
            count += 1
    if count == 0:
        return 0.0
    return score / count


def color_continuity_score(band_a, band_b):
    """
    Evaluate color continuity between two edge color bands.
    Returns a score from 0 to 2 (1.0 = perfect continuity).
    """
    if band_a is None or band_b is None:
        return 1.0
    a = np.array(band_a, dtype=np.float64)
    b = np.array(band_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 1.0
    min_len = min(len(a), len(b))
    a = a[:min_len].flatten()
    b = b[:min_len].flatten()
    ssd = np.sum((a - b) ** 2) / max(a.size, 1)
    score = 1.0 / (1.0 + ssd / 1000.0)
    return score


def extract_color_band_along_side(color_image, side_vertices, band_width=15):
    """
    Extract a color band along a side's vertices.
    Returns a list of average colors along the side.
    """
    if color_image is None or len(side_vertices) < 2:
        return None
    h, w = color_image.shape[:2]
    band = []
    n_samples = min(len(side_vertices), 50)
    step = max(1, len(side_vertices) // n_samples)
    for i in range(0, len(side_vertices), step):
        vx, vy = side_vertices[i]
        vx, vy = int(vx), int(vy)
        colors = []
        for dx in range(-band_width // 2, band_width // 2 + 1):
            nx, ny = vx + dx, vy
            if 0 <= ny < h and 0 <= nx < w:
                colors.append(color_image[ny, nx].tolist())
        if colors:
            band.append(np.mean(colors, axis=0).tolist())
    return band


def compute_color_similarity(hist_a, hist_b):
    """
    Compute overall color similarity between two histograms.
    Returns a score from 0 to 1.
    """
    sim = histogram_similarity(hist_a, hist_b)
    return max(0.0, sim)


class ORBMatcher:
    """
    ORB feature-based matcher for puzzle piece images.
    """

    def __init__(self, max_features=500):
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_compute(self, image):
        """
        Detect ORB keypoints and compute descriptors.
        Returns (keypoints, descriptors).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def match(self, des_a, des_b, ratio_threshold=0.75):
        """
        Match descriptors between two images.
        Returns list of good matches.
        """
        if des_a is None or des_b is None:
            return []
        if len(des_a) < 2 or len(des_b) < 2:
            return []
        matches = self.bf.match(des_a, des_b)
        matches = sorted(matches, key=lambda x: x.distance)
        good = [m for m in matches if m.distance < 50]
        return good

    def compute_match_score(self, image_a, image_b):
        """
        Compute a match score between two images using ORB features.
        Returns a score from 0 to 1 (higher = more similar).
        """
        _, des_a = self.detect_and_compute(image_a)
        _, des_b = self.detect_and_compute(image_b)
        matches = self.match(des_a, des_b)
        if not matches:
            return 0.0
        avg_distance = np.mean([m.distance for m in matches])
        max_possible = 256
        score = max(0.0, 1.0 - avg_distance / max_possible)
        n_features = max(len(des_a) if des_a is not None else 0,
                         len(des_b) if des_b is not None else 0, 1)
        coverage = min(len(matches) / max(n_features, 1), 1.0)
        return score * 0.6 + coverage * 0.4


def compute_combined_match_score(geometric_error, image_score, color_score=1.0,
                                 geo_weight=0.5, img_weight=0.3, color_weight=0.2):
    """
    Compute a combined match score from geometric, image, and color components.

    geometric_error: lower is better (from Side.error_when_fit_with)
    image_score: 0-1, higher is better (from ORB matching)
    color_score: 0-1, higher is better (from histogram comparison)

    Returns: 0-1 combined score, higher = better match.
    """
    geo_score = max(0.0, 1.0 - geometric_error / 10.0)
    combined = geo_weight * geo_score + img_weight * image_score + color_weight * color_score
    return combined


def match_piece_to_target(piece_features, target_cells, top_n=5):
    """
    Match a piece to the target grid.
    Returns list of (row, col, score) sorted by score descending.
    """
    from common.target import compute_target_match_score
    scores = []
    for (row, col), cell in target_cells.items():
        score = compute_target_match_score(piece_features, cell)
        scores.append((row, col, score))
    scores.sort(key=lambda x: -x[2])
    return scores[:top_n]
