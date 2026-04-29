"""
Target image (box photo) processing for puzzle solving.

Handles:
  - Loading and preprocessing target image
  - Grid segmentation into W x H cells
  - Color/texture feature extraction per cell
  - Piece-to-target matching
"""

import numpy as np
import cv2


class TargetImage:
    """
    Target image (box photo) processing.

    Divides the target image into a W x H grid and extracts
    color/texture features from each cell for matching.
    """

    def __init__(self, image_path, puzzle_width, puzzle_height):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Cannot load target image: {image_path}")
        self.width = puzzle_width
        self.height = puzzle_height
        self.cells = {}
        self._process()

    def _process(self):
        self._grid_segment()
        self._extract_cell_features()

    def _grid_segment(self):
        h, w = self.image.shape[:2]
        cell_w = w / self.width
        cell_h = h / self.height

        for row in range(self.height):
            for col in range(self.width):
                x1 = int(col * cell_w)
                y1 = int(row * cell_h)
                x2 = int((col + 1) * cell_w)
                y2 = int((row + 1) * cell_h)

                self.cells[(row, col)] = {
                    'bounds': (x1, y1, x2, y2),
                    'image': self.image[y1:y2, x1:x2],
                }

    def _extract_cell_features(self):
        for (row, col), cell in self.cells.items():
            cell_img = cell['image']
            if cell_img.size == 0:
                cell['center_color'] = [0, 0, 0]
                cell['histogram'] = None
                cell['edge_colors'] = {
                    'top': [0, 0, 0], 'bottom': [0, 0, 0],
                    'left': [0, 0, 0], 'right': [0, 0, 0],
                }
                continue

            ch, cw = cell_img.shape[:2]
            center = cell_img[ch // 4:3 * ch // 4, cw // 4:3 * cw // 4]
            if center.size > 0:
                cell['center_color'] = np.mean(center, axis=(0, 1)).tolist()
            else:
                cell['center_color'] = [0, 0, 0]

            hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
            mask = np.ones(hsv.shape[:2], dtype=np.uint8)
            cell['histogram'] = {
                'h': cv2.calcHist([hsv], [0], mask, [36], [0, 180]),
                's': cv2.calcHist([hsv], [1], mask, [32], [0, 256]),
                'v': cv2.calcHist([hsv], [2], mask, [32], [0, 256]),
            }

            top_band = cell_img[:max(ch // 6, 1), cw // 4:3 * cw // 4]
            bottom_band = cell_img[5 * ch // 6:, cw // 4:3 * cw // 4]
            left_band = cell_img[ch // 4:3 * ch // 4, :max(cw // 6, 1)]
            right_band = cell_img[ch // 4:3 * ch // 4, 5 * cw // 6:]

            cell['edge_colors'] = {
                'top': np.mean(top_band, axis=(0, 1)).tolist() if top_band.size > 0 else [0, 0, 0],
                'bottom': np.mean(bottom_band, axis=(0, 1)).tolist() if bottom_band.size > 0 else [0, 0, 0],
                'left': np.mean(left_band, axis=(0, 1)).tolist() if left_band.size > 0 else [0, 0, 0],
                'right': np.mean(right_band, axis=(0, 1)).tolist() if right_band.size > 0 else [0, 0, 0],
            }

    def get_candidate_positions(self, piece_features, top_n=5):
        """
        Given a piece's color features, return the top_n most likely
        positions on the target image.
        """
        scores = []
        for (row, col), cell in self.cells.items():
            score = compute_target_match_score(piece_features, cell)
            scores.append((row, col, score))

        scores.sort(key=lambda x: -x[2])
        return scores[:top_n]


def compute_target_match_score(piece_features, cell_features):
    """
    Compute match score between a piece and a target cell.
    Returns a score from 0 to 1 (higher = better match).
    """
    center_color = piece_features.get('center_color', None)
    cell_center = cell_features.get('center_color', [0, 0, 0])

    if center_color is not None:
        center_sim = 1.0 / (1.0 + np.sum(
            (np.array(center_color) - np.array(cell_center)) ** 2
        ) / 1000.0)
    else:
        center_sim = 0.5

    piece_hist = piece_features.get('overall_histogram', None)
    cell_hist = cell_features.get('histogram', None)

    hist_sim = 0.5
    if piece_hist is not None and cell_hist is not None:
        try:
            h_sim = cv2.compareHist(
                piece_hist['h'].astype(np.float32),
                cell_hist['h'].astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            hist_sim = max(0, h_sim)
        except Exception:
            hist_sim = 0.5

    return 0.4 * center_sim + 0.6 * hist_sim


def match_piece_to_target_grid(piece_features, target_cells, top_n=5):
    """
    Match a piece's color features against all target grid cells.
    Returns list of (row, col, score) sorted by score descending.
    """
    scores = []
    for (row, col), cell in target_cells.items():
        score = compute_target_match_score(piece_features, cell)
        scores.append((row, col, score))
    scores.sort(key=lambda x: -x[2])
    return scores[:top_n]
