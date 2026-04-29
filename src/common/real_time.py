"""
Real-time piece identification module.

Provides:
  - Live camera capture and piece detection
  - Single-piece identification against known piece database
  - Overlay of piece info on camera feed
"""

import os
import json
import time

import numpy as np
import cv2

from common.preprocess import preprocess_phone_photo
from common.segment_phone import segment_photo
from common.extract import extract_pieces_from_segmented
from common.preprocess import normalize_piece_size
from common.image_match import ORBMatcher, compute_color_histogram, histogram_similarity


class PieceDatabase:
    """
    Database of known puzzle pieces for real-time identification.
    """

    def __init__(self):
        self.pieces = {}
        self.orb_matcher = ORBMatcher(max_features=300)
        self._des_cache = {}

    def load_from_directory(self, vector_dir):
        """
        Load piece data from a vectorized output directory.
        Reads side_*.json and color_*.png files.
        """
        import pathlib
        vdir = pathlib.Path(vector_dir)

        piece_ids = set()
        for f in vdir.glob('side_*_0.json'):
            pid = int(f.name.split('_')[1])
            piece_ids.add(pid)

        for pid in piece_ids:
            piece_data = {
                'id': pid,
                'sides': [],
                'color_image': None,
                'descriptors': None,
                'histogram': None,
                'solved_position': None,
            }

            for j in range(4):
                side_file = vdir / f'side_{pid}_{j}.json'
                if side_file.exists():
                    with open(side_file) as f:
                        data = json.load(f)
                    piece_data['sides'].append(data)

            color_file = vdir / f'color_{pid}.png'
            if color_file.exists():
                img = cv2.imread(str(color_file))
                if img is not None:
                    piece_data['color_image'] = img
                    _, des = self.orb_matcher.detect_and_compute(img)
                    piece_data['descriptors'] = des
                    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
                    piece_data['histogram'] = compute_color_histogram(img, mask)

            color_feat_file = vdir / f'color_features_{pid}.json'
            if color_feat_file.exists():
                with open(color_feat_file) as f:
                    piece_data['color_features'] = json.load(f)

            self.pieces[pid] = piece_data

        print(f"Loaded {len(self.pieces)} pieces into database")
        return len(self.pieces)

    def identify_piece(self, piece_image, top_n=3):
        """
        Identify a piece from an image against the database.
        Returns list of (piece_id, confidence) sorted by confidence.
        """
        if not self.pieces:
            return []

        scores = []

        _, query_des = self.orb_matcher.detect_and_compute(piece_image)
        query_hist = compute_color_histogram(piece_image)

        for pid, piece_data in self.pieces.items():
            score = 0.0

            if query_des is not None and piece_data.get('descriptors') is not None:
                matches = self.orb_matcher.match(query_des, piece_data['descriptors'])
                if matches:
                    avg_dist = np.mean([m.distance for m in matches])
                    orb_score = max(0, 1.0 - avg_dist / 256)
                    n_features = max(len(query_des), len(piece_data['descriptors']), 1)
                    coverage = min(len(matches) / n_features, 1.0)
                    score += orb_score * 0.4 + coverage * 0.2
                else:
                    score += 0.1
            else:
                score += 0.1

            if query_hist is not None and piece_data.get('histogram') is not None:
                hist_sim = histogram_similarity(query_hist, piece_data['histogram'])
                score += max(0, hist_sim) * 0.4

            scores.append((pid, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_n]


class RealTimeIdentifier:
    """
    Real-time piece identification using camera feed.
    """

    def __init__(self, piece_db, camera_id=0):
        self.db = piece_db
        self.camera_id = camera_id
        self.cap = None
        self.running = False

    def start(self):
        """Open the camera."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        self.running = True

    def stop(self):
        """Close the camera."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def identify_frame(self):
        """
        Capture a frame, detect pieces, and identify them.
        Returns (frame, results) where results is list of
        (piece_image, [(piece_id, confidence), ...]).
        """
        if self.cap is None or not self.cap.isOpened():
            return None, []

        ret, frame = self.cap.read()
        if not ret:
            return None, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = segment_photo(gray, bgr=frame, method='adaptive')
        pieces = extract_pieces_from_segmented(binary, frame, 'live')

        results = []
        for piece in pieces:
            if piece.pixel_count < 500:
                continue
            sb, sc, _ = normalize_piece_size(piece.binary, piece.color,
                                              target_size=200)
            matches = self.db.identify_piece(sc)
            results.append((sc, matches))

        return frame, results

    def run_interactive(self, display=True):
        """
        Run interactive identification loop.
        Press 'q' to quit, SPACE to capture and identify.
        """
        self.start()
        print("Real-time identification started.")
        print("Press SPACE to capture and identify, 'q' to quit.")

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                if display:
                    display_frame = frame.copy()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    binary = segment_photo(gray, bgr=frame, method='adaptive')

                    contours, _ = cv2.findContours(
                        binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)

                    cv2.imshow('Puzzle Bot - Real-time', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    frame_result, results = self.identify_frame()
                    if results:
                        for i, (piece_img, matches) in enumerate(results):
                            if matches:
                                best_id, best_conf = matches[0]
                                print(f"  Piece {i}: best match = #{best_id} "
                                      f"(confidence: {best_conf:.2f})")
        finally:
            self.stop()
            cv2.destroyAllWindows()


def capture_single_piece(photo_path, piece_db, output_dir=None):
    """
    Identify a single piece from a photo file.
    Returns list of (piece_id, confidence).
    """
    bgr, gray = preprocess_phone_photo(photo_path)
    binary = segment_photo(gray, bgr=bgr, method='adaptive')
    pieces = extract_pieces_from_segmented(binary, bgr, 'single')

    all_results = []
    for piece in pieces:
        if piece.pixel_count < 500:
            continue
        sb, sc, _ = normalize_piece_size(piece.binary, piece.color,
                                          target_size=200)
        matches = piece_db.identify_piece(sc)
        all_results.append({
            'origin': piece.origin,
            'matches': matches,
        })

    return all_results
