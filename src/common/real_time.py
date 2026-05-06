"""
Real-time piece identification module.

Provides:
  - PieceMatcher: matches a piece candidate against the database using
    geometric + color verification, with incremental solving support
  - RealTimeIdentifier: live camera capture and piece detection
  - capture_single_piece: identify pieces from a single photo file

This module depends on common.database.PieceDatabase for database management.
"""

import os
import json
import math
import time

import numpy as np
import cv2

from common.preprocess import preprocess_phone_photo, normalize_piece_size
from common.segment_phone import segment_photo
from common.extract import extract_pieces_from_segmented
from common.image_match import (
    ORBMatcher, compute_color_histogram, histogram_similarity,
    compute_color_similarity,
)
from common.database import PieceDatabase
from common import sides


MATCH_THRESHOLD = 3.5
COLOR_VERIFY_THRESHOLD = 0.3
MAX_INCREMENTAL_SOLVE_ATTEMPTS = 3


class MatchResult:
    """Result of matching a single piece against the database."""

    def __init__(self, status, db_piece_id, solution_position=None,
                 confidence=0.0):
        self.status = status
        self.db_piece_id = db_piece_id
        self.solution_position = solution_position
        self.confidence = confidence

    def __repr__(self):
        if self.status == 'known':
            pos = self.solution_position or {}
            return (f"Piece #{self.db_piece_id} -> "
                    f"pos ({pos.get('x', '?')}, {pos.get('y', '?')}), "
                    f"rot {pos.get('rotation', '?')}, "
                    f"conf {self.confidence:.0%}")
        elif self.status == 'new_solved':
            pos = self.solution_position or {}
            return (f"New piece #{self.db_piece_id} -> "
                    f"pos ({pos.get('x', '?')}, {pos.get('y', '?')}) [new]")
        else:
            return f"New piece #{self.db_piece_id} -> position unknown"


class IdentificationResult:
    """Container for identification results from one or more pieces."""

    def __init__(self, results=None):
        self.results = results or []

    def __str__(self):
        return '\n'.join(str(r) for r in self.results)


class PieceMatcher:
    """
    Matches piece candidates against the PieceDatabase.

    Strategy:
      1. Geometric matching: compare four sides with all DB pieces
      2. Color verification: confirm geometric match with color similarity
      3. New piece handling: add to DB, update connectivity, incremental solve
    """

    def __init__(self, database):
        self.db = database
        self.orb_matcher = ORBMatcher(max_features=300)

    def match(self, piece_candidate):
        """
        Match a piece candidate (with .binary and .color) against the DB.
        Returns a MatchResult.
        """
        from common.vector import Vector
        from common.find_islands import save_island_as_bmp
        import tempfile

        binary = piece_candidate.binary
        color = piece_candidate.color

        with tempfile.TemporaryDirectory() as tmpdir:
            bmp_path = os.path.join(tmpdir, 'piece.bmp')
            save_island_as_bmp(binary, bmp_path)

            try:
                v = Vector(bmp_path, 0, tmpdir, {}, (0, 0), 1.0, False)
            except Exception as e:
                print(f"  Vectorization failed: {e}")
                return self._handle_new_piece(piece_candidate)

            if len(v.sides) != 4:
                return self._handle_new_piece(piece_candidate)

        best_matches = []
        for db_pid, db_pd in self.db.pieces.items():
            for si in range(4):
                for sj in range(4):
                    if si < len(v.sides) and v.sides[si].is_edge:
                        continue
                    if sj < len(db_pd.sides) and db_pd.sides[sj].is_edge:
                        continue
                    try:
                        error = v.sides[si].error_when_fit_with(db_pd.sides[sj])
                    except Exception:
                        continue
                    if error < MATCH_THRESHOLD:
                        best_matches.append((db_pid, si, sj, error))

        if best_matches:
            best = min(best_matches, key=lambda x: x[3])
            db_pid = best[0]

            if self._verify_color(piece_candidate, self.db.pieces[db_pid]):
                pos = self.db.get_solution_position(db_pid)
                return MatchResult(
                    status='known',
                    db_piece_id=db_pid,
                    solution_position=pos,
                    confidence=1.0 - best[3] / MATCH_THRESHOLD,
                )

        return self._handle_new_piece(piece_candidate)

    def _verify_color(self, candidate, db_piece):
        """Verify a geometric match with color similarity."""
        if candidate.color is None or db_piece.color_image is None:
            return True
        if db_piece.histogram is None:
            return True

        candidate_hist = compute_color_histogram(candidate.color)
        score = compute_color_similarity(candidate_hist, db_piece.histogram)
        return score > COLOR_VERIFY_THRESHOLD

    def _handle_new_piece(self, candidate):
        """Handle a new (unmatched) piece: add to DB and try incremental solve."""
        from common.vector import Vector
        from common.find_islands import save_island_as_bmp
        import tempfile

        new_id = self.db.next_available_id()

        binary = candidate.binary
        color = candidate.color

        side_data_list = []
        with tempfile.TemporaryDirectory() as tmpdir:
            bmp_path = os.path.join(tmpdir, 'piece.bmp')
            save_island_as_bmp(binary, bmp_path)

            try:
                v = Vector(bmp_path, 0, tmpdir, {}, (0, 0), 1.0, False)
                for s in v.sides:
                    side_data_list.append({
                        'vertices': s.vertices.tolist(),
                        'piece_center': s.piece_center,
                        'is_edge': s.is_edge,
                    })
            except Exception:
                return MatchResult(
                    status='new_unsolved', db_piece_id=new_id,
                    confidence=0.3,
                )

        if len(side_data_list) != 4:
            return MatchResult(
                status='new_unsolved', db_piece_id=new_id,
                confidence=0.3,
            )

        metadata = {
            'photo_source': getattr(candidate, 'photo_id', ''),
            'is_complete': getattr(candidate, 'is_complete', True),
        }

        self.db.add_piece(
            new_id, side_data_list,
            color_image=color, binary_image=binary,
            metadata=metadata,
        )

        self.db.incremental_connectivity_update(new_id)

        solution_found = self._incremental_solve(new_id)

        if solution_found:
            pos = self.db.get_solution_position(new_id)
            return MatchResult(
                status='new_solved',
                db_piece_id=new_id,
                solution_position=pos,
                confidence=0.8,
            )
        else:
            return MatchResult(
                status='new_unsolved',
                db_piece_id=new_id,
                confidence=0.5,
            )

    def _incremental_solve(self, new_id):
        """
        Incremental solving: try to find a position for the new piece.

        Strategy:
          1. If existing solution, try inserting the new piece into empty slots
          2. If no solution or can't insert, attempt partial solve
        """
        if self.db.solution is not None:
            return self._insert_into_existing_solution(new_id)
        else:
            return self._try_solve_with_new_piece(new_id)

    def _insert_into_existing_solution(self, new_id):
        """Try to insert a new piece into the existing solution."""
        from common.board import Board, OPPOSITE

        board = self.db.solution
        new_pd = self.db.pieces[new_id]

        empty_positions = []
        for y in range(board.height):
            for x in range(board.width):
                if board.get(x, y) is None:
                    empty_positions.append((x, y))

        if not empty_positions:
            return False

        for x, y in empty_positions:
            for orientation in range(4):
                ok, _ = board.can_place(
                    new_id, new_pd.fits, x, y, orientation
                )
                if ok:
                    board.place(new_id, new_pd.fits, x, y, orientation)
                    new_pd.solved_position = {
                        'x': x, 'y': y, 'rotation': orientation,
                    }
                    print(f"  Inserted new piece #{new_id} at ({x}, {y}) "
                          f"orientation={orientation}")
                    return True

        return False

    def _try_solve_with_new_piece(self, new_id):
        """Attempt a full solve with the current set of pieces."""
        from common.board import build as board_build

        if self.db.connectivity is None or len(self.db.pieces) < 4:
            return False

        try:
            ps = {}
            for pid, pd in self.db.pieces.items():
                ps[pid] = [[], [], [], []]
                for i in range(4):
                    ps[pid][i] = [
                        (f[0], f[1], f[2]) for f in pd.fits[i]
                    ]

            board = board_build(
                connectivity=ps,
                puzzle_width=self.db.width,
                puzzle_height=self.db.height,
            )
            self.db.solution = board
            self.db.load_from_solution(board)
            return True
        except Exception as e:
            print(f"  Incremental solve attempt failed: {e}")
            return False


class RealTimeIdentifier:
    """
    Real-time piece identification using camera feed or photo files.

    Uses PieceMatcher for identification against a PieceDatabase.
    """

    def __init__(self, piece_db, camera_id=0):
        self.db = piece_db
        self.matcher = PieceMatcher(piece_db)
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

    def identify(self, photo_path):
        """
        Identify pieces from a photo file.
        Returns IdentificationResult.
        """
        bgr, gray = preprocess_phone_photo(photo_path)
        binary = segment_photo(gray, bgr=bgr, method='adaptive')
        pieces = extract_pieces_from_segmented(binary, bgr, 'single')

        results = []
        for piece in pieces:
            if piece.pixel_count < 500:
                continue
            sb, sc, _ = normalize_piece_size(piece.binary, gray, piece.origin,
                                              color=piece.color, target_size=200)
            piece.binary = sb
            if sc is not None:
                piece.color = sc
            result = self.matcher.match(piece)
            results.append(result)

        return IdentificationResult(results)

    def identify_frame(self):
        """
        Capture a frame, detect pieces, and identify them.
        Returns (frame, results) where results is list of MatchResult.
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
            sb, sc, _ = normalize_piece_size(piece.binary, gray, piece.origin,
                                              color=piece.color, target_size=200)
            piece.binary = sb
            if sc is not None:
                piece.color = sc
            result = self.matcher.match(piece)
            results.append(result)

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
                        print("\n" + "=" * 50)
                        for r in results:
                            print(f"  {r}")
                            if r.status in ('known', 'new_solved') and r.solution_position:
                                pos = r.solution_position
                                print(f"    -> Place at row {pos['y']+1}, "
                                      f"col {pos['x']+1}, "
                                      f"{_describe_rotation(pos['rotation'])}")
                        print("=" * 50)
        finally:
            self.stop()
            cv2.destroyAllWindows()


def _describe_rotation(orientation):
    """Convert orientation index (0-3) to human-readable rotation."""
    descriptions = {
        0: "no rotation needed",
        1: "rotate 90 degrees clockwise",
        2: "rotate 180 degrees (upside down)",
        3: "rotate 90 degrees counter-clockwise",
    }
    return descriptions.get(orientation, f"rotation {orientation}")


def capture_single_piece(photo_path, piece_db, output_dir=None):
    """
    Identify a single piece from a photo file.
    Returns list of MatchResult.
    """
    identifier = RealTimeIdentifier(piece_db)
    result = identifier.identify(photo_path)
    return result.results
