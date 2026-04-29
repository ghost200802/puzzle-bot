"""
Tests for Milestone 2: Target image + visual output
"""

import os
import sys
import json
import math

import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestTargetImage:
    def _make_test_target(self, tmp_path, w=4, h=3):
        img = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
        path = str(tmp_path / 'target.jpg')
        cv2.imwrite(path, img)
        return path, img

    def test_target_image_init(self, tmp_path):
        from common.target import TargetImage
        path, _ = self._make_test_target(tmp_path, w=4, h=3)
        target = TargetImage(path, puzzle_width=4, puzzle_height=3)
        assert target.width == 4
        assert target.height == 3
        assert len(target.cells) == 12

    def test_target_image_grid_bounds(self, tmp_path):
        from common.target import TargetImage
        path, _ = self._make_test_target(tmp_path, w=4, h=3)
        target = TargetImage(path, puzzle_width=4, puzzle_height=3)
        for (row, col), cell in target.cells.items():
            x1, y1, x2, y2 = cell['bounds']
            assert x2 > x1
            assert y2 > y1

    def test_target_image_cell_features(self, tmp_path):
        from common.target import TargetImage
        path, _ = self._make_test_target(tmp_path, w=4, h=3)
        target = TargetImage(path, puzzle_width=4, puzzle_height=3)
        for (row, col), cell in target.cells.items():
            assert 'center_color' in cell
            assert 'histogram' in cell
            assert 'edge_colors' in cell
            assert len(cell['center_color']) == 3

    def test_target_image_get_candidates(self, tmp_path):
        from common.target import TargetImage
        path, _ = self._make_test_target(tmp_path, w=4, h=3)
        target = TargetImage(path, puzzle_width=4, puzzle_height=3)
        features = {'center_color': [128, 128, 128]}
        candidates = target.get_candidate_positions(features, top_n=3)
        assert len(candidates) == 3

    def test_target_image_not_found(self):
        from common.target import TargetImage
        with pytest.raises(FileNotFoundError):
            TargetImage('/nonexistent/path.jpg', 4, 3)


class TestImageMatch:
    def test_compute_color_histogram(self):
        from common.image_match import compute_color_histogram
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        hist = compute_color_histogram(img)
        assert 'h' in hist
        assert hist['h'].shape == (36, 1)

    def test_histogram_similarity_identical(self):
        from common.image_match import histogram_similarity
        hist = {
            'h': np.ones((36, 1), dtype=np.float32),
            's': np.ones((32, 1), dtype=np.float32),
            'v': np.ones((32, 1), dtype=np.float32),
        }
        score = histogram_similarity(hist, hist)
        assert abs(score - 1.0) < 0.01

    def test_color_continuity_score(self):
        from common.image_match import color_continuity_score
        band_a = [[100, 100, 100]] * 10
        band_b = [[100, 100, 100]] * 10
        score = color_continuity_score(band_a, band_b)
        assert score > 0.9

    def test_color_continuity_none(self):
        from common.image_match import color_continuity_score
        assert color_continuity_score(None, None) == 1.0

    def test_compute_color_similarity(self):
        from common.image_match import compute_color_similarity
        hist = {
            'h': np.ones((36, 1), dtype=np.float32),
            's': np.ones((32, 1), dtype=np.float32),
            'v': np.ones((32, 1), dtype=np.float32),
        }
        score = compute_color_similarity(hist, hist)
        assert score > 0.5


class TestOutputEnhanced:
    def test_generate_annotated_target(self, tmp_path):
        from common.output import generate_annotated_target
        from common.board import Board
        target_img = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
        b = Board(width=4, height=3)
        b.place(1, [[], [], [], []], 0, 0, 0)
        b.place(2, [[], [], [], []], 1, 0, 0)
        path = generate_annotated_target(target_img, b, str(tmp_path))
        assert os.path.exists(path)

    def test_generate_piece_catalog_html(self, tmp_path):
        from common.output import generate_piece_catalog_html
        pieces = [
            {'id': 1, 'is_corner': True, 'is_edge': True, 'solved_position': (0, 0)},
            {'id': 2, 'is_corner': False, 'is_edge': True, 'solved_position': (1, 0)},
            {'id': 3, 'is_corner': False, 'is_edge': False},
        ]
        path = generate_piece_catalog_html(pieces, str(tmp_path))
        assert os.path.exists(path)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        assert '#1' in content
        assert 'corner' in content

    def test_solution_grid_still_works(self, tmp_path):
        from common.output import generate_solution_grid
        from common.board import Board
        b = Board(width=3, height=2)
        b.place(1, [[], [], [], []], 0, 0, 0)
        path = generate_solution_grid(b, str(tmp_path))
        assert os.path.exists(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
