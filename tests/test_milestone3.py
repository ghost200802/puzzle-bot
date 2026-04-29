"""
Tests for Milestone 3: Image matching enhancement + real-time identification
"""

import os
import sys
import json
import math

import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestORBMatcher:
    def test_detect_and_compute_color(self):
        from common.image_match import ORBMatcher
        matcher = ORBMatcher(max_features=100)
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        kp, des = matcher.detect_and_compute(img)
        assert des is not None or len(kp) == 0

    def test_match_identical_images(self):
        from common.image_match import ORBMatcher
        matcher = ORBMatcher(max_features=200)
        img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        _, des = matcher.detect_and_compute(img)
        matches = matcher.match(des, des)
        assert len(matches) > 0

    def test_match_none_descriptors(self):
        from common.image_match import ORBMatcher
        matcher = ORBMatcher()
        assert matcher.match(None, None) == []

    def test_compute_match_score_identical(self):
        from common.image_match import ORBMatcher
        matcher = ORBMatcher(max_features=200)
        img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        score = matcher.compute_match_score(img, img)
        assert score > 0.5


class TestCombinedMatchScore:
    def test_combined_score_all_perfect(self):
        from common.image_match import compute_combined_match_score
        score = compute_combined_match_score(0, 1.0, 1.0)
        assert score == 1.0

    def test_combined_score_all_bad(self):
        from common.image_match import compute_combined_match_score
        score = compute_combined_match_score(20, 0.0, 0.0)
        assert score < 0.2

    def test_combined_score_geo_dominant(self):
        from common.image_match import compute_combined_match_score
        score_good_geo = compute_combined_match_score(0.5, 0.0, 0.0,
                                                       geo_weight=0.7, img_weight=0.15, color_weight=0.15)
        score_bad_geo = compute_combined_match_score(10.0, 1.0, 1.0,
                                                      geo_weight=0.7, img_weight=0.15, color_weight=0.15)
        assert score_good_geo > score_bad_geo


class TestPieceDatabase:
    def _create_test_piece_files(self, tmp_path, n_pieces=3):
        vdir = tmp_path / 'vector'
        vdir.mkdir()
        for pid in range(1, n_pieces + 1):
            for j in range(4):
                data = {
                    'vertices': [[10, 10], [50, 10], [50, 50], [10, 50]],
                    'piece_center': [30, 30],
                    'is_edge': j == 0,
                    'original_photo_name': 'test.jpg',
                    'photo_width': 500, 'photo_height': 500,
                    'photo_space_centroid': [250, 250],
                }
                with open(vdir / f'side_{pid}_{j}.json', 'w') as f:
                    json.dump(data, f)
            color_img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(vdir / f'color_{pid}.png'), color_img)
        return str(vdir)

    def test_load_from_directory(self, tmp_path):
        from common.real_time import PieceDatabase
        db = PieceDatabase()
        vdir = self._create_test_piece_files(tmp_path, n_pieces=3)
        count = db.load_from_directory(vdir)
        assert count == 3

    def test_identify_piece(self, tmp_path):
        from common.real_time import PieceDatabase
        db = PieceDatabase()
        vdir = self._create_test_piece_files(tmp_path, n_pieces=3)
        db.load_from_directory(vdir)
        test_img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        results = db.identify_piece(test_img, top_n=3)
        assert len(results) == 3

    def test_identify_empty_db(self):
        from common.real_time import PieceDatabase
        db = PieceDatabase()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        results = db.identify_piece(img)
        assert results == []


class TestRealTimeIdentifier:
    def test_init(self):
        from common.real_time import RealTimeIdentifier, PieceDatabase
        db = PieceDatabase()
        identifier = RealTimeIdentifier(db, camera_id=0)
        assert identifier.db is db

    def test_stop_without_start(self):
        from common.real_time import RealTimeIdentifier, PieceDatabase
        db = PieceDatabase()
        identifier = RealTimeIdentifier(db)
        identifier.stop()
        assert not identifier.running


class TestConnectColorMatching:
    def test_load_color_data(self, tmp_path):
        from common.connect import _load_color_data
        vdir = tmp_path / 'vector'
        vdir.mkdir()
        for pid in [1, 2]:
            cf = {
                'center_color': [128, 128, 128],
                'overall_histogram': {
                    'h': np.ones(36).tolist(),
                    's': np.ones(32).tolist(),
                    'v': np.ones(32).tolist(),
                }
            }
            with open(vdir / f'color_features_{pid}.json', 'w') as f:
                json.dump(cf, f)
        data = _load_color_data(str(vdir))
        assert len(data) >= 2

    def test_compute_color_match_bonus(self):
        from common.connect import _compute_color_match_bonus
        hist = {
            'h': np.ones(36).tolist(),
            's': np.ones(32).tolist(),
            'v': np.ones(32).tolist(),
        }
        color_data = {
            1: {'overall_histogram': hist},
            2: {'overall_histogram': hist},
        }
        bonus = _compute_color_match_bonus(1, 2, 0, 2, color_data)
        assert 0 <= bonus <= 1

    def test_compute_color_match_bonus_no_data(self):
        from common.connect import _compute_color_match_bonus
        bonus = _compute_color_match_bonus(1, 2, 0, 2, {})
        assert bonus == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
