import os
import math

import pytest

from tests.helpers import (
    create_synthetic_piece_bmp,
    match_corners_nearest_neighbor,
    print_corner_comparison_table,
    generate_corner_comparison_svg,
    generate_piece_outline_svg,
)


class TestCornerMatching:
    def test_match_identical_corners(self):
        detected = [(100, 200), (300, 400), (500, 600), (700, 800)]
        true = [(100, 200), (300, 400), (500, 600), (700, 800)]
        results, passed = match_corners_nearest_neighbor(detected, true, max_dist=50)
        assert passed is True
        assert all("MISS" not in r for r in results)

    def test_match_nearby_corners(self):
        detected = [(102, 201), (301, 402), (499, 599), (701, 798)]
        true = [(100, 200), (300, 400), (500, 600), (700, 800)]
        results, passed = match_corners_nearest_neighbor(detected, true, max_dist=50)
        assert passed is True

    def test_match_missing_corner(self):
        detected = [(100, 200), (300, 400), (500, 600), (9999, 9999)]
        true = [(100, 200), (300, 400), (500, 600), (700, 800)]
        results, passed = match_corners_nearest_neighbor(detected, true, max_dist=50)
        assert passed is False
        assert any("MISS" in r for r in results)

    def test_corner_comparison_table_output(self, capsys):
        results = [
            {
                'piece_id': 1,
                'detected': [(100, 200), (300, 400)],
                'true': [(100, 200), (300, 400)],
                'matches': ['0px', '0px'],
                'passed': True,
            },
            {
                'piece_id': 2,
                'detected': [(50, 50)],
                'true': [(100, 200)],
                'matches': ['MISS(180)'],
                'passed': False,
            },
        ]
        print_corner_comparison_table(results)
        captured = capsys.readouterr()
        assert 'Piece_ 1' in captured.out
        assert 'Piece_ 2' in captured.out
        assert 'OK' in captured.out
        assert 'FAIL' in captured.out
        assert 'SOME FAILED' in captured.out


class TestCornerDetectionOnRealData:
    @pytest.fixture(autouse=True)
    def setup_real_data(self):
        self.vector_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'example_data', '3_vector'
        )
        self.true_corners_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'example_data', 'input', 'true_corners.json'
        )
        if not os.path.isdir(self.vector_dir):
            pytest.skip("example_data/3_vector not found")

    def test_detected_corners_match_true_corners(self):
        import json
        from common.vector import Vector

        if not os.path.exists(self.true_corners_path):
            pytest.skip("true_corners.json not found")

        with open(self.true_corners_path) as f:
            true_data = json.load(f)

        for pid in range(1, 4):
            img_path = os.path.join(self.vector_dir, f'piece_{pid}.bmp')
            if not os.path.exists(img_path):
                continue

            p = Vector.from_file(img_path, id=pid)
            p.find_border_raster()
            p.vectorize()
            p.find_four_corners()
            det_c = [(int(c[0]), int(c[1])) for c in p.corners]

            true_corners = true_data.get(str(pid), [])
            if not true_corners:
                continue

            flat_true = []
            for tc_list in true_corners:
                flat_true.extend([tuple(tc) for tc in tc_list])

            results, passed = match_corners_nearest_neighbor(det_c, flat_true, max_dist=50)
            assert passed, f"Piece {pid} corner matching failed: {results}"

    def test_corner_svg_generation(self, tmp_path):
        from common.vector import Vector

        img_path = os.path.join(self.vector_dir, 'piece_1.bmp')
        if not os.path.exists(img_path):
            pytest.skip("piece_1.bmp not found")

        p = Vector.from_file(img_path, id=1)
        p.find_border_raster()
        p.vectorize()
        p.find_four_corners()

        true_corners = [[(int(c[0]), int(c[1]))] for c in p.corners]
        svg_path = str(tmp_path / 'result_piece_1.svg')
        status, det_c, results = generate_corner_comparison_svg(p, true_corners, svg_path)
        assert os.path.exists(svg_path)
        assert 'OK' in status or 'FAIL' in status

    def test_piece_outline_svg_generation(self, tmp_path):
        from common.vector import Vector

        img_path = os.path.join(self.vector_dir, 'piece_1.bmp')
        if not os.path.exists(img_path):
            pytest.skip("piece_1.bmp not found")

        p = Vector.from_file(img_path, id=1)
        p.find_border_raster()
        p.vectorize()
        p.find_four_corners()

        svg_path = str(tmp_path / 'outline_piece_1.svg')
        generate_piece_outline_svg(p, svg_path, piece_label=1)
        assert os.path.exists(svg_path)

        with open(svg_path) as f:
            content = f.read()
        assert '#1a1a2e' in content
        assert 'Piece 1' in content
        assert 'C0' in content
        assert 'C3' in content
