import os

import numpy as np
import pytest

from tests.helpers import create_synthetic_piece_bmp, print_vector_details


class TestNormalization:
    @pytest.fixture(autouse=True)
    def setup_real_data(self):
        self.vector_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'example_data', '3_vector'
        )
        if not os.path.isdir(self.vector_dir):
            pytest.skip("example_data/3_vector not found")

    def test_load_and_normalize_piece_bmp(self):
        from common.util import load_bmp_as_binary_pixels

        img_path = os.path.join(self.vector_dir, 'piece_1.bmp')
        if not os.path.exists(img_path):
            pytest.skip("piece_1.bmp not found")

        bp, bw, bh = load_bmp_as_binary_pixels(img_path)
        assert bw > 0
        assert bh > 0
        assert np.sum(bp == 1) > 0
        assert np.sum(bp == 0) > 0

    def test_normalized_bmp_round_trip(self, tmp_path):
        from common.find_islands import save_island_as_bmp
        from common.util import load_bmp_as_binary_pixels

        binary = np.zeros((100, 100), dtype=np.uint8)
        binary[20:80, 20:80] = 1

        bmp_path = str(tmp_path / 'test_round_trip.bmp')
        save_island_as_bmp(binary, bmp_path)

        loaded, w, h = load_bmp_as_binary_pixels(bmp_path)
        assert w == 100
        assert h == 100
        assert np.sum(loaded) == np.sum(binary)

    def test_vectorize_after_normalization(self):
        from common.vector import Vector

        img_path = os.path.join(self.vector_dir, 'piece_1.bmp')
        if not os.path.exists(img_path):
            pytest.skip("piece_1.bmp not found")

        v = Vector.from_file(img_path, id=1)
        v.find_border_raster()
        v.vectorize()
        assert len(v.vertices) > 10

    def test_corners_found_after_normalization(self):
        from common.vector import Vector

        img_path = os.path.join(self.vector_dir, 'piece_1.bmp')
        if not os.path.exists(img_path):
            pytest.skip("piece_1.bmp not found")

        v = Vector.from_file(img_path, id=1)
        v.find_border_raster()
        v.vectorize()
        v.find_four_corners()
        assert len(v.corners) == 4
        for corner in v.corners:
            assert corner[0] > 0
            assert corner[1] > 0

    def test_sides_extracted_after_normalization(self):
        from common.vector import Vector

        img_path = os.path.join(self.vector_dir, 'piece_1.bmp')
        if not os.path.exists(img_path):
            pytest.skip("piece_1.bmp not found")

        v = Vector.from_file(img_path, id=1)
        v.find_border_raster()
        v.vectorize()
        v.find_four_corners()
        v.extract_four_sides()
        assert len(v.sides) == 4
        for i, side in enumerate(v.sides):
            assert len(side.vertices) > 0
            assert hasattr(side, 'is_edge')
            assert isinstance(side.is_edge, bool)

    def test_vector_details_print(self, capsys, tmp_path):
        from common.vector import Vector

        path, _ = create_synthetic_piece_bmp(tmp_path)
        v = Vector.from_file(path, id=1)
        v.find_border_raster()
        v.vectorize()
        v.find_four_corners()

        print_vector_details(v, label="synthetic")
        captured = capsys.readouterr()
        assert 'synthetic' in captured.out
        assert 'Vector:' in captured.out
        assert 'Corner' in captured.out
        assert 'Side' in captured.out or 'extract_four_sides error' in captured.out
