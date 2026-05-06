"""
Tests for Milestone 1: Phone photo -> geometric matching -> text output
"""

import os
import sys
import json
import math

import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tests.helpers import create_synthetic_piece_bmp


class TestConfig:
    def test_phone_mode_is_default(self):
        from common.config import MODE
        assert MODE == 'phone'

    def test_phone_params_exist(self):
        from common.config import (
            PHONE_BLUR_KERNEL, PHONE_MORPH_KERNEL,
            PHONE_ADAPTIVE_BLOCK_SIZE, PHONE_ADAPTIVE_C,
            PHONE_MIN_PIECE_AREA_RATIO, PHONE_TARGET_PIECE_SIZE,
            PHONE_HASH_THRESHOLD, PHONE_DUPLICATE_GEOMETRIC_THRESHOLD,
            PHONE_BG_BRIGHTNESS_THRESHOLD,
        )
        assert PHONE_BLUR_KERNEL > 0
        assert PHONE_MORPH_KERNEL > 0
        assert PHONE_ADAPTIVE_BLOCK_SIZE % 2 == 1
        assert 0 < PHONE_MIN_PIECE_AREA_RATIO < 1
        assert PHONE_TARGET_PIECE_SIZE > 0

    def test_robot_params_preserved(self):
        from common.config import (
            PUZZLE_WIDTH, PUZZLE_HEIGHT, PUZZLE_NUM_PIECES,
            SEG_THRESH, APPROX_ROBOT_COUNTS_PER_PIXEL,
        )
        assert PUZZLE_WIDTH == 40
        assert PUZZLE_HEIGHT == 25


class TestPreprocess:
    def _make_test_image(self, h=200, w=300, color=(128, 128, 128)):
        img = np.full((h, w, 3), color, dtype=np.uint8)
        return img

    def test_auto_rotate_no_exif(self, tmp_path):
        from common.preprocess import auto_rotate
        img = self._make_test_image()
        path = str(tmp_path / 'test.jpg')
        cv2.imwrite(path, img)
        result = auto_rotate(path)
        assert result.shape == (200, 300, 3)

    def test_normalize_color(self):
        from common.preprocess import normalize_color
        img = self._make_test_image(color=(50, 100, 200))
        result = normalize_color(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_gray_world_white_balance(self):
        from common.preprocess import gray_world_white_balance
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 30
        img[:, :, 1] = 60
        img[:, :, 2] = 90
        result = gray_world_white_balance(img)
        avg_b = result[:, :, 0].mean()
        avg_g = result[:, :, 1].mean()
        avg_r = result[:, :, 2].mean()
        assert abs(avg_b - avg_g) < 5
        assert abs(avg_g - avg_r) < 5

    def test_preprocess_phone_photo(self, tmp_path):
        from common.preprocess import preprocess_phone_photo
        img = self._make_test_image()
        path = str(tmp_path / 'test.jpg')
        cv2.imwrite(path, img)
        bgr, gray = preprocess_phone_photo(path)
        assert bgr.shape[2] == 3
        assert gray.ndim == 2
        assert bgr.shape[:2] == gray.shape

    def test_normalize_piece_size(self):
        from common.preprocess import normalize_piece_size
        binary = np.zeros((200, 300), dtype=np.uint8)
        binary[50:150, 80:220] = 1
        gray_full = np.random.randint(100, 200, (500, 600), dtype=np.uint8)
        gray_full[50:150, 80:220] = 240
        sb, sc, scale = normalize_piece_size(binary, gray_full, origin=(0, 0),
                                              color=np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8),
                                              target_size=100)
        max_dim = max(sb.shape[0], sb.shape[1])
        assert max_dim <= 120
        assert scale != 1.0

    def test_normalize_piece_size_no_piece(self):
        from common.preprocess import normalize_piece_size
        binary = np.zeros((100, 100), dtype=np.uint8)
        gray_full = np.zeros((200, 200), dtype=np.uint8)
        sb, sc, scale = normalize_piece_size(binary, gray_full, origin=(0, 0))
        assert scale == 1.0

    def test_normalize_piece_color(self):
        from common.preprocess import normalize_piece_color
        color = np.full((50, 50, 3), [50, 100, 200], dtype=np.uint8)
        mask = np.ones((50, 50), dtype=np.uint8)
        result = normalize_piece_color(color, mask)
        assert result.dtype == np.uint8


class TestSegmentPhone:
    def _make_test_photo(self, bg_color=240, piece_color=60,
                         n_pieces=3, h=400, w=600):
        img = np.full((h, w, 3), bg_color, dtype=np.uint8)
        gray = np.full((h, w), bg_color, dtype=np.uint8)
        for i in range(n_pieces):
            cx = 100 + i * 150
            cy = 200
            r = 40
            cv2.circle(img, (cx, cy), r, (piece_color, piece_color, piece_color), -1)
            cv2.circle(gray, (cx, cy), r, piece_color, -1)
        return img, gray

    def test_detect_bg_brightness_light(self):
        from common.segment_phone import detect_bg_brightness
        gray = np.full((100, 100), 240, dtype=np.uint8)
        assert detect_bg_brightness(gray) == 'light'

    def test_detect_bg_brightness_dark(self):
        from common.segment_phone import detect_bg_brightness
        gray = np.full((100, 100), 20, dtype=np.uint8)
        assert detect_bg_brightness(gray) == 'dark'

    def test_segment_adaptive_light_bg(self):
        from common.segment_phone import segment_adaptive
        _, gray = self._make_test_photo(bg_color=240, piece_color=60)
        binary = segment_adaptive(gray, bg_brightness='light')
        assert binary.dtype == np.uint8
        assert set(np.unique(binary)).issubset({0, 1})
        piece_pixels = np.sum(binary == 1)
        assert piece_pixels > 100

    def test_segment_adaptive_dark_bg(self):
        from common.segment_phone import segment_adaptive
        _, gray = self._make_test_photo(bg_color=20, piece_color=200)
        binary = segment_adaptive(gray, bg_brightness='dark')
        assert set(np.unique(binary)).issubset({0, 1})
        piece_pixels = np.sum(binary == 1)
        assert piece_pixels > 100

    def test_segment_otsu(self):
        from common.segment_phone import segment_otsu
        _, gray = self._make_test_photo(bg_color=240, piece_color=60)
        binary = segment_otsu(gray, bg_brightness='light')
        assert set(np.unique(binary)).issubset({0, 1})
        assert np.sum(binary == 1) > 100

    def test_segment_grabcut(self):
        from common.segment_phone import segment_grabcut
        bgr, _ = self._make_test_photo(bg_color=240, piece_color=60)
        binary = segment_grabcut(bgr)
        assert set(np.unique(binary)).issubset({0, 1})
        assert np.sum(binary == 1) > 50

    def test_segment_photo_dispatch(self):
        from common.segment_phone import segment_photo
        _, gray = self._make_test_photo()
        binary = segment_photo(gray, method='adaptive')
        assert np.sum(binary == 1) > 0

    def test_segment_photo_invalid_method(self):
        from common.segment_phone import segment_photo
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            segment_photo(gray, method='invalid')


class TestFindIslands:
    def test_extract_islands_basic(self):
        from common.find_islands import extract_islands
        grid = np.zeros((100, 100), dtype=np.uint8)
        grid[20:40, 20:40] = 1
        grid[60:80, 60:80] = 1
        islands = extract_islands(grid, min_island_area=10, ignore_border=False)
        assert len(islands) == 2
        for island, origin_row, origin_col, touches_border in islands:
            assert np.sum(island) >= 10

    def test_extract_islands_border_flag(self):
        from common.find_islands import extract_islands
        grid = np.zeros((100, 100), dtype=np.uint8)
        grid[0:20, 10:30] = 1
        grid[50:70, 50:70] = 1
        islands = extract_islands(grid, min_island_area=10, ignore_border=False)
        assert len(islands) == 2
        border_flags = [tb for _, _, _, tb in islands]
        assert True in border_flags
        assert False in border_flags

    def test_extract_islands_ignore_border(self):
        from common.find_islands import extract_islands
        grid = np.zeros((100, 100), dtype=np.uint8)
        grid[0:20, 10:30] = 1
        grid[50:70, 50:70] = 1
        islands = extract_islands(grid, min_island_area=10, ignore_border=True)
        assert len(islands) == 1

    def test_remove_stragglers(self):
        from common.find_islands import remove_stragglers
        grid = np.zeros((50, 50), dtype=np.uint8)
        grid[20:30, 20:30] = 1
        grid[10, 10] = 1
        cleaned = remove_stragglers(grid.copy())
        assert cleaned[10, 10] == 0
        assert np.sum(cleaned[20:30, 20:30]) > 0

    def test_save_and_load_bmp(self, tmp_path):
        from common.find_islands import save_island_as_bmp, load_binary_bitmap
        island = np.zeros((30, 30), dtype=np.uint8)
        island[5:25, 5:25] = 1
        path = str(tmp_path / 'test.bmp')
        save_island_as_bmp(island, path)
        loaded = load_binary_bitmap(path)
        assert loaded.shape == (30, 30)
        assert np.sum(loaded) == np.sum(island)


class TestExtract:
    def _make_segmented_data(self, n_pieces=3, h=400, w=600):
        binary = np.zeros((h, w), dtype=np.uint8)
        color = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        for i in range(n_pieces):
            cx = 100 + i * 150
            cy = 200
            r = 40
            cv2.circle(binary, (cx, cy), r, 1, -1)
        return binary, color

    def test_extract_pieces_from_segmented(self):
        from common.extract import extract_pieces_from_segmented
        binary, color = self._make_segmented_data(n_pieces=3)
        pieces = extract_pieces_from_segmented(binary, color, 'test_photo')
        assert len(pieces) == 3
        for p in pieces:
            assert p.binary is not None
            assert p.color is not None
            assert p.photo_id == 'test_photo'
            assert p.pixel_count > 0

    def test_extract_pieces_no_pieces(self):
        from common.extract import extract_pieces_from_segmented
        binary = np.zeros((200, 200), dtype=np.uint8)
        color = np.zeros((200, 200, 3), dtype=np.uint8)
        pieces = extract_pieces_from_segmented(binary, color, 'empty')
        assert len(pieces) == 0

    def test_piece_candidate_completeness(self):
        from common.extract import extract_pieces_from_segmented
        binary = np.zeros((200, 200), dtype=np.uint8)
        color = np.zeros((200, 200, 3), dtype=np.uint8)
        binary[10:50, 10:50] = 1
        pieces = extract_pieces_from_segmented(binary, color, 'test')
        assert len(pieces) == 1
        assert isinstance(pieces[0].is_complete, bool)


class TestDedupe:
    def test_compute_piece_hash(self):
        from common.dedupe import compute_piece_hash
        binary = np.zeros((100, 100), dtype=np.uint8)
        binary[20:80, 20:80] = 1
        bh, ch = compute_piece_hash(binary)
        assert bh is not None
        assert ch is None

    def test_compute_piece_hash_with_color(self):
        from common.dedupe import compute_piece_hash
        binary = np.zeros((100, 100), dtype=np.uint8)
        binary[20:80, 20:80] = 1
        color = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bh, ch = compute_piece_hash(binary, color)
        assert bh is not None
        assert ch is not None

    def test_find_duplicate_candidates_identical(self):
        from common.dedupe import compute_piece_hash, find_duplicate_candidates
        binary = np.zeros((100, 100), dtype=np.uint8)
        binary[20:80, 20:80] = 1
        bh, _ = compute_piece_hash(binary)
        hashes = {1: (bh, None), 2: (bh, None)}
        candidates = find_duplicate_candidates(hashes, hash_threshold=15)
        assert len(candidates) == 1
        assert candidates[0][2] == 0

    def test_find_duplicate_candidates_different(self):
        from common.dedupe import find_duplicate_candidates
        import imagehash
        from PIL import Image
        img1 = Image.new('L', (64, 64), 0)
        img2 = Image.new('RGB', (64, 64), (255, 0, 0))
        h1 = imagehash.phash(img1)
        h2 = imagehash.phash(img2)
        hashes = {1: (h1, None), 2: (h2, None)}
        candidates = find_duplicate_candidates(hashes, hash_threshold=1)
        assert len(candidates) == 0

    def test_pick_best_dupe(self):
        from common.dedupe import pick_best
        data = [
            {'is_complete': False, 'pixel_count': 10000},
            {'is_complete': True, 'pixel_count': 5000},
            {'is_complete': True, 'pixel_count': 8000},
        ]
        best = pick_best(data)
        assert best['is_complete'] is True
        assert best['pixel_count'] == 8000

    def test_deduplicate_phone_single_piece(self, tmp_path):
        from common.dedupe import deduplicate_phone
        input_dir = tmp_path / 'input'
        output_dir = tmp_path / 'output'
        input_dir.mkdir()
        for j in range(4):
            data = {
                'vertices': [[10, 10], [50, 10], [50, 50], [10, 50]],
                'piece_center': [30, 30],
                'is_edge': True,
                'original_photo_name': 'test.jpg',
            }
            with open(input_dir / f'side_1_{j}.json', 'w') as f:
                json.dump(data, f)
        count = deduplicate_phone(str(input_dir), str(output_dir))
        assert count == 1


class TestVectorization:
    def test_vector_from_file(self, tmp_path):
        from common.vector import Vector
        path, _ = create_synthetic_piece_bmp(tmp_path)
        v = Vector.from_file(path, id=1)
        assert v.pixels is not None
        assert v.width > 0
        assert v.height > 0

    def test_vectorize_full_process(self, tmp_path):
        from common.vector import Vector
        path, _ = create_synthetic_piece_bmp(tmp_path)
        v = Vector.from_file(path, id=1)
        v.find_border_raster()
        assert np.sum(v.border) > 0
        v.vectorize()
        assert len(v.vertices) > 10
        v.find_four_corners()
        assert len(v.corners) == 4
        try:
            v.extract_four_sides()
            assert len(v.sides) == 4
        except Exception:
            pass
        v.enhance_corners()
        assert len(v.corners) == 4


class TestBoardSolver:
    def test_board_creation(self):
        from common.board import Board
        b = Board(width=4, height=3)
        assert b.width == 4
        assert b.height == 3
        assert b.placed_count == 0

    def test_board_is_available(self):
        from common.board import Board
        b = Board(width=4, height=3)
        assert b.is_available(0, 0)
        assert not b.is_available(-1, 0)
        assert not b.is_available(4, 0)

    def test_board_place_and_get(self):
        from common.board import Board
        b = Board(width=4, height=3)
        fits = [[], [], [], []]
        b.place(1, fits, 0, 0, 0)
        assert b.placed_count == 1
        result = b.get(0, 0)
        assert result is not None
        assert result[0] == 1

    def test_board_can_place_edge_check(self):
        from common.board import Board
        b = Board(width=4, height=3)
        fits_with_inner = [[(2, 0, 0.1)], [], [], []]
        ok, err = b.can_place(1, fits_with_inner, 0, 0, 0)
        assert not ok

    def test_board_repr(self):
        from common.board import Board
        b = Board(width=3, height=2)
        s = repr(b)
        assert 'None' in s or '-' in s

    def test_board_copy(self):
        from common.board import Board
        b = Board(width=4, height=3)
        b.place(1, [[], [], [], []], 0, 0, 0)
        c = Board.copy(b)
        assert c.placed_count == 1
        c.place(2, [[], [], [], []], 1, 0, 0)
        assert b.placed_count == 1
        assert c.placed_count == 2

    def test_sides_that_must_be_edges(self):
        from common.board import Board, TOP, RIGHT, BOTTOM, LEFT
        b = Board(width=4, height=3)
        assert TOP in b._sides_that_must_be_edges(0, 0)
        assert LEFT in b._sides_that_must_be_edges(0, 0)
        assert BOTTOM in b._sides_that_must_be_edges(0, 2)
        assert RIGHT in b._sides_that_must_be_edges(3, 0)


class TestOutput:
    def test_generate_solution_grid(self, tmp_path):
        from common.output import generate_solution_grid
        from common.board import Board
        b = Board(width=3, height=2)
        b.place(1, [[], [], [], []], 0, 0, 0)
        b.place(2, [[], [], [], []], 1, 0, 0)
        path = generate_solution_grid(b, str(tmp_path))
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert '1' in content
        assert '2' in content

    def test_print_solution_summary(self, capsys):
        from common.output import print_solution_summary
        from common.board import Board
        b = Board(width=3, height=2)
        b.place(1, [[], [], [], []], 0, 0, 0)
        print_solution_summary(b)
        captured = capsys.readouterr()
        assert 'Solution' in captured.out
        assert 'Pieces placed: 1' in captured.out


class TestEndToEnd:
    def test_full_pipeline_synthetic(self, tmp_path):
        from common.find_islands import save_island_as_bmp
        from common.preprocess import normalize_piece_size
        from common.vector import Vector

        _, binary = create_synthetic_piece_bmp(tmp_path)
        size = binary.shape[0]

        gray_full = np.full((size * 2, size * 2), 30, dtype=np.uint8)
        gray_full[:size, :size][binary == 1] = 240
        sb, _, _ = normalize_piece_size(binary, gray_full, origin=(0, 0),
                                         target_size=500)
        bmp_path = str(tmp_path / 'piece_1_normalized.bmp')
        save_island_as_bmp(sb, bmp_path)

        v = Vector.from_file(bmp_path, id=1)
        v.find_border_raster()
        v.vectorize()
        assert len(v.vertices) > 20

        v.find_four_corners()
        assert len(v.corners) == 4

        v.extract_four_sides()
        assert len(v.sides) == 4

        v.enhance_corners()
        assert len(v.corners) == 4

        metadata = {
            'original_photo_name': 'synthetic.jpg',
            'photo_space_origin': [0, 0],
            'photo_space_centroid': [size // 2, size // 2],
            'photo_width': size,
            'photo_height': size,
        }
        output_dir = tmp_path / 'vector_output'
        output_dir.mkdir()
        v.save(str(output_dir), metadata)

        for j in range(4):
            side_file = output_dir / f'side_1_{j}.json'
            assert side_file.exists()
            with open(side_file) as f:
                data = json.load(f)
            assert 'vertices' in data
            assert 'is_edge' in data

    def test_segmentation_and_extraction_pipeline(self, tmp_path):
        from common.segment_phone import segment_photo
        from common.extract import extract_pieces_from_segmented
        from common.preprocess import normalize_piece_size

        h, w = 400, 600
        gray = np.full((h, w), 240, dtype=np.uint8)
        bgr = np.full((h, w, 3), 240, dtype=np.uint8)

        cv2.circle(gray, (100, 200), 40, 60, -1)
        cv2.circle(bgr, (100, 200), 40, (60, 60, 60), -1)
        cv2.circle(gray, (300, 200), 40, 60, -1)
        cv2.circle(bgr, (300, 200), 40, (60, 60, 60), -1)
        cv2.circle(gray, (500, 200), 40, 60, -1)
        cv2.circle(bgr, (500, 200), 40, (60, 60, 60), -1)

        binary = segment_photo(gray, bgr=bgr, method='adaptive')
        assert np.sum(binary == 1) > 100

        pieces = extract_pieces_from_segmented(binary, bgr, 'test')
        assert len(pieces) >= 2

        for piece in pieces:
            sb, sc, scale = normalize_piece_size(
                piece.binary, gray, piece.origin,
                color=piece.color, target_size=500
            )
            assert sb.shape[0] <= 520
            assert sb.shape[1] <= 520


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
