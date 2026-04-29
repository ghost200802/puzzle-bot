"""
Tests for Milestone 4: Robustness and usability
"""

import os
import sys
import json
import time

import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestProgressBar:
    def test_progress_bar_completion(self, capsys):
        from common.pipeline_utils import ProgressBar
        pb = ProgressBar(total=10, prefix='Test', bar_length=20)
        for i in range(10):
            pb.update()
        captured = capsys.readouterr()
        assert '100.0%' in captured.out

    def test_progress_bar_with_current(self, capsys):
        from common.pipeline_utils import ProgressBar
        pb = ProgressBar(total=5, bar_length=10)
        pb.update(current=5)
        captured = capsys.readouterr()
        assert '100.0%' in captured.out

    def test_progress_bar_zero_total(self):
        from common.pipeline_utils import ProgressBar
        pb = ProgressBar(total=0)
        pb.update()


class TestRetry:
    def test_retry_success_first_try(self):
        from common.pipeline_utils import retry
        call_count = 0
        @retry(max_retries=3, delay=0.01)
        def good_func():
            nonlocal call_count
            call_count += 1
            return 'success'
        result = good_func()
        assert result == 'success'
        assert call_count == 1

    def test_retry_success_after_failures(self):
        from common.pipeline_utils import retry
        call_count = 0
        @retry(max_retries=3, delay=0.01, exceptions=(ValueError,))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError('not yet')
            return 'success'
        result = flaky_func()
        assert result == 'success'
        assert call_count == 3

    def test_retry_exhausted(self):
        from common.pipeline_utils import retry
        @retry(max_retries=2, delay=0.01, exceptions=(ValueError,))
        def always_fail():
            raise ValueError('nope')
        with pytest.raises(ValueError):
            always_fail()


class TestSafeExecute:
    def test_safe_execute_success(self):
        from common.pipeline_utils import safe_execute
        result = safe_execute(lambda: 42)
        assert result == 42

    def test_safe_execute_failure(self):
        from common.pipeline_utils import safe_execute
        result = safe_execute(lambda: 1 / 0, default=-1, log_errors=False)
        assert result == -1


class TestPipelineCheckpoint:
    def test_save_and_load(self, tmp_path):
        from common.pipeline_utils import PipelineCheckpoint
        cp = PipelineCheckpoint(str(tmp_path))
        cp.save(step=3, data={'pieces': 10})
        step, data = cp.load()
        assert step == 3
        assert data['pieces'] == 10

    def test_load_no_checkpoint(self, tmp_path):
        from common.pipeline_utils import PipelineCheckpoint
        cp = PipelineCheckpoint(str(tmp_path))
        step, data = cp.load()
        assert step == 0
        assert data is None

    def test_clear(self, tmp_path):
        from common.pipeline_utils import PipelineCheckpoint
        cp = PipelineCheckpoint(str(tmp_path))
        cp.save(step=5)
        cp.clear()
        step, _ = cp.load()
        assert step == 0

    def test_is_complete(self, tmp_path):
        from common.pipeline_utils import PipelineCheckpoint
        cp = PipelineCheckpoint(str(tmp_path))
        assert not cp.is_complete(3)
        cp.save(step=3)
        assert cp.is_complete(3)
        assert not cp.is_complete(4)


class TestSegmentationFallback:
    def test_segment_with_fallback_adaptive(self):
        from common.segment_phone import segment_with_fallback
        gray = np.full((400, 600), 240, dtype=np.uint8)
        cv2.circle(gray, (300, 200), 40, 60, -1)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        binary = segment_with_fallback(gray, bgr=bgr, min_piece_pixels=100)
        assert np.sum(binary == 1) > 0

    def test_segment_with_fallback_empty(self):
        from common.segment_phone import segment_with_fallback
        gray = np.full((100, 100), 128, dtype=np.uint8)
        binary = segment_with_fallback(gray, min_piece_pixels=99999)
        assert binary.shape == (100, 100)


class TestAutoResize:
    def test_no_resize_needed(self):
        from common.preprocess import auto_resize_for_processing
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        result, scale = auto_resize_for_processing(img, max_dimension=2000)
        assert scale == 1.0

    def test_resize_large_image(self):
        from common.preprocess import auto_resize_for_processing
        img = np.zeros((4000, 6000, 3), dtype=np.uint8)
        result, scale = auto_resize_for_processing(img, max_dimension=2000)
        assert scale < 1.0
        assert max(result.shape[:2]) <= 2000


class TestImageQuality:
    def test_detect_dark_image(self):
        from common.preprocess import detect_image_quality
        gray = np.full((100, 100), 15, dtype=np.uint8)
        quality = detect_image_quality(gray)
        assert 'too_dark' in quality['issues']

    def test_detect_bright_image(self):
        from common.preprocess import detect_image_quality
        gray = np.full((100, 100), 245, dtype=np.uint8)
        quality = detect_image_quality(gray)
        assert 'too_bright' in quality['issues']

    def test_detect_low_contrast(self):
        from common.preprocess import detect_image_quality
        gray = np.full((100, 100), 128, dtype=np.uint8)
        quality = detect_image_quality(gray)
        assert 'low_contrast' in quality['issues']


class TestImageEnhancement:
    def test_enhance_dark_image(self):
        from common.preprocess import enhance_image
        gray = np.full((100, 100), 20, dtype=np.uint8)
        bgr = np.full((100, 100, 3), 20, dtype=np.uint8)
        enhanced_gray, enhanced_bgr = enhance_image(gray, bgr)
        assert enhanced_gray.mean() > gray.mean()

    def test_enhance_bright_image(self):
        from common.preprocess import enhance_image
        gray = np.full((100, 100), 240, dtype=np.uint8)
        enhanced_gray, _ = enhance_image(gray)
        assert enhanced_gray.mean() < gray.mean()


class TestPieceValidation:
    def test_validate_good_piece(self):
        from common.preprocess import validate_piece
        piece = np.zeros((100, 100), dtype=np.uint8)
        piece[20:80, 20:80] = 1
        is_valid, reason = validate_piece(piece)
        assert is_valid

    def test_validate_too_small(self):
        from common.preprocess import validate_piece
        piece = np.zeros((100, 100), dtype=np.uint8)
        piece[50, 50] = 1
        is_valid, reason = validate_piece(piece, min_pixels=10)
        assert not is_valid

    def test_validate_bad_aspect_ratio(self):
        from common.preprocess import validate_piece
        piece = np.zeros((100, 1000), dtype=np.uint8)
        piece[45:55, :] = 1
        is_valid, reason = validate_piece(piece)
        assert not is_valid

    def test_validate_empty(self):
        from common.preprocess import validate_piece
        piece = np.zeros((100, 100), dtype=np.uint8)
        is_valid, reason = validate_piece(piece)
        assert not is_valid


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
