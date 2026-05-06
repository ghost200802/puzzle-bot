import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tests.helpers import create_synthetic_piece_bmp

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')


@pytest.fixture
def src_path():
    return os.path.join(PROJECT_ROOT, 'src')


@pytest.fixture
def sample_photo_dir():
    photo_dir = os.path.join(PROJECT_ROOT, '..', 'example_data', '0_photos')
    if not os.path.isdir(photo_dir):
        pytest.skip("example_data/0_photos not found")
    return photo_dir


@pytest.fixture
def synthetic_piece_bmp(tmp_path):
    return create_synthetic_piece_bmp(tmp_path)


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / 'output'
    d.mkdir()
    return str(d)
