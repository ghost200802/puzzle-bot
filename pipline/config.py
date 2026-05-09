import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR, VECTOR_DIR

_PROJECT_ROOT = os.path.join(_here, '..')


def get_output_dir(output_name=None):
    name = output_name or 'puzzle_new'
    return os.path.join(_PROJECT_ROOT, 'output', name)


def get_deduped_path(output_name=None):
    return os.path.join(get_output_dir(output_name), DEDUPED_DIR)


def get_connectivity_path(output_name=None):
    return os.path.join(get_output_dir(output_name), CONNECTIVITY_DIR)


def get_solution_path(output_name=None):
    return os.path.join(get_output_dir(output_name), SOLUTION_DIR)


def get_check_path(output_name=None):
    return os.path.join(get_output_dir(output_name), 'check')


def get_color_path(output_name=None):
    return os.path.join(get_output_dir(output_name), '2_piece_colors')


def get_vector_path(output_name=None):
    return os.path.join(get_output_dir(output_name), VECTOR_DIR)
