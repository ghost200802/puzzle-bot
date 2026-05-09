import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, CONNECTIVITY_DIR, SOLUTION_DIR, VECTOR_DIR

_PROJECT_ROOT = os.path.join(_here, '..')

OUTPUT_ROOT = os.environ.get('PUZZLE_OUTPUT_ROOT', '')


def set_output_root(path):
    global OUTPUT_ROOT
    OUTPUT_ROOT = os.path.abspath(path)
    os.environ['PUZZLE_OUTPUT_ROOT'] = OUTPUT_ROOT


def get_output_dir(output_name=None):
    if output_name:
        return os.path.join(_PROJECT_ROOT, 'output', output_name)
    if OUTPUT_ROOT:
        return OUTPUT_ROOT
    return os.path.join(_PROJECT_ROOT, 'output', 'puzzle_new')


def get_deduped_path(output_name=None):
    return os.path.join(get_output_dir(output_name), DEDUPED_DIR)


def get_connectivity_path(output_name=None):
    return os.path.join(get_output_dir(output_name), CONNECTIVITY_DIR)


def get_solution_path(output_name=None):
    return os.path.join(get_output_dir(output_name), SOLUTION_DIR)


def get_vector_path(output_name=None):
    return os.path.join(get_output_dir(output_name), VECTOR_DIR)


def get_check_path(output_name=None):
    return os.path.join(get_output_dir(output_name), 'check')


def get_color_path(output_name=None):
    return os.path.join(get_output_dir(output_name), '2_piece_colors')
