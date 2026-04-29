import os
import json
from typing import List
import multiprocessing

import numpy as np

from common import pieces, sides
from common.config import MODE


# Building the graph took 440.38 seconds


# Useful for debugging - punch in the piece ids and side ids that should match
# and we'll print extra debug info for these and assert they match
SOLUTION = [
    # [(1, 0), (4, 2)],
    # [(1, 1), (2, 3)],
    # [(1, 2), None],
    # [(1, 3), (10, 2)],
    # [(3, 3), (2, 1)],
]


def build(input_path, output_path, use_image_matching=False):
    """
    Build connectivity graph from piece data.
    When use_image_matching=True, also considers image/color similarity.
    """
    print("> Loading piece data...")
    ps = pieces.Piece.load_all(input_path, resample=True)
    print("\t ...Loaded")

    # Load color feature data for image-based matching (phone mode)
    color_data = {}
    if use_image_matching:
        color_data = _load_color_data(input_path)

    n_workers = min(os.cpu_count() or 1, 8)
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = [
            pool.apply_async(
                _find_potential_matches_for_piece,
                (ps, piece_id, color_data)
            )
            for piece_id in ps.keys()
        ]
        out = [r.get() for r in results]

    ps = { piece_id: piece for (piece_id, piece) in out }
    return _save(ps, output_path)


def _load_color_data(input_path):
    """
    Load color feature data for image-based matching.
    Returns dict {piece_id: {'histogram': ..., 'color_image_path': ...}}
    """
    import pathlib
    data = {}
    vdir = pathlib.Path(input_path)

    for cf_file in vdir.glob('color_features_*.json'):
        pid = int(cf_file.name.split('_')[2].split('.')[0])
        with open(cf_file) as f:
            features = json.load(f)
        data[pid] = features

    for color_file in vdir.glob('color_*.png'):
        pid = int(color_file.name.split('_')[1].split('.')[0])
        if pid not in data:
            data[pid] = {}
        data[pid]['color_image_path'] = str(color_file)

    return data


def _find_potential_matches_for_piece(ps, piece_id, color_data=None, debug=False):
    """
    Find other sides that fit with this piece's sides
    """
    piece = ps[piece_id]

    # for all other piece's sides, find the ones that fit with this piece's sides
    for si, side in enumerate(piece.sides):
        if side.is_edge:
            continue

        for other_piece_id, other_piece in ps.items():
            if other_piece_id == piece_id:
                continue

            for sj, other_side in enumerate(other_piece.sides):
                if other_side.is_edge:
                    continue

                # for debugging, we can optionally provide side-matches from the actual solution and see how well the algo thinks they fit together
                part_of_solution = ([(piece_id, si), (other_piece_id, sj)] in SOLUTION) or ([(other_piece_id, sj), (piece_id, si)] in SOLUTION)

                # compute the error between our piece's side and this other piece's side
                error = side.error_when_fit_with(other_side, render=part_of_solution or debug, debug_str=f'{piece_id}[{si}] vs {other_piece_id}[{sj}]')

                # Optionally adjust error based on color similarity (phone mode)
                if color_data and error <= sides.SIDE_MAX_ERROR_TO_MATCH * 2:
                    color_bonus = _compute_color_match_bonus(
                        piece_id, other_piece_id, si, sj, color_data
                    )
                    adjusted_error = error * (1.0 - color_bonus * 0.3)
                else:
                    adjusted_error = error

                if adjusted_error <= sides.SIDE_MAX_ERROR_TO_MATCH:
                    piece.fits[si].append((other_piece.id, sj, adjusted_error))

                if error > sides.SIDE_MAX_ERROR_TO_MATCH and part_of_solution:
                    raise ValueError(f"Should have matched but didn't: {piece_id}[{si}] vs {other_piece_id}[{sj}]")

        # make sure we have at least one match
        if len(piece.fits[si]) == 0:
            raise Exception(f'Piece {piece_id} side {si} has no matches but is not an edge')

        # sort by error
        piece.fits[si] = sorted(piece.fits[si], key=lambda x: x[2])
        least_error = piece.fits[si][0][2]

        # only keep the best matches
        WORST_MULTIPLIER = 6.0
        piece.fits[si] = [f for f in piece.fits[si] if f[2] <= least_error * WORST_MULTIPLIER]

        print(f"Piece {piece_id}[{si}] has {len(piece.fits[si])} matches, best: {least_error}")
        if debug:
            nth = 8
            if len(piece.fits[si]) > nth:
                nth_match_error = piece.fits[si][nth - 1][2]
                print(f"\t1st match error: {least_error} \t ==> {nth}th match error: {nth_match_error} \t ==> ratio: {nth_match_error / least_error}")

    return (piece_id, piece)


def _compute_color_match_bonus(pid_a, pid_b, side_a, side_b, color_data):
    """
    Compute a color similarity bonus between two pieces.
    Returns a value from 0 to 1.
    """
    data_a = color_data.get(pid_a, {})
    data_b = color_data.get(pid_b, {})

    hist_a = data_a.get('overall_histogram')
    hist_b = data_b.get('overall_histogram')

    if hist_a and hist_b:
        from common.image_match import histogram_similarity
        try:
            sim = 0.0
            count = 0
            for ch in ['h', 's', 'v']:
                ha = np.array(hist_a.get(ch, []), dtype=np.float32).reshape(-1, 1)
                hb = np.array(hist_b.get(ch, []), dtype=np.float32).reshape(-1, 1)
                if ha.size > 0 and hb.size > 0:
                    import cv2
                    s = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
                    sim += s
                    count += 1
            if count > 0:
                return max(0, sim / count)
        except Exception:
            pass

    return 0.0


def _save(pieces, out_directory):
    out = { p_id: p.to_dict() for (p_id, p) in pieces.items() }
    path = os.path.join(out_directory, 'connectivity.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    return out
