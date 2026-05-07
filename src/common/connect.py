import os
import json
import math
import multiprocessing

import numpy as np

from common import pieces, sides
from common import util


def build(input_path, output_path):
    print("> Loading piece data (raw)...")
    ps_raw = pieces.Piece.load_all(input_path, resample=False)
    print("\t ...Loaded %d pieces (raw)" % len(ps_raw))

    print("> Loading piece data (resampled)...")
    ps_resampled = pieces.Piece.load_all(input_path, resample=True)
    print("\t ...Loaded %d pieces (resampled)" % len(ps_resampled))

    n_workers = min(os.cpu_count() or 1, 8)
    piece_ids = sorted(ps_raw.keys())

    args_list = [
        (ps_raw, ps_resampled, piece_id)
        for piece_id in piece_ids
    ]

    print("> Building connectivity with %d workers..." % n_workers)
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.starmap(_find_potential_matches_for_piece, args_list)

    ps_out = {}
    for piece_id, fits in results:
        piece = ps_raw[piece_id]
        piece.fits = fits
        ps_out[piece_id] = piece

    return _save(ps_out, output_path)


def _can_potentially_match(piece_a, si, piece_b, sj):
    side_a = piece_a.sides[si]
    side_b = piece_b.sides[sj]

    if side_a.is_edge or side_b.is_edge:
        return False

    len_a = side_a.original_length
    len_b = side_b.original_length
    if len_a < 1 or len_b < 1:
        return False
    d_scale = 1.0 - (len_a / len_b)
    if abs(d_scale) > sides.SIDE_MAX_LENGTH_DISCREPANCY:
        return False

    if side_a.is_convex is not None and side_b.is_convex is not None:
        if side_a.is_convex == side_b.is_convex:
            return False

    sd_a = side_a.center_side_sign()
    sd_b = side_b.center_side_sign()
    if sd_a != 0 and sd_b != 0 and sd_a * sd_b < 0:
        return False

    rot_for_b = side_a.original_angle + math.pi - side_b.original_angle
    adj_map = {
        (si - 1) % 4: (sj + 1) % 4,
        (si + 1) % 4: (sj - 1) % 4,
    }
    for adj_a_si, adj_b_si in adj_map.items():
        adj_a = piece_a.sides[adj_a_si]
        adj_b = piece_b.sides[adj_b_si]

        if adj_a.is_edge != adj_b.is_edge:
            return False

        if adj_a.is_edge and adj_b.is_edge:
            adj_b_rotated = adj_b.original_angle + rot_for_b
            angle_diff = util.compare_angles(
                adj_a.original_angle, adj_b_rotated
            )
            if angle_diff > sides.EDGE_PARALLEL_THRESHOLD_RAD:
                return False

    return True


def _find_potential_matches_for_piece(ps_raw, ps_resampled, piece_id):
    piece_raw = ps_raw[piece_id]
    fits = [[], [], [], []]

    for si in range(4):
        side_raw = piece_raw.sides[si]
        if side_raw.is_edge:
            continue

        side_res = ps_resampled[piece_id].sides[si]

        for other_pid in ps_raw:
            if other_pid == piece_id:
                continue
            other_raw = ps_raw[other_pid]

            for sj in range(4):
                other_side_raw = other_raw.sides[sj]
                if other_side_raw.is_edge:
                    continue

                if not _can_potentially_match(piece_raw, si, other_raw, sj):
                    continue

                other_side_res = ps_resampled[other_pid].sides[sj]
                error, shift = side_res.error_when_fit_with(
                    other_side_res,
                    flip=True,
                    skip_edges=False,
                )

                if error > sides.SIDE_MAX_ERROR_TO_MATCH:
                    continue

                len_diff = abs(1.0 - (side_raw.original_length / other_side_raw.original_length))

                fits[si].append({
                    'pid': other_pid,
                    'si': sj,
                    'error': error,
                    'len_diff': len_diff,
                    'shift_x': float(shift[0]),
                    'shift_y': float(shift[1]),
                    'convex_a': side_raw.is_convex,
                    'convex_b': other_side_raw.is_convex,
                })

        fits[si] = sorted(fits[si], key=lambda x: x['error'])
        if fits[si]:
            print(f"Piece {piece_id}[{si}] has {len(fits[si])} matches, best: {fits[si][0]['error']:.4f}")
        else:
            if not piece_raw.sides[si].is_edge:
                print(f"Warning: Piece {piece_id}[{si}] has no matches (not an edge)")

    return (piece_id, fits)


def _save(pieces_dict, out_directory):
    out = {p_id: p.to_dict() for (p_id, p) in pieces_dict.items()}
    path = os.path.join(out_directory, 'connectivity.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    return out
