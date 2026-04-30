import os
import json
import numpy as np
import math
from glob import glob
import shutil
import itertools
from pathlib import Path

from common import util, sides
from common.config import *


DOUBLE_CHECK_GEOMETRIC_DUPLICATE_THRESHOLD = 2.2
SIDE_MISALIGNMENT_RAD = 12.0 * math.pi / 180


def deduplicate(batch_data_path, input_path, output_path):
    """
    Removes duplicate vector pieces by only copying over unique pieces to the output directory
    Algorithm finds pieces whose centroids are in the same physical space (using "motor space")
    and then compares the geometry of the pieces as a double-check / a way to flag that computer vision problems might have occurred
    """
    # Dispatch to phone mode dedup if applicable
    if MODE == 'phone':
        return deduplicate_phone(input_path, output_path)

    # open up all the pieces
    print(f"Loading piece data from {input_path}...")
    pieces = {}
    piece_photo_locations = {}
    input_path = Path(input_path)
    for path in input_path.glob("side_*_0.json"):
        i = int(path.parts[-1].split('_')[1])
        piece = []
        for j in range(4):
            json_path = input_path.joinpath(f'side_{i}_{j}.json')
            with open(json_path) as f:
                data = json.load(f)
                side = sides.Side(i, j, data['vertices'], piece_center=data['piece_center'],
                                  is_edge=data['is_edge'], resample=True, rotate=False,
                                  photo_filename=data['original_photo_name'])
                piece.append(side)

                # we'll also want to know where in the photo frame this piece was
                piece_photo_locations[i] = {
                    'photo_width': data['photo_width'],
                    'photo_height': data['photo_height'],
                    # we use centroids because they are generally quite stable between different photos of identical pieces
                    'photo_space_centroid': data['photo_space_centroid'],
                }
        pieces[i] = piece

    # open the metadata that tells us where each piece was photographed
    with open(batch_data_path) as f:
        batch_data_d = json.load(f)["photos"]
    batch_data = {}
    for d in batch_data_d:
        batch_data[d["file_name"]] = d["position"]

    uniques = set()
    dupes = set()

    def _photo_space_to_robot_space(robot_space_camera_position, photo_space_position):
        return (robot_space_camera_position[0] + (photo_space_position[0] * APPROX_ROBOT_COUNTS_PER_PIXEL),
                robot_space_camera_position[1] - (photo_space_position[1] * APPROX_ROBOT_COUNTS_PER_PIXEL))

    for i, sides0 in pieces.items():
        # if this piece is duplicating someone else, skip it
        if i in dupes:
            continue

        dupes_of_i = {}
        piece_i_photo_filename = sides0[0].photo_filename
        piece_i_robot_space_camera_position = batch_data[piece_i_photo_filename]
        piece_i_motor_space_centroid = _photo_space_to_robot_space(piece_i_robot_space_camera_position, piece_photo_locations[i]['photo_space_centroid'])

        for j, sides1 in pieces.items():
            if i == j or j in dupes:
                continue

            # duplicates are pieces in the same physical location
            piece_j_photo_filename = sides1[0].photo_filename
            piece_j_robot_space_camera_position = batch_data[piece_j_photo_filename]
            piece_j_motor_space_centroid = _photo_space_to_robot_space(piece_j_robot_space_camera_position, piece_photo_locations[j]['photo_space_centroid'])

            motor_space_centroid_distance = util.distance(piece_i_motor_space_centroid, piece_j_motor_space_centroid)
            pixel_distance = motor_space_centroid_distance / APPROX_ROBOT_COUNTS_PER_PIXEL
            if pixel_distance < DUPLICATE_CENTROID_DELTA_PX:
                print(f"[{i}]\t is duplicated by {j} \t Centroid distance: {pixel_distance}")
                dupes_of_i[j] = piece_photo_locations[j]

                # just for fun, let's compare geometries
                score = _compare(sides0, sides1)
                if score > DOUBLE_CHECK_GEOMETRIC_DUPLICATE_THRESHOLD:
                    print(f"[{i}]\t is in the same position as {j} but they don't seem to match. This is usually a problem... \t Geometric Similarity: {score}")
            elif pixel_distance < 3 * DUPLICATE_CENTROID_DELTA_PX:
                print(f"\t\t\t[{i}]\t is similar to {j} \t Centroid distance: {pixel_distance}")

        if len(dupes_of_i) == 0:
            # if this piece was truly unique, keep it
            uniques.add(i)
        else:
            # if this piece has duplciates, of all the duplicates, find the "best" one
            dupes_of_i[i] = piece_photo_locations[i]
            best_dupe_id = _pick_best_dupe(dupes_of_i)
            uniques.add(best_dupe_id)
            for j, _ in dupes_of_i.items():
                if j != best_dupe_id:
                    dupes.add(j)

    print(f"Started with {len(pieces)}; found {len(dupes)} duplicate pieces; resulting in {len(uniques)} unique pieces.")

    # finally, copy all the uniques to the output directory
    for id in uniques:
        # copy the json files
        for i in range(4):
            side_i = f'side_{id}_{i}.json'
            shutil.copyfile(os.path.join(input_path, side_i), os.path.join(output_path, side_i))
        # copy the vector file as well
        vector_filename = glob(f"{id}_*.svg", root_dir=input_path)[0] # Take the 0th element, there should be exactly 1
        input_vector_file = os.path.join(input_path, vector_filename)
        output_vector_file = os.path.join(output_path, vector_filename)
        shutil.copyfile(input_vector_file, output_vector_file)

    return len(uniques)


def _pick_best_dupe(pieces):
    """
    Given a dict of piece_ids :=> piece metadata dicts, pick the best one to keep
    by finding the one that was closest to the center of the photograph
    We have highest telecentricity and image quality in the center of the frame
    """
    # grab any element to get the center coordinate of any photo (half the width and height of the photo)
    any_id = next(iter(pieces))
    any_metadata = pieces[any_id]
    center = (any_metadata['photo_width']/2, any_metadata['photo_height']/2)

    # find which ID is closest to the center of the photo
    best_id = None
    best_score = 1000000
    for id, metadata in pieces.items():
        piece_center = metadata['photo_space_centroid']
        score = util.distance(center, piece_center)
        if score < best_score:
            best_id = id
            best_score = score

    return best_id


def _compare(sides0, sides1):
    """
    Compare this piece to another piece, returning a score of how similar they are
    0 = no error
    higher = more error
    Note: we cannot assume that sides0[0] is the same as sides1[0] - they might be in different indices
    """

    side00, side01, side02, side03 = sides0
    permutations = [
        [side00, side01, side02, side03],
        [side01, side02, side03, side00],
        [side02, side03, side00, side01],
        [side03, side00, side01, side02]
    ]

    min_cumulative_error = 1000
    for sides0 in permutations:
        # first check to see if the pieces are in approximately the same orientation
        # we expect no rotation between duplicates
        sides_aligned = True
        for i in range(4):
            angle_diff = util.compare_angles(sides0[i].angle, sides1[i].angle)
            if angle_diff > SIDE_MISALIGNMENT_RAD:
                sides_aligned = False
                break
        if not sides_aligned:
            continue

        # if the sides are in approximately the same orientation, see how close to perfect matches they are
        cumulative_error = 0
        for i in range(4):
            cumulative_error += sides0[i].error_when_fit_with(sides1[i], flip=False, skip_edges=False, render=False)

        if cumulative_error < min_cumulative_error:
            min_cumulative_error = cumulative_error

    return min_cumulative_error


# --- Phone mode deduplication ---

def compute_piece_hash(binary_image, color_image=None):
    """
    Compute perceptual hashes for a puzzle piece.
    Returns (binary_phash, color_phash).
    """
    from PIL import Image as PILImage
    import imagehash

    vis = (binary_image * 255).astype(np.uint8)
    pil_gray = PILImage.fromarray(vis, mode='L')
    binary_hash = imagehash.phash(pil_gray)

    color_hash = None
    if color_image is not None:
        rgb = color_image[:, :, ::-1].copy()
        pil_color = PILImage.fromarray(rgb, mode='RGB')
        color_hash = imagehash.phash(pil_color)

    return binary_hash, color_hash


def find_duplicate_candidates(piece_hashes, hash_threshold=15):
    """
    Coarse dedup: find pairs whose perceptual hash distance < threshold.
    piece_hashes: dict {piece_id: (binary_hash, color_hash)}
    Returns list of (id_a, id_b, hash_distance).
    """
    candidates = []
    ids = list(piece_hashes.keys())
    for i, j in itertools.combinations(range(len(ids)), 2):
        id_a, id_b = ids[i], ids[j]
        bh_a, ch_a = piece_hashes[id_a]
        bh_b, ch_b = piece_hashes[id_b]
        dist = bh_a - bh_b
        if ch_a is not None and ch_b is not None:
            dist = min(dist, ch_a - ch_b)
        if dist < hash_threshold:
            candidates.append((id_a, id_b, dist))
    return candidates


def deduplicate_phone(input_path, output_path):
    """
    Phone mode dedup: geometric verification.
    input_path: directory containing side_*.json files
    output_path: directory to copy unique pieces to
    Returns number of unique pieces.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all pieces
    pieces_data = {}
    for path in input_path.glob("side_*_0.json"):
        i = int(path.parts[-1].split('_')[1])
        piece_sides = []
        valid = True
        for j in range(4):
            json_path = input_path.joinpath(f'side_{i}_{j}.json')
            try:
                with open(json_path) as f:
                    data = json.load(f)
                    if not data.get('vertices'):
                        valid = False
                        break
                    side = sides.Side(
                        i, j, data['vertices'],
                        piece_center=data['piece_center'],
                        is_edge=data['is_edge'],
                        resample=True, rotate=False,
                        photo_filename=data.get('original_photo_name', ''),
                    )
                    piece_sides.append(side)
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"  Warning: skipping piece {i} side {j}: {e}")
                valid = False
                break
        if valid and len(piece_sides) == 4:
            pieces_data[i] = piece_sides

    if len(pieces_data) <= 1:
        for pid in pieces_data:
            _copy_piece_files(str(input_path), str(output_path), pid)
        return len(pieces_data)

    # Find duplicate groups via geometric comparison
    dup_candidates = []
    ids = list(pieces_data.keys())
    for i, j in itertools.combinations(range(len(ids)), 2):
        id_a, id_b = ids[i], ids[j]
        score = _compare(pieces_data[id_a], pieces_data[id_b])
        if score < PHONE_DUPLICATE_GEOMETRIC_THRESHOLD:
            dup_candidates.append((id_a, id_b, score))

    # Union-Find to group duplicates
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for id_a, id_b, _ in dup_candidates:
        union(id_a, id_b)

    # Pick one representative per group
    groups = {}
    for pid in pieces_data:
        root = find(pid)
        groups.setdefault(root, []).append(pid)

    uniques = set()
    dupes = set()
    for root, members in groups.items():
        # pick the first member as representative
        best = members[0]
        uniques.add(best)
        for m in members:
            if m != best:
                dupes.add(m)

    print(f"Dedup: started with {len(pieces_data)}, found {len(dupes)} duplicates, "
          f"resulting in {len(uniques)} unique pieces.")

    for pid in uniques:
        _copy_piece_files(str(input_path), str(output_path), pid)

    return len(uniques)


def _copy_piece_files(input_path, output_path, pid):
    """Copy all side JSON and SVG files for a piece."""
    for i in range(4):
        side_file = f'side_{pid}_{i}.json'
        src = os.path.join(input_path, side_file)
        dst = os.path.join(output_path, side_file)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
    svg_files = glob(f"{pid}_*.svg", root_dir=input_path)
    for svg in svg_files:
        src = os.path.join(input_path, svg)
        dst = os.path.join(output_path, svg)
        shutil.copyfile(src, dst)


def pick_best(piece_data_list):
    """
    Pick the best piece from a list of duplicate candidates.
    Scoring: completeness > resolution.
    Used by phone mode dedup.
    """
    def _score(d):
        s = 0
        if d.get('is_complete', True):
            s += 1000000
        s += d.get('pixel_count', 0)
        return s
    return max(piece_data_list, key=_score)
