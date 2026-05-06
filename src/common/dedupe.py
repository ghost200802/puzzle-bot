import os
import json
import numpy as np
import math
from glob import glob
import shutil
import itertools
from pathlib import Path
import cv2

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
    Phone mode dedup: two-stage strategy.

    Stage 1 (coarse): perceptual hash filtering to find candidate pairs
    Stage 2 (fine): geometric verification on candidates

    Groups duplicates via Union-Find and selects best representative
    per group. Optionally fuses multi-view data for improved quality.

    input_path: directory containing side_*.json files
    output_path: directory to copy unique pieces to
    Returns number of unique pieces.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    pieces_data = {}
    piece_origins = {}
    piece_metadata = {}
    for path in input_path.glob("side_*_0.json"):
        i = int(path.parts[-1].split('_')[1])
        piece_sides = []
        valid = True
        origin = None
        meta = {}
        for j in range(4):
            json_path = input_path.joinpath(f'side_{i}_{j}.json')
            try:
                with open(json_path) as f:
                    data = json.load(f)
                    if not data.get('vertices'):
                        valid = False
                        break
                    if j == 0:
                        origin = data.get('photo_space_origin')
                        meta['is_complete'] = data.get('is_complete', True)
                        meta['original_photo_name'] = data.get(
                            'original_photo_name', ''
                        )
                        meta['photo_width'] = data.get('photo_width', 500)
                        meta['photo_height'] = data.get('photo_height', 500)
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
            if origin:
                piece_origins[i] = origin
            piece_metadata[i] = meta

    if len(pieces_data) <= 1:
        for pid in pieces_data:
            _copy_piece_files(str(input_path), str(output_path), pid)
        return len(pieces_data)

    has_origins = len(piece_origins) == len(pieces_data)

    dup_candidates = []
    ids = list(pieces_data.keys())
    skipped_by_origin = 0

    try:
        piece_hashes = {}
        for pid in ids:
            vis = np.zeros((100, 100), dtype=np.uint8)
            for side in pieces_data[pid]:
                verts = side.vertices.astype(np.int32)
                cv2.fillPoly(vis, [verts], 255)
            try:
                bh, ch = compute_piece_hash(
                    (vis > 127).astype(np.uint8)
                )
                piece_hashes[pid] = (bh, ch)
            except Exception:
                piece_hashes[pid] = (None, None)

        hash_threshold = PHONE_HASH_THRESHOLD
        hash_candidates = find_duplicate_candidates(
            piece_hashes, hash_threshold
        )
        hash_candidate_set = set()
        for id_a, id_b, _ in hash_candidates:
            hash_candidate_set.add((min(id_a, id_b), max(id_a, id_b)))

        print(f"Dedup stage 1 (hash): {len(hash_candidates)} candidate pairs "
              f"from {len(ids)} pieces")

        for i, j in itertools.combinations(range(len(ids)), 2):
            id_a, id_b = ids[i], ids[j]

            if has_origins and id_a in piece_origins and id_b in piece_origins:
                oa, ob = piece_origins[id_a], piece_origins[id_b]
                dist = math.sqrt((oa[0] - ob[0])**2 + (oa[1] - ob[1])**2)
                if dist > DUPLICATE_CENTROID_DELTA_PX:
                    skipped_by_origin += 1
                    continue

            pair_key = (min(id_a, id_b), max(id_a, id_b))

            if pair_key in hash_candidate_set:
                score = _compare(pieces_data[id_a], pieces_data[id_b])
                if score < PHONE_DUPLICATE_GEOMETRIC_THRESHOLD:
                    dup_candidates.append((id_a, id_b, score))
            else:
                bh_a, ch_a = piece_hashes.get(id_a, (None, None))
                bh_b, ch_b = piece_hashes.get(id_b, (None, None))
                if bh_a is None or bh_b is None:
                    score = _compare(pieces_data[id_a], pieces_data[id_b])
                    if score < PHONE_DUPLICATE_GEOMETRIC_THRESHOLD:
                        dup_candidates.append((id_a, id_b, score))

        print(f"Dedup stage 2 (geometric): {len(dup_candidates)} confirmed duplicates")
    except ImportError:
        print("  imagehash not available, falling back to geometric-only dedup")
        for i, j in itertools.combinations(range(len(ids)), 2):
            id_a, id_b = ids[i], ids[j]

            if has_origins and id_a in piece_origins and id_b in piece_origins:
                oa, ob = piece_origins[id_a], piece_origins[id_b]
                dist = math.sqrt((oa[0] - ob[0])**2 + (oa[1] - ob[1])**2)
                if dist > DUPLICATE_CENTROID_DELTA_PX:
                    skipped_by_origin += 1
                    continue

            score = _compare(pieces_data[id_a], pieces_data[id_b])
            if score < PHONE_DUPLICATE_GEOMETRIC_THRESHOLD:
                dup_candidates.append((id_a, id_b, score))

    if has_origins:
        total_pairs = len(ids) * (len(ids) - 1) // 2
        print(f"Dedup: origin filter skipped {skipped_by_origin}/{total_pairs} pairs")

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

    groups = {}
    for pid in pieces_data:
        root = find(pid)
        groups.setdefault(root, []).append(pid)

    uniques = set()
    dupes = set()
    for root, members in groups.items():
        if len(members) == 1:
            uniques.add(members[0])
            continue

        member_metas = [
            piece_metadata.get(m, {'is_complete': True, 'pixel_count': 0})
            for m in members
        ]
        best = pick_best(member_metas)
        best_idx = member_metas.index(best)
        best_id = members[best_idx]
        uniques.add(best_id)

        if len(members) > 1:
            _fuse_multi_view(members, pieces_data, input_path, output_path,
                             best_id)

        for m in members:
            if m != best_id:
                dupes.add(m)

    print(f"Dedup: started with {len(pieces_data)}, found {len(dupes)} duplicates, "
          f"resulting in {len(uniques)} unique pieces.")

    for pid in uniques:
        if not os.path.exists(os.path.join(output_path, f'side_{pid}_0.json')):
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


def _fuse_multi_view(members, pieces_data, input_path, output_path, best_id):
    """
    Fuse multi-view data from duplicate group members into the best piece.

    Strategy:
      - Average vertex positions from all views for improved edge accuracy
      - Merge color features if available
      - Write fused data to output_path

    Args:
        members: list of piece IDs in the duplicate group
        pieces_data: dict {piece_id: [Side, ...]}
        input_path: source directory
        output_path: destination directory
        best_id: the chosen representative piece ID
    """
    if len(members) < 2:
        return

    best_sides = pieces_data[best_id]

    fused_sides = []
    for si in range(4):
        all_vertices = []
        for pid in members:
            if si < len(pieces_data[pid]):
                all_vertices.append(pieces_data[pid][si].vertices)

        if len(all_vertices) <= 1:
            fused_sides.append(best_sides[si])
            continue

        best_verts = best_sides[si].vertices
        if len(best_verts) != all_vertices[0].shape[0]:
            fused_sides.append(best_sides[si])
            continue

        all_verts_stacked = np.stack(all_vertices, axis=0)
        avg_verts = np.mean(all_verts_stacked, axis=0)

        max_diff = np.max(np.abs(avg_verts - best_verts))
        if max_diff < 0.5:
            fused_sides.append(best_sides[si])
        else:
            fused_side = sides.Side(
                best_id, si, avg_verts,
                piece_center=best_sides[si].piece_center,
                is_edge=best_sides[si].is_edge,
                resample=True, rotate=False,
                photo_filename=best_sides[si].photo_filename,
            )
            fused_sides.append(fused_side)

    if fused_sides != best_sides:
        for j, side in enumerate(fused_sides):
            side_file = os.path.join(output_path, f'side_{best_id}_{j}.json')
            with open(side_file, 'w') as f:
                json.dump({
                    'vertices': side.vertices.tolist(),
                    'piece_center': side.piece_center,
                    'is_edge': side.is_edge,
                    'photo_space_origin': (0, 0),
                    'photo_space_centroid': [0, 0],
                    'photo_width': 500,
                    'photo_height': 500,
                    'original_photo_name': side.photo_filename,
                }, f)
        print(f"  Fused {len(members)} views for piece {best_id}")

    color_features_files = []
    for pid in members:
        cf_path = os.path.join(
            str(input_path), f'color_features_{pid}.json'
        )
        if os.path.exists(cf_path):
            color_features_files.append(cf_path)

    if len(color_features_files) >= 2:
        try:
            all_features = []
            for cf_path in color_features_files:
                with open(cf_path) as f:
                    all_features.append(json.load(f))

            fused_features = {}
            for key in all_features[0]:
                values = [feat[key] for feat in all_features
                          if key in feat and isinstance(feat[key], (int, float))]
                if values:
                    fused_features[key] = sum(values) / len(values)
                elif key in all_features[0]:
                    fused_features[key] = all_features[0][key]

            out_cf = os.path.join(
                output_path, f'color_features_{best_id}.json'
            )
            with open(out_cf, 'w') as f:
                json.dump(fused_features, f)
        except Exception as e:
            print(f"  Color feature fusion failed for piece {best_id}: {e}")
