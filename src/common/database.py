"""
Puzzle database management module.

Provides the PieceDatabase class for managing the full lifecycle of
puzzle piece data, including:
  - Loading from a vectorized output directory
  - Saving/loading a serialized database
  - Adding new pieces (for realtime mode)
  - Querying piece positions from solutions
  - Incremental connectivity updates and solving
"""

import os
import json
import math
import pathlib
import shutil
from glob import glob

import numpy as np

from common import sides
from common.config import PUZZLE_WIDTH, PUZZLE_HEIGHT


class PieceData:
    """Data container for a single puzzle piece."""

    def __init__(self, piece_id):
        self.id = piece_id
        self.sides = []
        self.color_image = None
        self.binary_image = None
        self.thumbnail = None
        self.color_features = None
        self.descriptors = None
        self.histogram = None
        self.is_edge = False
        self.is_corner = False
        self.photo_source = ''
        self.is_complete = True
        self.hash = None
        self.fits = [[], [], [], []]
        self.solved_position = None

    def to_dict(self):
        d = {
            'id': self.id,
            'is_edge': self.is_edge,
            'is_corner': self.is_corner,
            'photo_source': self.photo_source,
            'is_complete': self.is_complete,
            'solved_position': self.solved_position,
        }
        if self.hash is not None:
            d['hash'] = str(self.hash)
        return d

    def update_edge_info(self):
        edge_count = sum(1 for s in self.sides if s.is_edge)
        self.is_edge = edge_count > 0
        self.is_corner = edge_count > 1


class PieceDatabase:
    """
    Full puzzle database for offline preprocessing and online realtime use.

    Lifecycle:
      1. Offline: load_from_directory() → build_connectivity() → solve()
      2. Save: save() to persist the database
      3. Online: load() → add_piece() / identify_piece()
    """

    def __init__(self, puzzle_width=None, puzzle_height=None):
        self.width = puzzle_width or PUZZLE_WIDTH
        self.height = puzzle_height or PUZZLE_HEIGHT
        self.pieces = {}
        self.connectivity = None
        self.solution = None
        self.target = None
        self.metadata = {
            'expected_count': self.width * self.height,
            'actual_count': 0,
            'missing_count': 0,
            'is_complete': False,
        }
        self._next_id = 1

    def load_from_directory(self, vector_dir, resample=True):
        """
        Load piece data from a vectorized output directory.
        Reads side_*.json, color_*.png, color_features_*.json files.
        Returns the number of pieces loaded.
        """
        from common.image_match import ORBMatcher, compute_color_histogram

        vdir = pathlib.Path(vector_dir)
        orb = ORBMatcher(max_features=300)

        piece_ids = set()
        for f in vdir.glob('side_*_0.json'):
            pid = int(f.name.split('_')[1])
            piece_ids.add(pid)

        for pid in sorted(piece_ids):
            pd = PieceData(pid)

            for j in range(4):
                side_file = vdir / f'side_{pid}_{j}.json'
                if side_file.exists():
                    with open(side_file) as f:
                        data = json.load(f)
                    side = sides.Side(
                        piece_id=pid, side_id=j,
                        vertices=np.array(data['vertices']),
                        piece_center=data['piece_center'],
                        is_edge=data['is_edge'],
                        resample=resample,
                    )
                    pd.sides.append(side)

            if len(pd.sides) == 4:
                pd.update_edge_info()

            color_file = vdir / f'color_{pid}.png'
            if color_file.exists():
                img = np.array(pathlib.Path(str(color_file)).read_bytes())
                import cv2
                img = cv2.imread(str(color_file))
                if img is not None:
                    pd.color_image = img
                    _, des = orb.detect_and_compute(img)
                    pd.descriptors = des
                    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
                    pd.histogram = compute_color_histogram(img, mask)

            feat_file = vdir / f'color_features_{pid}.json'
            if feat_file.exists():
                with open(feat_file) as f:
                    pd.color_features = json.load(f)

            svg_files = glob(f'{pid}_*.svg', root_dir=str(vdir))
            if svg_files:
                pd.photo_source = svg_files[0]

            side0_file = vdir / f'side_{pid}_0.json'
            if side0_file.exists():
                with open(side0_file) as f:
                    meta = json.load(f)
                pd.photo_source = meta.get('original_photo_name', '')
                pd.is_complete = meta.get('is_complete', True)

            self.pieces[pid] = pd

        self._next_id = max(piece_ids, default=0) + 1
        self.metadata['actual_count'] = len(self.pieces)
        self.metadata['missing_count'] = max(
            0, self.metadata['expected_count'] - len(self.pieces)
        )
        self.metadata['is_complete'] = (
            len(self.pieces) >= self.metadata['expected_count']
        )

        print(f"Loaded {len(self.pieces)} pieces from {vector_dir}")
        return len(self.pieces)

    def load_from_solution(self, solution_board):
        """
        Load solution data from a solved Board object.
        Sets self.solution and updates each piece's solved_position.
        """
        self.solution = solution_board
        for y in range(solution_board.height):
            for x in range(solution_board.width):
                cell = solution_board.get(x, y)
                if cell is not None:
                    piece_id, _, orientation = cell
                    if piece_id in self.pieces:
                        self.pieces[piece_id].solved_position = {
                            'x': x, 'y': y, 'rotation': orientation
                        }

    def get_solution_position(self, piece_id):
        """Get the solved position for a piece, or None if unsolved."""
        pd = self.pieces.get(piece_id)
        if pd is not None:
            return pd.solved_position
        return None

    def get_piece_at_position(self, x, y):
        """Get the piece ID at a given board position, or None."""
        for pid, pd in self.pieces.items():
            pos = pd.solved_position
            if pos is not None and pos['x'] == x and pos['y'] == y:
                return pid
        return None

    def next_available_id(self):
        """Return the next available piece ID."""
        while self._next_id in self.pieces:
            self._next_id += 1
        return self._next_id

    def add_piece(self, piece_id, side_data_list, color_image=None,
                  binary_image=None, color_features=None, metadata=None):
        """
        Add a new piece to the database.

        Args:
            piece_id: Unique piece identifier
            side_data_list: List of 4 dicts with 'vertices', 'piece_center', 'is_edge'
            color_image: BGR image of the piece
            binary_image: Binary mask of the piece
            color_features: Dict of color features
            metadata: Dict with 'photo_source', 'is_complete', etc.
        """
        pd = PieceData(piece_id)

        for j, side_data in enumerate(side_data_list):
            side = sides.Side(
                piece_id=piece_id, side_id=j,
                vertices=np.array(side_data['vertices']),
                piece_center=side_data['piece_center'],
                is_edge=side_data['is_edge'],
                resample=True,
            )
            pd.sides.append(side)

        if len(pd.sides) == 4:
            pd.update_edge_info()

        pd.color_image = color_image
        pd.binary_image = binary_image
        pd.color_features = color_features

        if metadata:
            pd.photo_source = metadata.get('photo_source', '')
            pd.is_complete = metadata.get('is_complete', True)

        if color_image is not None:
            from common.image_match import ORBMatcher, compute_color_histogram
            orb = ORBMatcher(max_features=300)
            _, des = orb.detect_and_compute(color_image)
            pd.descriptors = des
            mask = np.ones(color_image.shape[:2], dtype=np.uint8) * 255
            pd.histogram = compute_color_histogram(color_image, mask)

        self.pieces[piece_id] = pd
        self.metadata['actual_count'] = len(self.pieces)
        self.metadata['missing_count'] = max(
            0, self.metadata['expected_count'] - len(self.pieces)
        )

        if piece_id >= self._next_id:
            self._next_id = piece_id + 1

    def remove_piece(self, piece_id):
        """Remove a piece from the database."""
        if piece_id in self.pieces:
            del self.pieces[piece_id]
            self.metadata['actual_count'] = len(self.pieces)
            self.metadata['missing_count'] = max(
                0, self.metadata['expected_count'] - len(self.pieces)
            )

    def save(self, db_dir):
        """
        Save the full database to a directory.

        Directory structure:
          database_dir/
            database_meta.json
            pieces/
              piece_XXX/
                sides.json
                color.png (optional)
                color_features.json (optional)
                binary.png (optional)
            connectivity/
              connectivity.json (optional)
            solution/
              solution.json (optional)
        """
        db_path = pathlib.Path(db_dir)
        db_path.mkdir(parents=True, exist_ok=True)

        pieces_dir = db_path / 'pieces'
        pieces_dir.mkdir(parents=True, exist_ok=True)

        import cv2

        for pid, pd in self.pieces.items():
            piece_dir = pieces_dir / f'piece_{pid}'
            piece_dir.mkdir(parents=True, exist_ok=True)

            sides_data = []
            for s in pd.sides:
                sides_data.append({
                    'vertices': s.vertices.tolist(),
                    'piece_center': s.piece_center,
                    'is_edge': s.is_edge,
                })
            with open(piece_dir / 'sides.json', 'w') as f:
                json.dump(sides_data, f)

            if pd.color_image is not None:
                cv2.imwrite(str(piece_dir / 'color.png'), pd.color_image)
            if pd.binary_image is not None:
                cv2.imwrite(str(piece_dir / 'binary.png'),
                           pd.binary_image * 255)
            if pd.color_features is not None:
                with open(piece_dir / 'color_features.json', 'w') as f:
                    json.dump(pd.color_features, f)
            with open(piece_dir / 'meta.json', 'w') as f:
                json.dump(pd.to_dict(), f)

        if self.connectivity is not None:
            conn_dir = db_path / 'connectivity'
            conn_dir.mkdir(parents=True, exist_ok=True)
            with open(conn_dir / 'connectivity.json', 'w') as f:
                json.dump(self.connectivity, f)

        if self.solution is not None:
            sol_dir = db_path / 'solution'
            sol_dir.mkdir(parents=True, exist_ok=True)
            sol_data = []
            for y in range(self.solution.height):
                for x in range(self.solution.width):
                    cell = self.solution.get(x, y)
                    if cell is not None:
                        pid, _, ori = cell
                        sol_data.append({
                            'piece_id': pid, 'x': x, 'y': y,
                            'orientation': ori
                        })
            with open(sol_dir / 'solution.json', 'w') as f:
                json.dump(sol_data, f)

        self.metadata['puzzle_width'] = self.width
        self.metadata['puzzle_height'] = self.height
        with open(db_path / 'database_meta.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Database saved to {db_dir} ({len(self.pieces)} pieces)")

    @classmethod
    def load(cls, db_dir):
        """
        Load a previously saved database from a directory.
        Returns a PieceDatabase instance.
        """
        import cv2

        db_path = pathlib.Path(db_dir)
        db = cls()

        meta_file = db_path / 'database_meta.json'
        if meta_file.exists():
            with open(meta_file) as f:
                db.metadata = json.load(f)
            db.width = db.metadata.get('puzzle_width', PUZZLE_WIDTH)
            db.height = db.metadata.get('puzzle_height', PUZZLE_HEIGHT)

        pieces_dir = db_path / 'pieces'
        if pieces_dir.exists():
            for piece_dir in sorted(pieces_dir.iterdir()):
                if not piece_dir.is_dir():
                    continue
                pid_str = piece_dir.name.replace('piece_', '')
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue

                pd = PieceData(pid)

                sides_file = piece_dir / 'sides.json'
                if sides_file.exists():
                    with open(sides_file) as f:
                        sides_data = json.load(f)
                    for j, sd in enumerate(sides_data):
                        side = sides.Side(
                            piece_id=pid, side_id=j,
                            vertices=np.array(sd['vertices']),
                            piece_center=sd['piece_center'],
                            is_edge=sd['is_edge'],
                            resample=True,
                        )
                        pd.sides.append(side)

                if len(pd.sides) == 4:
                    pd.update_edge_info()

                color_file = piece_dir / 'color.png'
                if color_file.exists():
                    pd.color_image = cv2.imread(str(color_file))

                binary_file = piece_dir / 'binary.png'
                if binary_file.exists():
                    gray = cv2.imread(str(binary_file), cv2.IMREAD_GRAYSCALE)
                    if gray is not None:
                        pd.binary_image = (gray > 127).astype(np.uint8)

                feat_file = piece_dir / 'color_features.json'
                if feat_file.exists():
                    with open(feat_file) as f:
                        pd.color_features = json.load(f)

                meta_piece_file = piece_dir / 'meta.json'
                if meta_piece_file.exists():
                    with open(meta_piece_file) as f:
                        piece_meta = json.load(f)
                    pd.photo_source = piece_meta.get('photo_source', '')
                    pd.is_complete = piece_meta.get('is_complete', True)
                    pd.solved_position = piece_meta.get('solved_position')

                if pd.color_image is not None:
                    from common.image_match import ORBMatcher, compute_color_histogram
                    orb = ORBMatcher(max_features=300)
                    _, des = orb.detect_and_compute(pd.color_image)
                    pd.descriptors = des
                    mask = np.ones(pd.color_image.shape[:2], dtype=np.uint8) * 255
                    pd.histogram = compute_color_histogram(pd.color_image, mask)

                db.pieces[pid] = pd

        db._next_id = max(db.pieces.keys(), default=0) + 1

        conn_file = db_path / 'connectivity' / 'connectivity.json'
        if conn_file.exists():
            with open(conn_file) as f:
                db.connectivity = json.load(f)

        sol_file = db_path / 'solution' / 'solution.json'
        if sol_file.exists():
            with open(sol_file) as f:
                sol_data = json.load(f)
            from common.board import Board
            board = Board(width=db.width, height=db.height)
            for entry in sol_data:
                pid = entry['piece_id']
                x, y, ori = entry['x'], entry['y'], entry['orientation']
                fits = [[], [], [], []]
                if pid in db.pieces:
                    fits = db.pieces[pid].fits
                board.place(pid, fits, x, y, ori)
            db.solution = board
            for pid, pd in db.pieces.items():
                pos = pd.solved_position
                if pos is None:
                    for entry in sol_data:
                        if entry['piece_id'] == pid:
                            pd.solved_position = {
                                'x': entry['x'], 'y': entry['y'],
                                'rotation': entry['orientation']
                            }
                            break

        db.metadata['actual_count'] = len(db.pieces)
        print(f"Database loaded from {db_dir} ({len(db.pieces)} pieces)")
        return db

    def build_connectivity(self, output_path=None):
        """
        Build the connectivity graph from all pieces in the database.
        Updates self.connectivity and each piece's fits.
        """
        from common import connect

        temp_dir = output_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '_temp_conn'
        )
        os.makedirs(temp_dir, exist_ok=True)

        for pid, pd in self.pieces.items():
            for j, side in enumerate(pd.sides):
                side_file = os.path.join(temp_dir, f'side_{pid}_{j}.json')
                with open(side_file, 'w') as f:
                    json.dump({
                        'vertices': side.vertices.tolist(),
                        'piece_center': side.piece_center,
                        'is_edge': side.is_edge,
                        'original_photo_name': pd.photo_source,
                        'photo_space_origin': (0, 0),
                        'photo_space_centroid': [0, 0],
                        'photo_width': 500,
                        'photo_height': 500,
                    }, f)

        self.connectivity = connect.build(temp_dir, temp_dir)

        for pid_str, fits in self.connectivity.items():
            pid = int(pid_str)
            if pid in self.pieces:
                for i in range(4):
                    self.pieces[pid].fits[i] = [
                        (f[0], f[1], f[2]) for f in fits[i]
                    ]

        return self.connectivity

    def solve(self, piece_edge_info=None):
        """
        Attempt to solve the puzzle using the connectivity graph.
        Sets self.solution if successful.
        """
        from common.board import build as board_build

        if self.connectivity is None:
            self.build_connectivity()

        try:
            board = board_build(
                connectivity=self.connectivity,
                puzzle_width=self.width,
                puzzle_height=self.height,
                piece_edge_info=piece_edge_info,
            )
            self.solution = board
            self.load_from_solution(board)
            return board
        except Exception as e:
            print(f"Full solve failed: {e}")
            return None

    def incremental_connectivity_update(self, new_piece_id):
        """
        Update connectivity graph incrementally for a newly added piece.
        Only computes matches between the new piece and existing pieces.
        """
        new_pd = self.pieces.get(new_piece_id)
        if new_pd is None:
            return

        for si, side in enumerate(new_pd.sides):
            if side.is_edge:
                continue
            for other_pid, other_pd in self.pieces.items():
                if other_pid == new_piece_id:
                    continue
                for sj, other_side in enumerate(other_pd.sides):
                    if other_side.is_edge:
                        continue
                    error = side.error_when_fit_with(other_side)
                    if error <= sides.SIDE_MAX_ERROR_TO_MATCH:
                        new_pd.fits[si].append((other_pid, sj, error))
                        other_pd.fits[sj].append((new_piece_id, si, error))

        for si in range(4):
            new_pd.fits[si] = sorted(new_pd.fits[si], key=lambda x: x[2])

    def identify_piece(self, piece_image, top_n=3):
        """
        Identify a piece from an image against the database.
        Returns list of (piece_id, confidence) sorted by confidence.
        """
        from common.image_match import ORBMatcher, compute_color_histogram, histogram_similarity

        if not self.pieces:
            return []

        orb = ORBMatcher(max_features=300)
        scores = []

        _, query_des = orb.detect_and_compute(piece_image)
        query_hist = compute_color_histogram(piece_image)

        for pid, pd in self.pieces.items():
            score = 0.0

            if query_des is not None and pd.descriptors is not None:
                matches = orb.match(query_des, pd.descriptors)
                if matches:
                    avg_dist = np.mean([m.distance for m in matches])
                    orb_score = max(0, 1.0 - avg_dist / 256)
                    n_features = max(len(query_des), len(pd.descriptors), 1)
                    coverage = min(len(matches) / n_features, 1.0)
                    score += orb_score * 0.4 + coverage * 0.2
                else:
                    score += 0.1
            else:
                score += 0.1

            if query_hist is not None and pd.histogram is not None:
                hist_sim = histogram_similarity(query_hist, pd.histogram)
                score += max(0, hist_sim) * 0.4

            scores.append((pid, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_n]

    def check_completeness(self):
        """
        Check if the database has all expected pieces.
        Returns (missing_count, is_complete).
        """
        expected = self.metadata['expected_count']
        actual = len(self.pieces)
        missing = max(0, expected - actual)
        is_complete = actual >= expected
        self.metadata['missing_count'] = missing
        self.metadata['is_complete'] = is_complete
        return missing, is_complete
