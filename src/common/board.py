import os
import time
import heapq

from common.config import *

"""
(0, 0) ... (w, 0)
  .          .
  .          .
(0, h) ... (w, h)

Piece orientation:

 ___0___
|       |
3       1
|       |
 ---2---

"""

TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3

OPPOSITE = {
    TOP: BOTTOM, RIGHT: LEFT, BOTTOM: TOP, LEFT: RIGHT,
}

MAX_ITERATIONS_TO_FIND_BORDER = 1000
MAX_ITERATIONS = 150000000
MAX_PARTIAL_SOLVE_ITERATIONS = 5000000

class Orientation(object):
    ZERO_POINTS_UP = 0
    ZERO_POINTS_RIGHT = 1
    ZERO_POINTS_DOWN = 2
    ZERO_POINTS_LEFT = 3


class Board(object):
    @staticmethod
    def copy(board):
        # make a deep copy of each element in the board
        _board = [list(e) for e in board._board]
        return Board(board.width, board.height, _board=_board, _placed_piece_ids=set(board._placed_piece_ids))

    def __init__(self, width, height, _board=None, _placed_piece_ids=None) -> None:
        self.width = width
        self.height = height

        if _board is not None:
            self._board = _board
        else:
            self._board = []
            for y in range(self.height):
                self._board.append([])
                for x in range(self.width):
                    self._board[y].append(None)

        self._placed_piece_ids = _placed_piece_ids or set()

    def __repr__(self) -> str:
        num_digits = math.floor(math.log(self.width * self.height, 10)) + 3
        s = '\n  ' + '-' * num_digits * self.width + '\n'
        for y in range(self.height):
            for x in range(self.width):
                cell = self._board[y][x]
                if cell is None:
                    spaces = ' ' * (num_digits) + '-'
                    s += spaces
                else:
                    piece_id, _, orientation = cell
                    ori_str = '^>v<'[orientation]
                    s += '{:>{}}{}'.format(piece_id, num_digits, ori_str)
            s += '\n\n'
        return s

    def is_available(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self._board[y][x] is None

    def can_place(self, piece_id, fits, x, y, orientation, relax_edge_constraints=False):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False, f"Cannot place {piece_id} at ({x}, {y}) because it is outside the board"

        if piece_id in self._placed_piece_ids:
            return False, f"Cannot place {piece_id} at ({x}, {y}) because it has already been placed"

        if self._board[y][x] is not None:
            return False, f"Cannot place {piece_id} at ({x}, {y}) because it is already occupied by {self._board[y][x][0]}"

        sides_that_must_be_edges = self._sides_that_must_be_edges(x, y)
        for side_i in range(4):
            expect_edge = side_i in sides_that_must_be_edges
            rotated_i = (side_i - orientation) % 4
            fits_i = fits[rotated_i]
            is_edge = len(fits_i) == 0
            if not is_edge and expect_edge:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because side @ index {rotated_i} is not an edge piece"
            elif is_edge and not expect_edge and not relax_edge_constraints:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because side @ index {rotated_i} is an edge but it shouldn't be"

        # check connectivity of neighbors
        # if we have someone in a space next to us, let's make sure we connect properly to that space
        if x > 0 and self._board[y][x - 1] is not None:
            # check left neighbor has a fit
            neighbor_piece_id, _, _ = self._board[y][x - 1]
            rotated_i = (LEFT - orientation) % 4
            fits_i = [f[0] for f in fits[rotated_i]]
            if neighbor_piece_id not in fits_i:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because it does not connect to the right neighbor {neighbor_piece_id} (only connects to {fits_i})"
        if x < self.width - 1 and self._board[y][x + 1] is not None:
            # check right neighbor has a fit
            neighbor_piece_id, _, _ = self._board[y][x + 1]
            rotated_i = (RIGHT - orientation) % 4
            fits_i = [f[0] for f in fits[rotated_i]]
            if neighbor_piece_id not in fits_i:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because it does not connect to the right neighbor {neighbor_piece_id} (only connects to {fits_i})"
        if y > 0 and self._board[y - 1][x] is not None:
            # check top neighbor has a fit
            neighbor_piece_id, _, _ = self._board[y - 1][x]
            rotated_i = (TOP - orientation) % 4
            fits_i = [f[0] for f in fits[rotated_i]]
            if neighbor_piece_id not in fits_i:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because it does not connect to the right neighbor {neighbor_piece_id} (only connects to {fits_i})"
        if y < self.height - 1 and self._board[y + 1][x] is not None:
            # check bottom neighbor has a fit
            neighbor_piece_id, _, _ = self._board[y + 1][x]
            rotated_i = (BOTTOM - orientation) % 4
            fits_i = [f[0] for f in fits[rotated_i]]
            if neighbor_piece_id not in fits_i:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because it does not connect to the right neighbor {neighbor_piece_id} (only connects to {fits_i})"
        return True, None

    def place(self, piece_id, fits, x, y, orientation):
        self._board[y][x] = (piece_id, fits, orientation)
        self._placed_piece_ids.add(piece_id)

    @property
    def placed_count(self):
        return len(self._placed_piece_ids)

    def _sides_that_must_be_edges(self, x, y):
        sides = []
        if y == 0:
            sides.append(TOP)
        if y == self.height - 1:
            sides.append(BOTTOM)
        if x == 0:
            sides.append(LEFT)
        if x == self.width - 1:
            sides.append(RIGHT)
        return sides

    def __lt__(self, other):
        return self.placed_count < other.placed_count

    def get(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None
        return self._board[y][x]

    @property
    def missing_positions(self):
        """Return all unoccupied positions as list of (x, y)."""
        positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self._board[y][x] is None:
                    positions.append((x, y))
        return positions

    @property
    def total_positions(self):
        return self.width * self.height


def build(connectivity=None, input_path=None, output_path=None,
          puzzle_width=None, puzzle_height=None, piece_edge_info=None):
    """
    Builds the puzzle
    Takes in either a path to a directory that contains the connectivity graph, or the connectivity graph itself
    TODO: somehow pass the output along
    """
    # Allow overriding puzzle dimensions (for phone mode)
    pw = puzzle_width or PUZZLE_WIDTH
    ph = puzzle_height or PUZZLE_HEIGHT

    if connectivity is None:
        print("> Loading connectivity graph...")
        with open(os.path.join(input_path, 'connectivity.json'), 'r') as f:
            ps_raw = json.load(f)
    else:
        print("> Using provided connectivity graph...")
        ps_raw = connectivity

    ps = {}
    for piece_id, fits in ps_raw.items():
        piece_id = int(piece_id)
        ps[piece_id] = [[], [], [], []]
        for i in range(4):
            for other_piece_id, other_side_id, error in fits[i]:
                ps[piece_id][i].append((other_piece_id, other_side_id, error))

    corners = []
    edges = []
    edge_length = 2 * (pw + ph) - 4
    for piece_id, neighbors in ps.items():
        if piece_edge_info and piece_id in piece_edge_info:
            edge_flags = piece_edge_info[piece_id]
            edge_count = sum(1 for f in edge_flags if f)
        else:
            edge_count = sum([1 for n in neighbors if len(n) == 0])
        if edge_count > 0:
            edges.append(piece_id)
            if edge_count > 1:
                corners.append(piece_id)

    print(f"Corners: {corners}, Edges: {len(edges)}")
    if len(corners) != 4:
        print(f"Warning: Expected 4 corners, got {len(corners)}. Attempting partial solve...")
    if len(edges) != edge_length:
        print(f"Warning: Expected {edge_length} pieces on the edge, got {len(edges)}")

    success = False
    solution = None

    total_expected = pw * ph
    is_incomplete = len(ps) < total_expected
    is_corner_incomplete = len(corners) != 4

    if is_incomplete:
        print(f"Warning: only {len(ps)}/{total_expected} pieces available, "
              f"attempting partial solve...")

    # For a given puzzle, put the corners in a predictable order using the
    # arbitrary but repeatible heuristic of how many adjacent pieces they would fit.
    # This ensures the solution will appear in the same orientation, for any given puzzle.
    corners = sorted(
        corners,
        key=lambda c: sum([len(fits) for fits in ps[c]]),
        reverse=True
    )

    if not is_corner_incomplete and not is_incomplete:
        for i in range(0, 4):
            try:
                solution = build_from_corner(ps, start_piece_id=corners[i], edge_length=edge_length,
                                             puzzle_width=pw, puzzle_height=ph)
            except Exception as e:
                print(f"Failed to build from corner {i}: {e}")
                continue
            success = True
            break

    if not success:
        print("Attempting partial solve...")
        try:
            solution = build_partial(ps, puzzle_width=pw, puzzle_height=ph)
            if solution is not None and solution.placed_count > 0:
                success = True
                print(f"Partial solution found: {solution.placed_count}/{total_expected} pieces")
        except Exception as e:
            print(f"Partial solve also failed: {e}")

    if not success:
        raise Exception("Failed to solve")
    return solution

def _get_combined_cost(board, candidate_pid, ps, x, y):
    costs = []
    for dx, dy, facing_us in [(-1, 0, RIGHT), (1, 0, LEFT), (0, -1, BOTTOM), (0, 1, TOP)]:
        nx, ny = x + dx, y + dy
        neighbor = board.get(nx, ny)
        if neighbor is None:
            continue
        neighbor_pid, _, neighbor_ori = neighbor
        neighbor_side_facing_us = (facing_us - neighbor_ori) % 4
        for n_pid, n_side, n_error in ps[neighbor_pid][neighbor_side_facing_us]:
            if n_pid == candidate_pid:
                costs.append(n_error)
                break
    if not costs:
        return 0
    return max(costs)


def build_from_corner(ps, start_piece_id, edge_length,
                      puzzle_width=None, puzzle_height=None,
                      on_milestone=None, ps_fallback=None,
                      stop_after_border=False):
    pw = puzzle_width or PUZZLE_WIDTH
    ph = puzzle_height or PUZZLE_HEIGHT
    total = pw * ph

    print(f"\n===============================\nBuilding from corner {start_piece_id}...")
    start_piece_fits = ps[start_piece_id]
    start_orientation = _orient_start_corner_to_top_left(start_piece_fits)
    board = Board(width=pw, height=ph)

    x, y = (0, 0)
    board.place(start_piece_id, start_piece_fits, x, y, start_orientation)

    direction = RIGHT
    x += 1

    priority_q = []
    initial_push = (board, start_piece_id, start_orientation, x, y, direction)
    heapq.heappush(priority_q, (0, initial_push))

    iteration = 0
    longest = 0
    best_board = None
    t_start = time.time()
    last_report = time.time()
    border_done = False
    last_milestone_pct = 0

    while priority_q:
        priority, data = heapq.heappop(priority_q)
        board, start_piece_id, start_orientation, x, y, direction = data

        if iteration % 1000 == 0:
            now = time.time()
            elapsed = now - t_start
            print(f"  iter {iteration:>8d} | cur {board.placed_count:>3d} | best {longest:>3d}/{total} | cost {priority:.4f} | {elapsed:.1f}s")
            last_report = now

            if (iteration > MAX_ITERATIONS_TO_FIND_BORDER and longest < edge_length) or iteration > MAX_ITERATIONS:
                print(f"Gave up after {iteration} iterations, longest: {longest}")
                break

        if board.placed_count == total:
            print(f"Placed {total} pieces in {iteration} iterations ({time.time() - t_start:.1f}s)")
            break
        elif board.placed_count > longest:
            longest = board.placed_count
            best_board = board

            if not border_done and longest >= edge_length:
                border_done = True
                print(f"  *** Border complete: {longest} pieces at iter {iteration} ***")
                if on_milestone:
                    on_milestone(best_board, "border", iteration, time.time() - t_start)
                if stop_after_border:
                    print(f"  *** stop_after_border=True, returning border solution ***")
                    return best_board

            if border_done:
                pct = int(longest * 100 / total)
                threshold = (pct // 5) * 5
                if threshold > last_milestone_pct and threshold > 0:
                    last_milestone_pct = threshold
                    print(f"  *** Milestone {threshold}%: {longest}/{total} at iter {iteration} ***")
                    if on_milestone:
                        on_milestone(best_board, f"pct{threshold}", iteration, time.time() - t_start)

        iteration += 1

        placed_neighbor_info = []
        for dx, dy, facing_us in [(-1, 0, RIGHT), (1, 0, LEFT), (0, -1, BOTTOM), (0, 1, TOP)]:
            nx, ny = x + dx, y + dy
            nb = board.get(nx, ny)
            if nb is not None:
                placed_neighbor_info.append((nb[0], nb[2], facing_us))

        pool = []

        if len(placed_neighbor_info) >= 2:
            ncc_sets = []
            fb_used = False
            for nb_pid, nb_ori, facing_us in placed_neighbor_info:
                nb_side = (facing_us - nb_ori) % 4
                ncc_pids = {n_pid for n_pid, _, _ in ps[nb_pid][nb_side]}
                if ncc_pids:
                    ncc_sets.append(ncc_pids)
                elif ps_fallback is not None:
                    fb_pids = {n_pid for n_pid, _, _ in ps_fallback[nb_pid][nb_side]}
                    ncc_sets.append(fb_pids)
                    fb_used = True
                else:
                    ncc_sets.append(set())

            common_pids = set.intersection(*ncc_sets) - board._placed_piece_ids

            if not common_pids and ps_fallback is not None:
                fb_sets = []
                for nb_pid, nb_ori, facing_us in placed_neighbor_info:
                    nb_side = (facing_us - nb_ori) % 4
                    pids = {n_pid for n_pid, _, _ in ps_fallback[nb_pid][nb_side]}
                    fb_sets.append(pids)
                fb_common = set.intersection(*fb_sets) - board._placed_piece_ids
                for pid in fb_common:
                    nb_pid0, nb_ori0, facing_us0 = placed_neighbor_info[0]
                    nb_side0 = (facing_us0 - nb_ori0) % 4
                    orientation = None
                    for n_pid, n_side, _ in ps_fallback[nb_pid0][nb_side0]:
                        if n_pid == pid:
                            orientation = (OPPOSITE[facing_us0] - n_side) % 4
                            break
                    if orientation is None:
                        continue
                    ok, _ = board.can_place(pid, ps[pid], x, y, orientation)
                    if ok:
                        combined = _get_combined_cost(board, pid, ps_fallback, x, y)
                        pool.append((combined + 50.0, pid, orientation))

            for pid in common_pids:
                nb_pid0, nb_ori0, facing_us0 = placed_neighbor_info[0]
                nb_side0 = (facing_us0 - nb_ori0) % 4
                orientation = None
                for n_pid, n_side, _ in ps[nb_pid0][nb_side0]:
                    if n_pid == pid:
                        orientation = (OPPOSITE[facing_us0] - n_side) % 4
                        break
                if orientation is None:
                    continue
                ok, _ = board.can_place(pid, ps[pid], x, y, orientation)
                if ok:
                    cost_ps = ps_fallback if fb_used else ps
                    combined = _get_combined_cost(board, pid, cost_ps, x, y)
                    penalty = 50.0 if fb_used else 0.0
                    pool.append((combined + penalty, pid, orientation))
        else:
            index_of_neighbor_in_direction = (direction - start_orientation) % 4
            for neighbor_piece_id, neighbor_side_index, _ in ps[start_piece_id][index_of_neighbor_in_direction]:
                neighbor_orientation = (OPPOSITE[direction] - neighbor_side_index) % 4
                ok, _ = board.can_place(piece_id=neighbor_piece_id, fits=ps[neighbor_piece_id], x=x, y=y, orientation=neighbor_orientation)
                if ok:
                    combined = _get_combined_cost(board, neighbor_piece_id, ps, x, y)
                    pool.append((combined, neighbor_piece_id, neighbor_orientation))

        pool.sort()
        for combined, pid, ori in pool:
            next_board = Board.copy(board)
            next_board.place(pid, ps[pid], x, y, ori)
            next_direction = direction
            next_x = x + (1 if next_direction == RIGHT else -1 if next_direction == LEFT else 0)
            next_y = y + (1 if next_direction == BOTTOM else -1 if next_direction == TOP else 0)

            if not next_board.is_available(next_x, next_y):
                next_direction = (direction + 1) % 4
                next_x = x + (1 if next_direction == RIGHT else -1 if next_direction == LEFT else 0)
                next_y = y + (1 if next_direction == BOTTOM else -1 if next_direction == TOP else 0)

            data = [next_board, pid, ori, next_x, next_y, next_direction]
            heapq.heappush(priority_q, (combined, data))

    if board.placed_count == total:
        print(f"Found solution after {iteration} iterations!")
        return board
    else:
        result = best_board if best_board is not None else board
        print(f"Partial solution: {result.placed_count}/{total} after {iteration} iterations ({time.time() - t_start:.1f}s)")
        return result


def _orient_start_corner_to_top_left(p):
    if len(p[0]) == 0 and len(p[1]) == 0:
        # ''|   --> |''
        return Orientation.ZERO_POINTS_LEFT
    elif len(p[1]) == 0 and len(p[2]) == 0:
        #  __|  -->  |''
        return Orientation.ZERO_POINTS_DOWN
    elif len(p[2]) == 0 and len(p[3]) == 0:
        #  |__  -->  |''
        return Orientation.ZERO_POINTS_RIGHT
    elif len(p[3]) == 0 and len(p[0]) == 0:
        # |''  -->  |''
        return Orientation.ZERO_POINTS_UP
    else:
        raise ValueError(f"Piece {p} is not a corner piece")

def _try_place_at(board, ps, x, y):
    if not board.is_available(x, y):
        return False, board

    neighbor_constraints = []
    for dx, dy, neighbor_facing_us, our_facing_neighbor in [
        (-1, 0, RIGHT, LEFT), (1, 0, LEFT, RIGHT),
        (0, -1, BOTTOM, TOP), (0, 1, TOP, BOTTOM),
    ]:
        nx, ny = x + dx, y + dy
        neighbor = board.get(nx, ny)
        if neighbor is not None:
            neighbor_pid, _, neighbor_ori = neighbor
            neighbor_fits = ps[neighbor_pid][(neighbor_facing_us - neighbor_ori) % 4]
            neighbor_constraints.append((neighbor_pid, neighbor_fits, our_facing_neighbor))

    if not neighbor_constraints:
        return False, board

    active_constraints = [(pid, fits, opp) for pid, fits, opp in neighbor_constraints
                          if len(fits) > 0]

    if not active_constraints:
        return False, board

    candidate_pieces = None
    for neighbor_pid, neighbor_fits, opp_side in active_constraints:
        matching_pids = set()
        for n_pid, n_side, n_error in neighbor_fits:
            if n_pid not in board._placed_piece_ids:
                matching_pids.add((n_pid, n_side, n_error))
        if candidate_pieces is None:
            candidate_pieces = matching_pids
        else:
            candidate_pieces = candidate_pieces.intersection(
                {(p[0], p[1], p[2]) for p in matching_pids}
            )

    if not candidate_pieces:
        return False, board

    candidates = []
    for n_pid, n_side, n_error in candidate_pieces:
        best_opp = None
        conflict = False
        for _, neighbor_fits, opp_side in active_constraints:
            found = False
            for f_pid, f_side, f_error in neighbor_fits:
                if f_pid == n_pid:
                    orientation = (opp_side - f_side) % 4
                    if best_opp is not None and orientation != best_opp:
                        conflict = True
                        break
                    best_opp = orientation
                    found = True
                    break
            if conflict or not found:
                conflict = True
                break
        if not conflict and best_opp is not None:
            ok, _ = board.can_place(n_pid, ps[n_pid], x, y, best_opp,
                                     relax_edge_constraints=True)
            if ok:
                candidates.append((n_error, n_pid, best_opp))

    if candidates:
        candidates.sort()
        _, best_pid, best_ori = candidates[0]
        board.place(best_pid, ps[best_pid], x, y, best_ori)
        return True, board

    return False, board


def _fill_interior(board, ps, puzzle_width, puzzle_height):
    pw, ph = puzzle_width, puzzle_height
    total_before = board.placed_count
    improved = True
    pass_num = 0

    while improved:
        improved = False
        pass_num += 1
        for y in range(1, ph - 1):
            for x in range(1, pw - 1):
                if board.get(x, y) is not None:
                    continue
                has_neighbor = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if board.get(x + dx, y + dy) is not None:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    continue
                placed, board = _try_place_at(board, ps, x, y)
                if placed:
                    improved = True

    total_after = board.placed_count
    if total_after > total_before:
        print(f"  Interior fill placed {total_after - total_before} more pieces "
              f"(total: {total_after})")
    else:
        print(f"  Interior fill: no improvement ({pass_num} passes, {total_before} placed)")

    return board


def build_partial(ps, puzzle_width=None, puzzle_height=None):
    """
    Attempt to solve a puzzle with missing pieces.

    Strategy:
      1. Find corner pieces and start building from the best corner
      2. Skip empty positions where no piece fits (missing pieces)
      3. Continue building around gaps
      4. Fill interior positions greedily
      5. Return the best partial solution found

    Args:
        ps: connectivity graph {piece_id: [[fits], ...]}
        puzzle_width: width of the puzzle
        puzzle_height: height of the puzzle

    Returns:
        Board with partial placement, or None if no placement found
    """
    pw = puzzle_width or PUZZLE_WIDTH
    ph = puzzle_height or PUZZLE_HEIGHT

    corners = []
    for piece_id, neighbors in ps.items():
        edge_count = sum(1 for n in neighbors if len(n) == 0)
        if edge_count >= 2:
            corners.append(piece_id)

    if not corners:
        print("No corner pieces found for partial solve")
        return None

    corners.sort(
        key=lambda c: sum(len(fits) for fits in ps[c]),
        reverse=True
    )

    best_solution = None
    best_count = 0

    for start_piece_id in corners[:2]:
        try:
            start_piece_fits = ps[start_piece_id]
            start_orientation = _orient_start_corner_to_top_left(
                start_piece_fits
            )
        except ValueError:
            continue

        board = Board(width=pw, height=ph)
        board.place(start_piece_id, start_piece_fits, 0, 0,
                    start_orientation)

        direction = RIGHT
        x, y = 1, 0

        iteration = 0
        while iteration < MAX_PARTIAL_SOLVE_ITERATIONS:
            iteration += 1

            if x < 0 or x >= pw or y < 0 or y >= ph:
                if direction == RIGHT:
                    direction = BOTTOM
                    x = 0
                    y = 1
                elif direction == BOTTOM:
                    direction = LEFT
                    x = pw - 2
                    y = ph - 1
                elif direction == LEFT:
                    direction = TOP
                    x = 1
                    y = ph - 2
                else:
                    break
                continue

            if board.get(x, y) is not None:
                next_x = x + (1 if direction == RIGHT
                              else -1 if direction == LEFT else 0)
                next_y = y + (1 if direction == BOTTOM
                              else -1 if direction == TOP else 0)

                if not board.is_available(next_x, next_y):
                    if direction == RIGHT:
                        direction = BOTTOM
                        x = 0
                        y = 1
                    elif direction == BOTTOM:
                        direction = LEFT
                        x = pw - 2
                        y = ph - 1
                    elif direction == LEFT:
                        direction = TOP
                        x = 1
                        y = ph - 2
                    else:
                        break
                else:
                    x, y = next_x, next_y
                continue

            index_of_neighbor = (direction - start_orientation) % 4
            current_piece = board.get(x - (1 if direction == RIGHT
                                           else -1 if direction == LEFT
                                           else 0),
                                      y - (1 if direction == BOTTOM
                                           else -1 if direction == TOP
                                           else 0))
            if current_piece is None:
                break

            current_pid = current_piece[0]
            current_ori = current_piece[2]

            placed = False
            candidates = []
            for neighbor_pid, neighbor_side, error in ps.get(current_pid, [[]])[(
                direction - current_ori) % 4]:
                if neighbor_pid in board._placed_piece_ids:
                    continue
                neighbor_orientation = (OPPOSITE[direction] - neighbor_side) % 4
                ok, _ = board.can_place(
                    neighbor_pid, ps[neighbor_pid],
                    x, y, neighbor_orientation
                )
                if ok:
                    candidates.append((error, neighbor_pid, neighbor_orientation))

            if candidates:
                candidates.sort()
                _, best_pid, best_ori = candidates[0]
                board.place(best_pid, ps[best_pid], x, y, best_ori)
            else:
                pass

            next_x = x + (1 if direction == RIGHT
                          else -1 if direction == LEFT else 0)
            next_y = y + (1 if direction == BOTTOM
                          else -1 if direction == TOP else 0)

            if not board.is_available(next_x, next_y):
                if direction == RIGHT:
                    direction = BOTTOM
                    x = 0
                    y = 1
                elif direction == BOTTOM:
                    direction = LEFT
                    x = pw - 2
                    y = ph - 1
                elif direction == LEFT:
                    direction = TOP
                    x = 1
                    y = ph - 2
                else:
                    break
            else:
                x, y = next_x, next_y

        print(f"  Border phase placed {board.placed_count} pieces, filling interior...")
        board = _fill_interior(board, ps, pw, ph)

        if board.placed_count > best_count:
            best_count = board.placed_count
            best_solution = board

    return best_solution


def evaluate_solution(board, connectivity=None):
    """
    Evaluate the quality of a (partial) solution.

    Returns dict with:
      - placed_count: number of placed pieces
      - total_positions: total grid positions
      - coverage: percentage of positions filled
      - matched_edges: number of matched neighbor pairs
      - total_possible_edges: maximum possible matched edges
      - match_quality: percentage of possible edges matched
    """
    placed = board.placed_count
    total = board.total_positions
    coverage = placed / total if total > 0 else 0

    matched = 0
    possible = 0
    for y in range(board.height):
        for x in range(board.width):
            cell = board.get(x, y)
            if cell is None:
                continue
            if x < board.width - 1:
                right = board.get(x + 1, y)
                if right is not None:
                    matched += 1
                possible += 1
            if y < board.height - 1:
                below = board.get(x, y + 1)
                if below is not None:
                    matched += 1
                possible += 1

    quality = matched / possible if possible > 0 else 0

    return {
        'placed_count': placed,
        'total_positions': total,
        'coverage': coverage,
        'matched_edges': matched,
        'total_possible_edges': possible,
        'match_quality': quality,
    }
