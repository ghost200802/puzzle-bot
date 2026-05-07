import math
from typing import List, Tuple
import numpy as np

from common import util

0
SIDE_MAX_ERROR_TO_MATCH = 1.5

SIDE_MAX_LENGTH_DISCREPANCY = 0.05

SIDE_RESAMPLE_VERTEX_COUNT = 26

EDGE_PARALLEL_THRESHOLD_RAD = math.radians(10)


class Side(object):
    def __init__(self, piece_id, side_id, vertices, piece_center, is_edge, resample=False, rotate=True, photo_filename=None) -> None:
        self.piece_id = piece_id
        self.side_id = side_id
        self.piece_center = piece_center
        self.is_edge = is_edge
        self.vertices = vertices
        self.p1 = vertices[0]
        self.p2 = vertices[-1]
        self.original_p1 = vertices[0]
        self.original_p2 = vertices[-1]
        self.original_angle = util.angle_between(self.original_p1, self.original_p2)
        self.original_length = util.distance(self.original_p1, self.original_p2)
        self.is_convex = self._compute_convexity(vertices)
        self.photo_filename = photo_filename

        if resample:
            vertices, self.v_length = util.resample_polyline(vertices, n=SIDE_RESAMPLE_VERTEX_COUNT)
            if rotate:
                angle = self.angle
                self.vertices = Side.rotated(vertices=vertices, from_angle=angle, desired_angle=0)
                self.vertices_flipped = Side.rotated(vertices=vertices, from_angle=angle, desired_angle=math.pi)[::-1]
            else:
                self.vertices = np.array(vertices)
            self.p1 = self.vertices[0]
            self.p2 = self.vertices[-1]
        else:
            self.v_length = util.polyline_length(vertices)

    def __repr__(self) -> str:
        return f"Side({self.p1}->{self.p2} @ {int(self.angle * 180/math.pi)} deg, len={self.length}, n_vertices={len((self.vertices))}, is_edge={self.is_edge})"

    @property
    def angle(self) -> float:
        angle = util.angle_between(self.p1, self.p2)
        return angle

    @property
    def segment(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (self.p1, self.p2)

    @property
    def length(self) -> float:
        return util.distance(self.p1, self.p2)

    @staticmethod
    def _signed_distance_to_line(point, line_p1, line_p2):
        dx = line_p2[0] - line_p1[0]
        dy = line_p2[1] - line_p1[1]
        cross = dx * (point[1] - line_p1[1]) - dy * (point[0] - line_p1[0])
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return 0
        return cross / length

    def _compute_convexity(self, vertices):
        if self.is_edge:
            return None
        if self.piece_center is None:
            return None

        center_sd = Side._signed_distance_to_line(
            self.piece_center, self.original_p1, self.original_p2
        )
        if abs(center_sd) < 0.001:
            return None

        p1 = tuple(self.original_p1)
        p2 = tuple(self.original_p2)
        total_sd = 0
        for v in vertices:
            total_sd += Side._signed_distance_to_line(tuple(v), p1, p2)
        avg_sd = total_sd / len(vertices)

        return (avg_sd * center_sd) > 0

    def center_side_sign(self):
        if self.piece_center is None:
            return 0
        return Side._signed_distance_to_line(
            self.piece_center, self.original_p1, self.original_p2
        )

    def error_when_fit_with(self, side, flip=True, render=False, skip_edges=True, debug_str=None):
        if skip_edges and (self.is_edge or side.is_edge):
            return 1000

        d_scale = 1.0 - (self.length / side.length)
        if abs(d_scale) > SIDE_MAX_LENGTH_DISCREPANCY:
            return 1000

        polyline1 = self.vertices
        if flip:
            polyline2 = side.vertices_flipped
        else:
            polyline2 = side.vertices

        error, shift = util.error_between_polylines(polyline1, polyline2, p1_len=side.v_length)

        if render and debug_str and error <= SIDE_MAX_ERROR_TO_MATCH:
            print(debug_str)
            shifted0 = [(x - shift[0], y - shift[1]) for x, y in polyline1]
            print(f"\t ==> Error = {error}, shift: {shift}")
            util.render_polylines([shifted0, polyline2])

        return error

    @staticmethod
    def rotated(vertices, from_angle, desired_angle) -> List[Tuple[int, int]]:
        o = vertices[0]

        translated = []
        for i, (x, y) in enumerate(vertices):
            translated.append((x - o[0], y - o[1]))

        angle_diff = desired_angle - from_angle
        rotated = []

        for v in translated:
            rotated.append(util.rotate(v, around=translated[0], angle=angle_diff))

        if desired_angle != 0:
            min_x = min([v[0] for v in rotated])
            rotated = [(v[0] - min_x, v[1]) for v in rotated]

        return np.array(rotated)
