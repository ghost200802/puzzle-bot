import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.texture_verify import load_side_data, _resample_polyline, N_SAMPLES
from show_connectivity import _load_piece_data, _get_outline


def load_fonts(title_size=22, idx_size=11, tiny_size=9):
    try:
        return {
            'title': ImageFont.truetype("arialbd.ttf", title_size),
            'idx': ImageFont.truetype("arialbd.ttf", idx_size),
            'tiny': ImageFont.truetype("arial.ttf", tiny_size),
        }
    except Exception:
        default = ImageFont.load_default()
        return {k: default for k in ['title', 'idx', 'tiny']}


class CanvasViewport:
    def __init__(self, all_points, canvas_w=1600, target_h=900, margin=80, top_offset=50):
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        self._min_x = min(xs)
        self._max_x = max(xs)
        self._min_y = min(ys)
        self._max_y = max(ys)
        data_w = self._max_x - self._min_x
        data_h = self._max_y - self._min_y

        self._margin = margin
        self._top_offset = top_offset
        self._scale = min(
            (canvas_w - 2 * margin) / data_w if data_w > 0 else 1,
            (target_h - 2 * margin) / data_h if data_h > 0 else 1,
        )
        canvas_h = int(data_h * self._scale) + 2 * margin + top_offset + 30

        self.canvas = Image.new('RGBA', (canvas_w, canvas_h), (30, 30, 30, 255))
        self.draw = ImageDraw.Draw(self.canvas)

    def tc(self, x, y):
        return (
            (x - self._min_x) * self._scale + self._margin,
            (y - self._min_y) * self._scale + self._margin + self._top_offset,
        )

    def __call__(self, x, y):
        return self.tc(x, y)


def transform_point(v, src_mid, tgt_mid, rot):
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    dx = v[0] - src_mid[0]
    dy = v[1] - src_mid[1]
    return np.array([
        dx * cos_r - dy * sin_r + tgt_mid[0],
        dx * sin_r + dy * cos_r + tgt_mid[1],
    ])


def rotate_vector(v, rot):
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    return np.array([
        v[0] * cos_r - v[1] * sin_r,
        v[0] * sin_r + v[1] * cos_r,
    ])


def compute_tangent_normal(points, piece_center):
    results = []
    n = len(points)
    for i in range(n):
        pos = points[i]
        if i == 0:
            tangent = points[1] - points[0]
        elif i == n - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[i + 1] - points[i - 1]
        tlen = np.linalg.norm(tangent)
        if tlen < 1e-6:
            results.append({
                'pos': pos, 'tangent': np.array([0, 0]),
                'normal': np.array([0, 0]), 'band_pos': pos,
            })
            continue
        tangent = tangent / tlen
        normal = np.array([-tangent[1], tangent[0]])
        to_c = piece_center - pos
        if np.dot(normal, to_c) < 0:
            normal = -normal
        from common.texture_verify import INNER_OFFSET, BAND_WIDTH
        band_pos = pos + normal * (INNER_OFFSET + BAND_WIDTH // 2)
        results.append({
            'pos': pos, 'tangent': tangent,
            'normal': normal, 'band_pos': band_pos,
        })
    return results


def load_debug_pair(deduped_path, pid_a, si_a, pid_b, si_b):
    side_a = load_side_data(deduped_path, pid_a, si_a)
    side_b = load_side_data(deduped_path, pid_b, si_b)
    verts_a = side_a['vertices']
    verts_bf = side_b['vertices'][::-1].copy()
    sample_a = _resample_polyline(verts_a, N_SAMPLES)
    return side_a, side_b, verts_a, verts_bf, sample_a


def draw_piece_outlines(draw, piece_data, pid_a, pid_b, tc, transform_b=None):
    outline_a = _get_outline(piece_data[pid_a])
    pts_a = [tc(x, y) for x, y in outline_a]
    if len(pts_a) >= 3:
        draw.polygon(pts_a, fill=None, outline=(80, 120, 200, 160), width=2)

    outline_b_raw = _get_outline(piece_data[pid_b])
    if transform_b is not None:
        outline_b = [transform_b(np.array(p)) for p in outline_b_raw]
    else:
        outline_b = outline_b_raw
    pts_b = [tc(x, y) for x, y in outline_b]
    if len(pts_b) >= 3:
        draw.polygon(pts_b, fill=None, outline=(200, 80, 80, 160), width=2)


def draw_sample_points(draw, sample_a, sample_b, tc, fonts, radius=5):
    for i in range(len(sample_a)):
        ax, ay = tc(sample_a[i][0], sample_a[i][1])
        draw.ellipse(
            [ax - radius, ay - radius, ax + radius, ay + radius],
            fill=(50, 140, 255, 255), outline=(255, 255, 255, 200),
        )
        draw.text((ax + radius + 1, ay - 9), str(i),
                  fill=(100, 200, 255, 255), font=fonts['idx'])

    for i in range(len(sample_b)):
        bx, by = tc(sample_b[i][0], sample_b[i][1])
        draw.ellipse(
            [bx - radius, by - radius, bx + radius, by + radius],
            fill=(255, 80, 80, 255), outline=(255, 255, 255, 200),
        )
        draw.text((bx - 18, by + 4), str(i),
                  fill=(255, 150, 150, 255), font=fonts['idx'])
