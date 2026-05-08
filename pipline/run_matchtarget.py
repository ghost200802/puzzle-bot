import os
import sys
import json
import math
import argparse
import re
import time

import cv2
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, SOLUTION_DIR
from solve_display import generate_assembly_png, compute_piece_transforms
from common import board as board_mod

MAX_ASSEMBLY_LONG_SIDE = 2000


def load_solution(solution_dir):
    meta_path = os.path.join(solution_dir, 'solution_meta.json')
    pw, ph = 0, 0

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        pw = meta.get('width', 0)
        ph = meta.get('height', 0)
    else:
        parent = os.path.dirname(solution_dir)
        alt = os.path.join(parent, 'solution_meta.json')
        if os.path.exists(alt):
            with open(alt, 'r') as f:
                meta = json.load(alt)
            pw = meta.get('width', 0)
            ph = meta.get('height', 0)

    grid_path = os.path.join(solution_dir, 'solution_grid.txt')
    with open(grid_path, 'r') as f:
        grid_text = f.read()

    placed = {}
    arrow_map = {'^': 0, '>': 1, 'v': 2, '<': 3}
    lines = [l.strip() for l in grid_text.split('\n')
             if l.strip() and not l.strip().startswith('--')]
    row = 0
    for line in lines:
        tokens = line.split()
        col = 0
        for tok in tokens:
            m = re.match(r'^(\d+)([\^v<>])$', tok)
            if m:
                pid = int(m.group(1))
                ori = arrow_map[m.group(2)]
                placed[pid] = {'gx': col, 'gy': row, 'orientation': ori}
            col += 1
        row += 1

    if pw == 0:
        pw = max(p['gx'] for p in placed.values()) + 1 if placed else 0
    if ph == 0:
        ph = max(p['gy'] for p in placed.values()) + 1 if placed else 0

    return pw, ph, placed


def _build_board_from_placed(pw, ph, placed, ps_raw):
    b = board_mod.Board(pw, ph)
    for pid, info in placed.items():
        fits = ps_raw.get(pid, [[], [], [], []])
        b.place(pid, fits, info['gx'], info['gy'], info['orientation'])
    return b


def _resize_to_max(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _detect_puzzle_rect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    edges = cv2.dilate(edges, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    best = contours[0]
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        ordered = pts[np.argsort(angles)]
        top = ordered[ordered[:, 1] <= center[1]]
        bot = ordered[ordered[:, 1] > center[1]]
        if len(top) < 2 or len(bot) < 2:
            x, y, w, h = cv2.boundingRect(best)
            return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        top = top[np.argsort(top[:, 0])]
        bot = bot[np.argsort(bot[:, 0])]
        return np.array([top[0], top[-1], bot[-1], bot[0]], dtype=np.float32)

    x, y, w, h = cv2.boundingRect(best)
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def _ncc_score(img_a, img_b):
    min_h = min(img_a.shape[0], img_b.shape[0])
    min_w = min(img_a.shape[1], img_b.shape[1])
    a = img_a[:min_h, :min_w].astype(np.float32)
    b = img_b[:min_h, :min_w].astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    sa = a.std() + 1e-8
    sb = b.std() + 1e-8
    return float(np.mean(a * b) / (sa * sb))


def _histogram_match_cdf(src_vals, ref_vals):
    src_hist, _ = np.histogram(src_vals.astype(np.uint8), bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(ref_vals.astype(np.uint8), bins=256, range=(0, 256))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1
    ref_cdf /= ref_cdf[-1] if ref_cdf[-1] > 0 else 1
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapping[i] = int(np.argmin(np.abs(ref_cdf - src_cdf[i])))
    return mapping


class TargetMatcher:
    def __init__(self, target_image_path, pw, ph, placed, output_dir,
                 piece_color_dir=None, deduped_dir=None, output_root=None):
        self.target_path = target_image_path
        self.pw = pw
        self.ph = ph
        self.placed = placed
        self.output_dir = output_dir

        self.target_raw = cv2.imread(target_image_path)
        if self.target_raw is None:
            raise FileNotFoundError(f"Cannot load target image: {target_image_path}")

        self.output_root = output_root
        if output_root:
            self.color_dir = piece_color_dir or os.path.join(output_root, '2_piece_colors')
            self.deduped_dir = deduped_dir or os.path.join(output_root, DEDUPED_DIR)
        else:
            self.color_dir = piece_color_dir or self._find_dir('2_piece_colors')
            self.deduped_dir = deduped_dir or self._find_dir(DEDUPED_DIR)

        self.target_aligned = None
        self.best_rotation = 0
        self.match_results = {}
        self._assembly_img = None
        self._cell_w = 0
        self._cell_h = 0
        self._grid_offset_x = 0.0
        self._grid_offset_y = 0.0
        self._canvas_info = None
        self._resize_scale = 1.0
        self._gen_scale = 1.0
        self._margin = 0.0
        self._header_h = 60
        self._piece_transforms = None
        self._erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def _find_dir(self, dirname):
        d = self.output_dir
        for _ in range(5):
            candidate = os.path.join(d, dirname)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        return os.path.join(self.output_dir, dirname)

    def _generate_assembly(self):
        print("\n--- Phase 1: Generate Assembly Image ---")
        ps_raw = {}
        for pid in self.placed:
            sides = []
            for si in range(4):
                json_path = os.path.join(self.deduped_dir, f'side_{pid}_{si}.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        sides.append(json.load(f))
                else:
                    sides.append(None)
            ps_raw[pid] = sides

        sol_board = _build_board_from_placed(self.pw, self.ph, self.placed, ps_raw)

        assembly_path = os.path.join(self.output_dir, '_match_assembly.png')
        canvas_info = generate_assembly_png(sol_board, self.deduped_dir,
                                            os.path.dirname(self.color_dir), assembly_path)

        raw_assembly = cv2.imread(assembly_path)
        if raw_assembly is None:
            print("  Failed to generate assembly image")
            return False

        self._assembly_img, self._resize_scale = _resize_to_max(raw_assembly, MAX_ASSEMBLY_LONG_SIDE)
        ah, aw = self._assembly_img.shape[:2]
        print(f"  Assembly generated, resized to {aw}x{ah}")

        self._canvas_info = canvas_info
        self._piece_transforms, _, _ = compute_piece_transforms(sol_board, self.deduped_dir)
        return True

    def _compute_grid_layout(self):
        ci = self._canvas_info
        if not ci:
            print("  No canvas info, using fallback")
            ah, aw = self._assembly_img.shape[:2]
            m = aw * 0.03
            self._cell_w = (aw - 2 * m) / self.pw
            self._cell_h = (ah - 2 * m) / self.ph
            self._grid_offset_x = m
            self._grid_offset_y = m
            return

        min_x, min_y = ci['min_x'], ci['min_y']
        max_x, max_y = ci['max_x'], ci['max_y']
        data_w = max_x - min_x
        data_h = max_y - min_y
        if data_w <= 0 or data_h <= 0:
            ah, aw = self._assembly_img.shape[:2]
            m = aw * 0.03
            self._cell_w = (aw - 2 * m) / self.pw
            self._cell_h = (ah - 2 * m) / self.ph
            self._grid_offset_x = m
            self._grid_offset_y = m
            return

        self._margin = max(data_w, data_h) * 0.05
        self._header_h = 60

        canvas_w = data_w + 2 * self._margin
        canvas_h = data_h + 2 * self._margin + self._header_h
        max_size = 12000
        if max(canvas_w, canvas_h) > max_size:
            self._gen_scale = max_size / max(canvas_w, canvas_h)
        else:
            self._gen_scale = 1.0

        grid_x = self._margin * self._gen_scale
        grid_y = self._margin * self._gen_scale + self._header_h
        grid_w = data_w * self._gen_scale
        grid_h = data_h * self._gen_scale

        rs = self._resize_scale
        self._grid_offset_x = grid_x * rs
        self._grid_offset_y = grid_y * rs
        self._cell_w = grid_w * rs / self.pw
        self._cell_h = grid_h * rs / self.ph

        print(f"  Grid layout: offset=({self._grid_offset_x:.1f},{self._grid_offset_y:.1f}), "
              f"cell={self._cell_w:.1f}x{self._cell_h:.1f}, gen_scale={self._gen_scale:.4f}")

    def rectify_target(self):
        if not self._generate_assembly():
            self.target_aligned = self.target_raw.copy()
            return

        self._compute_grid_layout()

        print("\n--- Phase 2: Align Target to Assembly ---")
        ah, aw = self._assembly_img.shape[:2]
        content_x2 = int(self._grid_offset_x + self.pw * self._cell_w)
        content_y2 = int(self._grid_offset_y + self.ph * self._cell_h)
        content_w = content_x2 - int(self._grid_offset_x)
        content_h = content_y2 - int(self._grid_offset_y)

        corners = _detect_puzzle_rect(self.target_raw)

        if corners is not None:
            print(f"  Detected target corners: {corners.tolist()}")
            dst = np.array([
                [0, 0],
                [content_w - 1, 0],
                [content_w - 1, content_h - 1],
                [0, content_h - 1],
            ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(corners, dst)
            target_rect = cv2.warpPerspective(self.target_raw, M, (content_w, content_h))
        else:
            print("  No puzzle boundary detected, using simple resize")
            target_rect = cv2.resize(self.target_raw, (content_w, content_h))

        target_rect_rotated = cv2.rotate(target_rect, cv2.ROTATE_180)

        ox = int(self._grid_offset_x)
        oy = int(self._grid_offset_y)
        ch, cw = target_rect.shape[:2]
        region = self._assembly_img[oy:min(oy + ch, ah), ox:min(ox + cw, aw)]
        rh, rw = region.shape[:2]

        ncc_normal = _ncc_score(region, target_rect[:rh, :rw])
        ncc_rotated = _ncc_score(region, target_rect_rotated[:rh, :rw])

        print(f"  Normal NCC: {ncc_normal:.4f}")
        print(f"  Rotated180 NCC: {ncc_rotated:.4f}")

        if ncc_rotated > ncc_normal:
            chosen = target_rect_rotated
            self.best_rotation = 2
            print(f"  Using 180-degree rotation")
        else:
            chosen = target_rect
            self.best_rotation = 0
            print(f"  Using normal orientation")

        canvas = np.full((ah, aw, 3), 255, dtype=np.uint8)
        if oy + ch <= ah and ox + cw <= aw:
            canvas[oy:oy + ch, ox:ox + cw] = chosen
        else:
            canvas[oy:min(oy + ch, ah), ox:min(ox + cw, aw)] = \
                chosen[:min(ch, ah - oy), :min(cw, aw - ox)]

        self.target_aligned = canvas

        aligned_path = os.path.join(self.output_dir, 'target_aligned.png')
        cv2.imwrite(aligned_path, self.target_aligned)
        print(f"  Aligned target saved: {aligned_path}")

    def _prepare_piece(self, pid):
        if self._piece_transforms is None or pid not in self._piece_transforms:
            return None, None, None, None, None, None

        rotation, translation, ic = self._piece_transforms[pid]
        color_path = os.path.join(self.color_dir, f'piece_{pid}.png')
        if not os.path.exists(color_path):
            return None, None, None, None, None, None

        piece_img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        if piece_img is None:
            return None, None, None, None, None, None

        if piece_img.ndim == 2:
            piece_bgr = cv2.cvtColor(piece_img, cv2.COLOR_GRAY2BGR)
            alpha_raw = np.full_like(piece_img, 255, dtype=np.uint8)
        elif piece_img.shape[2] == 4:
            piece_bgr = piece_img[:, :, :3]
            alpha_raw = piece_img[:, :, 3]
        else:
            piece_bgr = piece_img
            alpha_raw = np.full(piece_img.shape[:2], 255, dtype=np.uint8)

        h_img, w_img = piece_bgr.shape[:2]
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
        img_pts = []
        for cx, cy in corners:
            dx = cx - ic[0]
            dy = cy - ic[1]
            ox = dx * cos_r - dy * sin_r + ic[0] + translation[0]
            oy = dx * sin_r + dy * cos_r + ic[1] + translation[1]
            img_pts.append((ox, oy))

        img_min_x = min(p[0] for p in img_pts)
        img_min_y = min(p[1] for p in img_pts)
        img_max_x = max(p[0] for p in img_pts)
        img_max_y = max(p[1] for p in img_pts)

        gs = self._gen_scale
        out_w_gen = int(math.ceil((img_max_x - img_min_x) * gs)) + 2
        out_h_gen = int(math.ceil((img_max_y - img_min_y) * gs)) + 2

        cos_neg = math.cos(-rotation)
        sin_neg = math.sin(-rotation)
        sm_x = (img_min_x - ic[0] - translation[0]) * gs
        sm_y = (img_min_y - ic[1] - translation[1]) * gs

        a_v = cos_neg * gs
        b_v = -sin_neg * gs
        c_v = cos_neg * sm_x - sin_neg * sm_y + ic[0]
        d_v = sin_neg * gs
        e_v = cos_neg * gs
        f_v = sin_neg * sm_x + cos_neg * sm_y + ic[1]

        M_pil = np.array([[a_v, b_v, c_v], [d_v, e_v, f_v], [0, 0, 1]], dtype=np.float64)
        M_aff = np.linalg.inv(M_pil)[:2, :]

        piece_gen = cv2.warpAffine(piece_bgr, M_aff, (out_w_gen, out_h_gen),
                                    flags=cv2.INTER_AREA,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        alpha_gen = cv2.warpAffine(alpha_raw, M_aff, (out_w_gen, out_h_gen),
                                    flags=cv2.INTER_AREA,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        rs = self._resize_scale
        out_w = max(1, int(out_w_gen * rs))
        out_h = max(1, int(out_h_gen * rs))

        piece_final = cv2.resize(piece_gen, (out_w, out_h), interpolation=cv2.INTER_AREA)
        alpha_final = cv2.resize(alpha_gen, (out_w, out_h), interpolation=cv2.INTER_AREA)

        mask_eroded = cv2.erode((alpha_final > 128).astype(np.uint8), self._erode_kernel) > 0
        mask_raw = alpha_final > 128
        piece_gray = cv2.cvtColor(piece_final, cv2.COLOR_BGR2GRAY)

        paste_x = (img_min_x - self._canvas_info['min_x'] + self._margin) * gs * rs
        paste_y = ((img_min_y - self._canvas_info['min_y'] + self._margin) * gs + self._header_h) * rs

        return piece_gray, mask_eroded, (paste_x, paste_y), (out_w, out_h), piece_final, mask_raw

    def _match_single_piece(self, pid, info):
        piece_gray, mask_eroded, paste_pos, size, _, _ = self._prepare_piece(pid)
        if piece_gray is None:
            return None

        paste_x, paste_y = paste_pos
        out_w, out_h = size
        th, tw = self.target_aligned.shape[:2]
        search_margin = 40

        tx1 = max(0, int(paste_x) - search_margin)
        ty1 = max(0, int(paste_y) - search_margin)
        tx2 = min(tw, int(paste_x + out_w) + search_margin)
        ty2 = min(th, int(paste_y + out_h) + search_margin)
        target_region_gray = cv2.cvtColor(
            self.target_aligned[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2GRAY)

        base_x = int(paste_x) - tx1
        base_y = int(paste_y) - ty1

        best_score = -999
        best_dx = 0
        best_dy = 0

        coarse_step = 4
        for dy in range(-search_margin, search_margin + 1, coarse_step):
            for dx in range(-search_margin, search_margin + 1, coarse_step):
                rx = base_x + dx
                ry = base_y + dy
                x1t = max(0, rx)
                y1t = max(0, ry)
                x2t = min(target_region_gray.shape[1], rx + out_w)
                y2t = min(target_region_gray.shape[0], ry + out_h)
                x1p = x1t - rx
                y1p = y1t - ry
                pw_ = x2t - x1t
                ph_ = y2t - y1t
                if pw_ <= 0 or ph_ <= 0:
                    continue
                m = mask_eroded[y1p:y1p + ph_, x1p:x1p + pw_]
                if m.sum() < 100:
                    continue
                p = piece_gray[y1p:y1p + ph_, x1p:x1p + pw_][m].astype(np.float64)
                t = target_region_gray[y1t:y1t + ph_, x1t:x1t + pw_][m].astype(np.float64)
                p_n = p - p.mean()
                t_n = t - t.mean()
                denom = np.sqrt(np.sum(p_n ** 2) * np.sum(t_n ** 2))
                score = float(np.sum(p_n * t_n) / denom) if denom > 1e-6 else 0.0
                if score > best_score:
                    best_score = score
                    best_dx = dx
                    best_dy = dy

        fine_range = coarse_step + 1
        for dy in range(best_dy - fine_range, best_dy + fine_range + 1):
            for dx in range(best_dx - fine_range, best_dx + fine_range + 1):
                rx = base_x + dx
                ry = base_y + dy
                x1t = max(0, rx)
                y1t = max(0, ry)
                x2t = min(target_region_gray.shape[1], rx + out_w)
                y2t = min(target_region_gray.shape[0], ry + out_h)
                x1p = x1t - rx
                y1p = y1t - ry
                pw_ = x2t - x1t
                ph_ = y2t - y1t
                if pw_ <= 0 or ph_ <= 0:
                    continue
                m = mask_eroded[y1p:y1p + ph_, x1p:x1p + pw_]
                if m.sum() < 100:
                    continue
                p = piece_gray[y1p:y1p + ph_, x1p:x1p + pw_][m].astype(np.float64)
                t = target_region_gray[y1t:y1t + ph_, x1t:x1t + pw_][m].astype(np.float64)
                p_n = p - p.mean()
                t_n = t - t.mean()
                denom = np.sqrt(np.sum(p_n ** 2) * np.sum(t_n ** 2))
                score = float(np.sum(p_n * t_n) / denom) if denom > 1e-6 else 0.0
                if score > best_score:
                    best_score = score
                    best_dx = dx
                    best_dy = dy

        # Phase 2: rotation search at best translation
        fx = int(paste_x) + best_dx
        fy = int(paste_y) + best_dy
        fx1 = max(0, fx)
        fy1 = max(0, fy)
        fx2 = min(tw, fx + out_w)
        fy2 = min(th, fy + out_h)
        px1 = fx1 - fx
        py1 = fy1 - fy
        pw_ = fx2 - fx1
        ph_ = fy2 - fy1

        target_at_best_gray = cv2.cvtColor(
            self.target_aligned[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2GRAY)

        angles = np.arange(-5, 5.5, 0.5)
        best_angle = 0.0

        if best_score < 0.2:
            pass
        else:
            for angle in angles:
                if abs(angle) < 0.01:
                    rot_gray = piece_gray[py1:py1 + ph_, px1:px1 + pw_]
                    rot_mask = mask_eroded[py1:py1 + ph_, px1:px1 + pw_]
                else:
                    M_rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), angle, 1.0)
                    rot_bgr = cv2.warpAffine(
                        cv2.cvtColor(piece_gray, cv2.COLOR_GRAY2BGR), M_rot, (out_w, out_h),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                    rot_gray = cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2GRAY)
                    rot_alpha = cv2.warpAffine(
                        (mask_eroded.astype(np.uint8) * 255), M_rot, (out_w, out_h),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    rot_mask = cv2.erode(rot_alpha, self._erode_kernel) > 0
                    rot_gray = rot_gray[py1:py1 + ph_, px1:px1 + pw_]
                    rot_mask = rot_mask[py1:py1 + ph_, px1:px1 + pw_]

                m = rot_mask
                if m.sum() < 100:
                    continue
                p = rot_gray[m].astype(np.float64)
                t = target_at_best_gray[m].astype(np.float64)
                p_n = p - p.mean()
                t_n = t - t.mean()
                denom = np.sqrt(np.sum(p_n ** 2) * np.sum(t_n ** 2))
                score = float(np.sum(p_n * t_n) / denom) if denom > 1e-6 else 0.0
                if score > best_score:
                    best_score = score
                    best_angle = angle

        # Phase 3: histmatch NCC at final position
        if abs(best_angle) < 0.01:
            final_gray = piece_gray[py1:py1 + ph_, px1:px1 + pw_]
            final_mask = mask_eroded[py1:py1 + ph_, px1:px1 + pw_]
        else:
            M_rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), best_angle, 1.0)
            rot_bgr = cv2.warpAffine(
                cv2.cvtColor(piece_gray, cv2.COLOR_GRAY2BGR), M_rot, (out_w, out_h),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            final_gray = cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2GRAY)
            rot_alpha = cv2.warpAffine(
                (mask_eroded.astype(np.uint8) * 255), M_rot, (out_w, out_h),
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            final_mask = cv2.erode(rot_alpha, self._erode_kernel) > 0
            final_gray = final_gray[py1:py1 + ph_, px1:px1 + pw_]
            final_mask = final_mask[py1:py1 + ph_, px1:px1 + pw_]

        hm_score = best_score
        if final_mask.sum() >= 100:
            p = final_gray[final_mask].astype(np.float64)
            t = target_at_best_gray[final_mask].astype(np.float64)
            hm_map = _histogram_match_cdf(p, t)
            pm = hm_map[p.astype(np.uint8)].astype(np.float64)
            pm_n = pm - pm.mean()
            t_n = t - t.mean()
            denom = np.sqrt(np.sum(pm_n ** 2) * np.sum(t_n ** 2))
            hm_score = float(np.sum(pm_n * t_n) / denom) if denom > 1e-6 else 0.0

        ncc_score = max(0, best_score)
        combined = max(ncc_score, hm_score)

        return {
            'score': combined,
            'ncc_score': ncc_score,
            'hist_score': max(0, hm_score),
            'dx': float(best_dx),
            'dy': float(best_dy),
            'angle': float(best_angle),
        }

    def match_all_pieces(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print("\n--- Phase 3: Per-Piece Matching (color image + translation + rotation) ---")
        results = {}
        pids = sorted(self.placed.keys())
        total = len(pids)

        n_workers = min(8, os.cpu_count() or 4)
        print(f"  Using {n_workers} threads for {total} pieces")

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for pid in pids:
                info = self.placed[pid]
                futures[executor.submit(self._match_single_piece, pid, info)] = pid

            done_count = 0
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[pid] = result
                except Exception as e:
                    print(f"  Error matching piece {pid}: {e}")

                done_count += 1
                if done_count % 10 == 0 or done_count == total:
                    elapsed = time.time() - t0
                    print(f"  Matched {done_count}/{total} ({elapsed:.1f}s)")

        elapsed = time.time() - t0
        self.match_results = results

        scores = [r['score'] for r in results.values()]
        if scores:
            print(f"  Score stats: min={min(scores):.3f}, max={max(scores):.3f}, "
                  f"mean={np.mean(scores):.3f}, median={np.median(scores):.3f}")
        print(f"  Total matching time: {elapsed:.1f}s")
        return results

    def refine_positions(self, threshold=0.7):
        print(f"\n--- Phase 4: Position Refinement ---")
        refined_count = 0
        for pid, result in self.match_results.items():
            result['confidence'] = result['score']
            result['refined'] = result['score'] >= threshold
            if result['refined']:
                refined_count += 1
        print(f"  High confidence: {refined_count}/{len(self.match_results)}")
        return self.match_results

    def generate_report(self, output_dir=None):
        print("\n--- Phase 5: Generating Report ---")
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)

        report = {}
        for pid, result in self.match_results.items():
            info = self.placed[pid]
            report[str(pid)] = {
                'grid_pos': [info['gx'], info['gy']],
                'orientation': info['orientation'],
                'score': round(float(result.get('score', 0)), 4),
                'ncc_score': round(float(result.get('ncc_score', 0)), 4),
                'hist_score': round(float(result.get('hist_score', 0)), 4),
                'dx': round(float(result.get('dx', 0)), 2),
                'dy': round(float(result.get('dy', 0)), 2),
                'angle': round(float(result.get('angle', 0)), 2),
                'confidence': round(float(result.get('confidence', 0)), 4),
                'refined': bool(result.get('refined', False)),
            }

        report_path = os.path.join(out, 'target_match_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved: {report_path}")
        return report

    def generate_visual(self, output_dir=None):
        from PIL import Image, ImageDraw, ImageFont

        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)

        target_bgr = self.target_aligned.copy()
        th, tw = target_bgr.shape[:2]

        for pid, result in self.match_results.items():
            piece_gray, mask_eroded, paste_pos, size, piece_bgr, mask_raw = self._prepare_piece(pid)
            if piece_bgr is None:
                continue

            paste_x, paste_y = paste_pos
            out_w, out_h = size
            dx = result.get('dx', 0)
            dy = result.get('dy', 0)
            angle = result.get('angle', 0)

            if abs(angle) > 0.01:
                M_rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), angle, 1.0)
                piece_bgr = cv2.warpAffine(piece_bgr, M_rot, (out_w, out_h),
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                alpha_uint8 = (mask_raw.astype(np.uint8)) * 255
                alpha_rot = cv2.warpAffine(alpha_uint8, M_rot, (out_w, out_h),
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                mask_raw = alpha_rot > 128

            fx = int(paste_x) + int(dx)
            fy = int(paste_y) + int(dy)

            px1 = max(0, fx)
            py1 = max(0, fy)
            px2 = min(tw, fx + out_w)
            py2 = min(th, fy + out_h)

            sx1 = px1 - fx
            sy1 = py1 - fy
            sx2 = sx1 + (px2 - px1)
            sy2 = sy1 + (py2 - py1)

            if sx2 <= sx1 or sy2 <= sy1:
                continue

            m = mask_raw[sy1:sy2, sx1:sx2]
            if m.sum() < 50:
                continue

            blend = m.astype(np.float32) * 0.95
            blend3 = np.stack([blend] * 3, axis=2)

            region = target_bgr[py1:py2, px1:px2]
            piece_region = piece_bgr[sy1:sy2, sx1:sx2]
            target_bgr[py1:py2, px1:px2] = (
                region.astype(np.float32) * (1 - blend3) + piece_region.astype(np.float32) * blend3
            ).astype(np.uint8)

        overlay_pil = Image.fromarray(cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGBA))
        draw = ImageDraw.Draw(overlay_pil)

        try:
            font = ImageFont.truetype("arialbd.ttf", max(10, min(20, int(self._cell_w / 8))))
        except Exception:
            font = ImageFont.load_default()

        for pid, result in self.match_results.items():
            info = self.placed[pid]
            gx, gy = info['gx'], info['gy']
            score = result.get('score', 0)
            dx = result.get('dx', 0)
            dy = result.get('dy', 0)

            x1 = int(self._grid_offset_x + gx * self._cell_w + dx)
            y1 = int(self._grid_offset_y + gy * self._cell_h + dy)
            x2 = x1 + int(self._cell_w)
            y2 = y1 + int(self._cell_h)

            if score >= 0.7:
                color = (0, 200, 0, 200)
            elif score >= 0.4:
                color = (0, 200, 200, 200)
            else:
                color = (0, 0, 200, 200)

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"{pid}:{score:.2f}"
            draw.text((x1 + 2, y2 - 14), label, fill=(255, 255, 255, 255), font=font)

        vis_bgr = cv2.cvtColor(np.array(overlay_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        vis_path = os.path.join(out, 'target_match_visual.png')
        cv2.imwrite(vis_path, vis_bgr)
        print(f"  Visual saved: {vis_path}")
        return vis_path

    def run(self, refine_threshold=0.7):
        self.rectify_target()
        self.match_all_pieces()
        self.refine_positions(threshold=refine_threshold)
        self.generate_report()
        self.generate_visual()
        return self.match_results


def match_target(target_image_path, solution, deduped_dir, output_dir,
                 piece_color_dir=None, refine_threshold=0.7):
    pw = solution.width
    ph = solution.height

    placed = {}
    for gy in range(ph):
        for gx in range(pw):
            cell = solution.get(gx, gy)
            if cell is not None:
                pid, fits, orientation = cell
                placed[pid] = {'gx': gx, 'gy': gy, 'orientation': orientation}

    matcher = TargetMatcher(
        target_image_path=target_image_path,
        pw=pw, ph=ph,
        placed=placed,
        output_dir=os.path.join(output_dir, SOLUTION_DIR),
        piece_color_dir=piece_color_dir,
        deduped_dir=deduped_dir,
    )
    matcher.run(refine_threshold=refine_threshold)

    result = {}
    for pid, r in matcher.match_results.items():
        result[pid] = {
            'score': r.get('score', 0),
            'confidence': r.get('confidence', 0),
            'refined': r.get('refined', False),
        }
    result['orientation'] = matcher.best_rotation
    return result


def main():
    parser = argparse.ArgumentParser(description='Match puzzle solution against target image')
    parser.add_argument('--target', required=True, help='Path to target image')
    parser.add_argument('--solution', required=True, help='Path to solution directory')
    parser.add_argument('--output', default=None, help='Output directory for results')
    parser.add_argument('--output-root', default=None, help='Root output directory')
    parser.add_argument('--refine-threshold', type=float, default=0.7)
    args = parser.parse_args()

    print("=" * 60)
    print("Target Image Matching")
    print("=" * 60)

    pw, ph, placed = load_solution(args.solution)
    print(f"Solution: {pw}x{ph}, {len(placed)} placed pieces")

    output_dir = args.output or os.path.dirname(args.solution)
    matcher = TargetMatcher(
        target_image_path=args.target,
        pw=pw, ph=ph,
        placed=placed,
        output_dir=output_dir,
        output_root=args.output_root,
    )
    matcher.run(refine_threshold=args.refine_threshold)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
