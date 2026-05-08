import os
import sys
import json
import math
import time
import argparse
import shutil

import cv2
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))
sys.path.insert(0, _here)

from common.config import DEDUPED_DIR
from common.board import Board
from common import output as board_output
from solve_display import generate_assembly_png, compute_piece_transforms
from run_matchtarget import (
    load_solution, _build_board_from_placed, _resize_to_max,
    _histogram_match_cdf, TargetMatcher,
)

OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
DEDUPED_PATH = os.path.join(OUTPUT_DIR, DEDUPED_DIR)

ERODE_PX = 5
NCC_ACCEPT = 0.8
SEARCH_MARGIN = 40
COARSE_STEP = 4
ROTATION_RANGE = 5.0
ROTATION_STEP = 0.5

ORI_TO_ANGLE = {0: 0, 1: 90, 2: 180, 3: 270}


def _load_connectivity(connectivity_file):
    with open(connectivity_file, 'r') as f:
        raw = json.load(f)
    ps = {}
    for pid_str, sides in raw.items():
        pid = int(pid_str)
        ps[pid] = [[], [], [], []]
        for si in range(4):
            for m in sides[si]:
                ps[pid][si].append((m['pid'], m['si'], m['error']))
    return ps


def _load_edge_info(edge_file):
    with open(edge_file, 'r') as f:
        return {int(k): v for k, v in json.load(f).items()}


def _load_ps_raw(deduped_dir, pids):
    ps_raw = {}
    for pid in pids:
        sides = []
        for si in range(4):
            json_path = os.path.join(deduped_dir, f'side_{pid}_{si}.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    sides.append(json.load(f))
            else:
                sides.append(None)
        ps_raw[pid] = sides
    return ps_raw


def _load_piece_images(color_dir, pids):
    images = {}
    alphas = {}
    for pid in pids:
        color_path = os.path.join(color_dir, f'piece_{pid}.png')
        if not os.path.exists(color_path):
            continue
        img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            images[pid] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            alphas[pid] = np.full_like(img, 255, dtype=np.uint8)
        elif img.shape[2] == 4:
            images[pid] = img[:, :, :3]
            alphas[pid] = img[:, :, 3]
        else:
            images[pid] = img
            alphas[pid] = np.full(img.shape[:2], 255, dtype=np.uint8)
    return images, alphas


def _prepare_piece_at_cell(pid, ori, cell_x, cell_y, cell_w, cell_h,
                           piece_img, piece_alpha, target_h, target_w):
    angle_deg = ORI_TO_ANGLE.get(ori, 0)

    h_img, w_img = piece_img.shape[:2]
    ic = (w_img / 2.0, h_img / 2.0)

    cos_r = math.cos(math.radians(angle_deg))
    sin_r = math.sin(math.radians(angle_deg))
    corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
    img_pts = []
    for cx, cy in corners:
        dx = cx - ic[0]
        dy = cy - ic[1]
        ox = dx * cos_r - dy * sin_r + ic[0]
        oy = dx * sin_r + dy * cos_r + ic[1]
        img_pts.append((ox, oy))

    img_min_x = min(p[0] for p in img_pts)
    img_min_y = min(p[1] for p in img_pts)
    img_max_x = max(p[0] for p in img_pts)
    img_max_y = max(p[1] for p in img_pts)

    out_w = int(math.ceil(img_max_x - img_min_x)) + 2
    out_h = int(math.ceil(img_max_y - img_min_y)) + 2

    scale_x = cell_w / out_w if out_w > 0 else 1.0
    scale_y = cell_h / out_h if out_h > 0 else 1.0
    scale = min(scale_x, scale_y)

    new_w = max(1, int(out_w * scale))
    new_h = max(1, int(out_h * scale))

    cos_neg = math.cos(math.radians(-angle_deg))
    sin_neg = math.sin(math.radians(-angle_deg))
    sm_x = (img_min_x - ic[0]) * scale
    sm_y = (img_min_y - ic[1]) * scale

    a = cos_neg * scale
    b = -sin_neg * scale
    c = cos_neg * sm_x - sin_neg * sm_y + ic[0]
    d = sin_neg * scale
    e = cos_neg * scale
    f = sin_neg * sm_x + cos_neg * sm_y + ic[1]

    M_pil = np.array([[a, b, c], [d, e, f], [0, 0, 1]], dtype=np.float64)
    M_aff = np.linalg.inv(M_pil)[:2, :]

    piece_rot = cv2.warpAffine(piece_img, M_aff, (new_w, new_h),
                                flags=cv2.INTER_AREA,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    alpha_rot = cv2.warpAffine(piece_alpha, M_aff, (new_w, new_h),
                                flags=cv2.INTER_AREA,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX * 2 + 1, ERODE_PX * 2 + 1))
    mask_eroded = cv2.erode((alpha_rot > 128).astype(np.uint8), erode_kernel) > 0
    mask_raw = alpha_rot > 128
    piece_gray = cv2.cvtColor(piece_rot, cv2.COLOR_BGR2GRAY)

    paste_x = cell_x + (cell_w - new_w) / 2.0
    paste_y = cell_y + (cell_h - new_h) / 2.0

    return piece_gray, mask_eroded, mask_raw, piece_rot, (paste_x, paste_y), (new_w, new_h)


def _ncc_search(piece_gray, mask_eroded, paste_pos, size, target_aligned):
    paste_x, paste_y = paste_pos
    out_w, out_h = size
    th, tw = target_aligned.shape[:2]

    tx1 = max(0, int(paste_x) - SEARCH_MARGIN)
    ty1 = max(0, int(paste_y) - SEARCH_MARGIN)
    tx2 = min(tw, int(paste_x + out_w) + SEARCH_MARGIN)
    ty2 = min(th, int(paste_y + out_h) + SEARCH_MARGIN)
    target_region_gray = cv2.cvtColor(
        target_aligned[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2GRAY)

    base_x = int(paste_x) - tx1
    base_y = int(paste_y) - ty1

    best_score = -999
    best_dx = 0
    best_dy = 0

    for dy in range(-SEARCH_MARGIN, SEARCH_MARGIN + 1, COARSE_STEP):
        for dx in range(-SEARCH_MARGIN, SEARCH_MARGIN + 1, COARSE_STEP):
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

    fine_range = COARSE_STEP + 1
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
        target_aligned[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2GRAY)

    best_angle = 0.0
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX * 2 + 1, ERODE_PX * 2 + 1))

    if best_score >= 0.3:
        angles = np.arange(-ROTATION_RANGE, ROTATION_RANGE + 0.01, ROTATION_STEP)
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
                rot_mask = cv2.erode(rot_alpha, erode_kernel) > 0
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

    hm_score = best_score
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
        final_mask = cv2.erode(rot_alpha, erode_kernel) > 0
        final_gray = final_gray[py1:py1 + ph_, px1:px1 + pw_]
        final_mask = final_mask[py1:py1 + ph_, px1:px1 + pw_]

    if final_mask.sum() >= 100:
        p = final_gray[final_mask].astype(np.float64)
        t = target_at_best_gray[final_mask].astype(np.float64)
        hm_map = _histogram_match_cdf(p, t)
        pm = hm_map[p.astype(np.uint8)].astype(np.float64)
        pm_n = pm - pm.mean()
        t_n = t - t.mean()
        denom = np.sqrt(np.sum(pm_n ** 2) * np.sum(t_n ** 2))
        hm_score = float(np.sum(pm_n * t_n) / denom) if denom > 1e-6 else 0.0

    return max(best_score, hm_score), best_dx, best_dy, best_angle


class TargetedSolver:
    def __init__(self, target_image_path, solution_dir, output_root,
                 confidence_threshold=NCC_ACCEPT):
        self.output_root = output_root
        self.deduped_dir = os.path.join(output_root, DEDUPED_DIR)
        self.color_dir = os.path.join(output_root, '2_piece_colors')
        self.confidence_threshold = confidence_threshold
        self.solution_dir = solution_dir

        self.pw, self.ph, self.placed = load_solution(solution_dir)

        report_path = os.path.join(os.path.dirname(solution_dir), 'target_match_report.json')
        with open(report_path, 'r') as f:
            self.report = json.load(f)

        connectivity_file = os.path.join(output_root, '5_connectivity', 'connectivity.json')
        self.connectivity = _load_connectivity(connectivity_file)

        all_pids = set(self.connectivity.keys())
        self.ps_raw = _load_ps_raw(self.deduped_dir, all_pids)
        self.piece_images, self.piece_alphas = _load_piece_images(self.color_dir, all_pids)

        matcher = TargetMatcher(
            target_image_path=target_image_path,
            pw=self.pw, ph=self.ph,
            placed=self.placed,
            output_dir=os.path.dirname(solution_dir),
            output_root=output_root,
        )
        matcher.rectify_target()
        self.target_aligned = matcher.target_aligned
        self._grid_ox = matcher._grid_offset_x
        self._grid_oy = matcher._grid_offset_y
        self._cell_w = matcher._cell_w
        self._cell_h = matcher._cell_h

        self.board = Board(self.pw, self.ph)
        self.fixed_pids = set()
        for pid, info in self.placed.items():
            score = self.report.get(str(pid), {}).get('score', 0)
            if score >= self.confidence_threshold:
                fits = self.ps_raw.get(pid, [[], [], [], []])
                self.board.place(pid, fits, info['gx'], info['gy'], info['orientation'])
                self.fixed_pids.add(pid)

        full_board = Board(self.pw, self.ph)
        for pid, info in self.placed.items():
            fits = self.ps_raw.get(pid, [[], [], [], []])
            full_board.place(pid, fits, info['gx'], info['gy'], info['orientation'])
        self._orig_transforms, _, self._orig_canvas_info = compute_piece_transforms(
            full_board, self.deduped_dir)

        raw_ass_path = os.path.join(os.path.dirname(solution_dir), '_match_assembly.png')
        raw_ass = cv2.imread(raw_ass_path)
        if raw_ass is not None:
            _, self._orig_resize_scale = _resize_to_max(raw_ass, 2000)
        else:
            self._orig_resize_scale = 1.0

        self.ncc_data = {}
        for pid_str, r in self.report.items():
            pid = int(pid_str)
            if pid in self.fixed_pids:
                self.ncc_data[pid] = {
                    'dx': r.get('dx', 0), 'dy': r.get('dy', 0),
                    'angle': r.get('angle', 0), 'score': r.get('score', 0),
                }

        print(f"Initial: {self.board.placed_count}/{len(self.placed)} fixed "
              f"(threshold={self.confidence_threshold})")

    def _get_used_pids(self):
        used = set()
        for gy in range(self.ph):
            for gx in range(self.pw):
                cell = self.board.get(gx, gy)
                if cell is not None:
                    used.add(cell[0])
        return used

    def _find_adjacent_empty(self):
        empty = []
        for gy in range(self.ph):
            for gx in range(self.pw):
                if self.board.get(gx, gy) is not None:
                    continue
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    if self.board.get(gx + dx, gy + dy) is not None:
                        empty.append((gx, gy))
                        break
        return empty

    def _get_candidates(self, gx, gy):
        constraints = []
        for d, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            cell = self.board.get(gx + dx, gy + dy)
            if cell is None:
                continue
            adj_pid, _, adj_ori = cell
            s_adj = (d - adj_ori) % 4
            if adj_pid not in self.connectivity:
                continue
            back_d = (d + 2) % 4
            constraint = []
            for match_pid, match_si, error in self.connectivity[adj_pid][s_adj]:
                required_ori = (back_d - match_si) % 4
                constraint.append((match_pid, required_ori, error))
            constraints.append(constraint)

        if not constraints:
            return []

        if len(constraints) == 1:
            return list(set((pid, ori) for pid, ori, _ in constraints[0]))

        pid_oris = {}
        for constraint in constraints:
            for pid, ori, _ in constraint:
                pid_oris.setdefault(pid, set()).add(ori)
        return [(pid, oris.pop()) for pid, oris in pid_oris.items() if len(oris) == 1]

    def _try_piece(self, pid, ori, gx, gy):
        if pid not in self.piece_images:
            return 0.0, 0, 0, 0.0

        cell_x = self._grid_ox + gx * self._cell_w
        cell_y = self._grid_oy + gy * self._cell_h
        th, tw = self.target_aligned.shape[:2]

        result = _prepare_piece_at_cell(
            pid, ori, cell_x, cell_y, self._cell_w, self._cell_h,
            self.piece_images[pid], self.piece_alphas[pid], th, tw)
        if result[0] is None:
            return 0.0, 0, 0, 0.0

        piece_gray, mask_eroded, _, _, paste_pos, size = result
        score, dx, dy, angle = _ncc_search(
            piece_gray, mask_eroded, paste_pos, size, self.target_aligned)
        return score, dx, dy, angle

    def _draw_progress(self, iteration, output_dir):
        th, tw = self.target_aligned.shape[:2]
        canvas = self.target_aligned.copy()

        transforms = self._orig_transforms
        canvas_info = self._orig_canvas_info
        if not transforms or not canvas_info:
            cv2.imwrite(os.path.join(output_dir, f'progress_iter{iteration}.png'), canvas)
            return

        min_x = canvas_info['min_x']
        min_y = canvas_info['min_y']
        margin = max(canvas_info['max_x'] - min_x, canvas_info['max_y'] - min_y) * 0.05
        header_h = 60
        gen_scale = 1.0
        rs = self._orig_resize_scale

        for gy in range(self.ph):
            for gx in range(self.pw):
                cell = self.board.get(gx, gy)
                if cell is None:
                    continue
                pid, _, ori = cell

                if pid not in self.piece_images or pid not in transforms:
                    continue

                rotation, translation, ic = transforms[pid]
                piece_bgr = self.piece_images[pid]
                alpha_raw = self.piece_alphas[pid]
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

                out_w_gen = int(math.ceil((img_max_x - img_min_x) * gen_scale)) + 2
                out_h_gen = int(math.ceil((img_max_y - img_min_y) * gen_scale)) + 2

                cos_neg = math.cos(-rotation)
                sin_neg = math.sin(-rotation)
                sm_x = (img_min_x - ic[0] - translation[0]) * gen_scale
                sm_y = (img_min_y - ic[1] - translation[1]) * gen_scale

                a_v = cos_neg * gen_scale
                b_v = -sin_neg * gen_scale
                c_v = cos_neg * sm_x - sin_neg * sm_y + ic[0]
                d_v = sin_neg * gen_scale
                e_v = cos_neg * gen_scale
                f_v = sin_neg * sm_x + cos_neg * sm_y + ic[1]

                M_pil = np.array([[a_v, b_v, c_v], [d_v, e_v, f_v], [0, 0, 1]], dtype=np.float64)
                M_aff = np.linalg.inv(M_pil)[:2, :]

                piece_gen = cv2.warpAffine(piece_bgr, M_aff, (out_w_gen, out_h_gen),
                                            flags=cv2.INTER_AREA,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                alpha_gen = cv2.warpAffine(alpha_raw, M_aff, (out_w_gen, out_h_gen),
                                            flags=cv2.INTER_AREA,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                out_w = max(1, int(out_w_gen * rs))
                out_h = max(1, int(out_h_gen * rs))
                piece_final = cv2.resize(piece_gen, (out_w, out_h), interpolation=cv2.INTER_AREA)
                alpha_final = cv2.resize(alpha_gen, (out_w, out_h), interpolation=cv2.INTER_AREA)

                mask_raw = alpha_final > 128

                nd = self.ncc_data.get(pid, {})
                dx = nd.get('dx', 0)
                dy = nd.get('dy', 0)
                angle = nd.get('angle', 0)

                if abs(angle) > 0.01:
                    M_rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), angle, 1.0)
                    piece_final = cv2.warpAffine(piece_final, M_rot, (out_w, out_h),
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                    alpha_u8 = (mask_raw.astype(np.uint8)) * 255
                    alpha_rot = cv2.warpAffine(alpha_u8, M_rot, (out_w, out_h),
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    mask_raw = alpha_rot > 128

                paste_x = (img_min_x - min_x + margin) * gen_scale * rs
                paste_y = ((img_min_y - min_y + margin) * gen_scale + header_h) * rs

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
                region = canvas[py1:py2, px1:px2]
                piece_region = piece_final[sy1:sy2, sx1:sx2]
                canvas[py1:py2, px1:px2] = (
                    region.astype(np.float32) * (1 - blend3) +
                    piece_region.astype(np.float32) * blend3
                ).astype(np.uint8)

        path = os.path.join(output_dir, f'progress_iter{iteration}.png')
        cv2.imwrite(path, canvas)
        print(f"  Saved {path}")

    def solve(self):
        print(f"\n{'='*60}")
        print(f"Targeted Solve (threshold={self.confidence_threshold})")
        print(f"{'='*60}")

        output_dir = os.path.join(os.path.dirname(self.solution_dir), 'targeted_solve')
        os.makedirs(output_dir, exist_ok=True)

        self._draw_progress(0, output_dir)

        changed = True
        iteration = 0
        t0 = time.time()

        while changed:
            changed = False
            iteration += 1
            empty_positions = self._find_adjacent_empty()
            used_pids = self._get_used_pids()

            print(f"\nIter {iteration}: {len(empty_positions)} empty, {self.board.placed_count} placed")

            for gx, gy in empty_positions:
                candidates = self._get_candidates(gx, gy)
                candidates = [(pid, ori) for pid, ori in candidates
                              if pid not in used_pids and pid in self.piece_images]
                if not candidates:
                    continue

                best_score = 0
                best_candidate = None
                best_params = (0, 0, 0.0)

                for pid, ori in candidates:
                    score, dx, dy, angle = self._try_piece(pid, ori, gx, gy)
                    if score > best_score:
                        best_score = score
                        best_candidate = (pid, ori)
                        best_params = (dx, dy, angle)

                if best_score >= self.confidence_threshold and best_candidate:
                    pid, ori = best_candidate
                    fits = self.ps_raw.get(pid, [[], [], [], []])
                    self.board.place(pid, fits, gx, gy, ori)
                    used_pids.add(pid)
                    dx, dy, angle = best_params
                    self.ncc_data[pid] = {
                        'dx': dx, 'dy': dy, 'angle': angle, 'score': best_score,
                    }
                    changed = True
                    print(f"  + {pid} at ({gx},{gy}) ori={ori} NCC={best_score:.4f}")

            self._draw_progress(iteration, output_dir)

        elapsed = time.time() - t0
        print(f"\nDone: {self.board.placed_count}/{self.pw * self.ph} in "
              f"{iteration} iters ({elapsed:.1f}s)")
        return self.board

    def save_results(self):
        output_dir = os.path.join(os.path.dirname(self.solution_dir), 'targeted_solve')
        os.makedirs(output_dir, exist_ok=True)

        board_output.generate_solution_grid(self.board, output_dir)

        try:
            generate_assembly_png(self.board, self.deduped_dir,
                                  os.path.dirname(self.color_dir),
                                  os.path.join(output_dir, 'assembly.png'))
        except Exception as e:
            print(f"  Assembly failed: {e}")

        report = {}
        for gy in range(self.ph):
            for gx in range(self.pw):
                cell = self.board.get(gx, gy)
                if cell is not None:
                    pid, _, ori = cell
                    nd = self.ncc_data.get(pid, {})
                    report[str(pid)] = {
                        'grid_pos': [gx, gy],
                        'orientation': ori,
                        'score': nd.get('score', 0),
                        'dx': nd.get('dx', 0),
                        'dy': nd.get('dy', 0),
                        'angle': nd.get('angle', 0),
                        'source': 'original' if pid in self.fixed_pids else 'targeted_solve',
                    }

        with open(os.path.join(output_dir, 'targeted_solve_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  Saved report to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Targeted puzzle solver using target image NCC')
    parser.add_argument('--target', required=True, help='Path to target image')
    parser.add_argument('--solution', required=True, help='Path to solution directory')
    parser.add_argument('--output-root', default=None, help='Root output directory')
    parser.add_argument('--threshold', type=float, default=NCC_ACCEPT)
    args = parser.parse_args()

    output_root = args.output_root or os.path.join(os.path.dirname(args.solution), '..', '..')
    output_root = os.path.abspath(output_root)

    solver = TargetedSolver(
        target_image_path=args.target,
        solution_dir=args.solution,
        output_root=output_root,
        confidence_threshold=args.threshold,
    )
    solver.solve()
    solver.save_results()


if __name__ == '__main__':
    main()
