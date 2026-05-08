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
from common import output as board_output, util
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

        target_aligned_path = os.path.join(os.path.dirname(solution_dir), 'target_aligned.png')
        if target_image_path and os.path.exists(target_image_path):
            matcher = TargetMatcher(
                target_image_path=target_image_path,
                pw=self.pw, ph=self.ph,
                placed=self.placed,
                output_dir=os.path.dirname(solution_dir),
                output_root=output_root,
            )
            matcher.rectify_target()
            self.target_aligned = matcher.target_aligned
        elif os.path.exists(target_aligned_path):
            self.target_aligned = cv2.imread(target_aligned_path)
            print(f"  Loaded target_aligned from {target_aligned_path}")
        else:
            raise FileNotFoundError(f"Need either target image or target_aligned.png")

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
        constraint_sets = []
        for d, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            cell = self.board.get(gx + dx, gy + dy)
            if cell is None:
                continue
            adj_pid, _, adj_ori = cell
            back_d = (d + 2) % 4
            s_adj = (back_d - adj_ori) % 4
            if adj_pid not in self.connectivity:
                continue
            s = set()
            for match_pid, match_si, error in self.connectivity[adj_pid][s_adj]:
                required_ori = (d - match_si) % 4
                s.add((match_pid, required_ori))
            constraint_sets.append(s)

        if not constraint_sets:
            return []

        result = constraint_sets[0]
        for s in constraint_sets[1:]:
            result = result & s
        return list(result)

    def _compute_transform(self, pid, ori, gx, gy):
        cand_sides = self.ps_raw.get(pid)
        if not cand_sides or cand_sides[0] is None:
            return None
        cand_ic = tuple(cand_sides[0]['incenter'])
        new_sides = util.rotate_list([0, 1, 2, 3], -ori)

        rotations = []
        translation_samples = []
        for d, (ddx, ddy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            cell_n = self.board.get(gx + ddx, gy + ddy)
            if cell_n is None:
                continue
            adj_pid, _, adj_ori = cell_n
            back_d = (d + 2) % 4
            s_adj = (back_d - adj_ori) % 4

            if adj_pid not in self._orig_transforms:
                continue
            rot_n, trans_n, ic_n = self._orig_transforms[adj_pid]

            adj_side = self.ps_raw.get(adj_pid, [None]*4)[s_adj]
            if adj_side is None:
                continue
            adj_verts = adj_side['vertices']
            adj_rotated = [util.rotate(v, ic_n, rot_n) for v in adj_verts]
            adj_translated = [(r[0] + trans_n[0], r[1] + trans_n[1]) for r in adj_rotated]

            adj_angle = math.atan2(
                adj_translated[-1][1] - adj_translated[0][1],
                adj_translated[-1][0] - adj_translated[0][0]
            ) % (2 * math.pi)

            cand_phys_side = new_sides[d]
            cand_side = cand_sides[cand_phys_side]
            if cand_side is None:
                continue
            cand_verts = cand_side['vertices']
            cand_angle = math.atan2(
                cand_verts[-1][1] - cand_verts[0][1],
                cand_verts[-1][0] - cand_verts[0][0]
            ) % (2 * math.pi)

            rot = adj_angle - cand_angle - math.pi
            rotations.append(rot)

            cand_rotated = [util.rotate(v, cand_ic, rot) for v in cand_verts]
            sample = util.subtract(adj_translated[-1], cand_rotated[0])
            translation_samples.append(sample)

        if not rotations:
            return None
        rotation = util.average_angles(rotations)
        translation = util.multimidpoint(translation_samples)
        return rotation, translation, cand_ic

    def _try_piece(self, pid, ori, gx, gy):
        if pid not in self.piece_images:
            return 0.0, 0, 0, 0.0

        transform = self._compute_transform(pid, ori, gx, gy)
        if transform is None:
            return 0.0, 0, 0, 0.0
        rotation, translation, ic = transform

        ci = self._orig_canvas_info
        min_x = ci['min_x']
        min_y = ci['min_y']
        margin = max(ci['max_x'] - min_x, ci['max_y'] - min_y) * 0.05
        header_h = 60
        gen_scale = 1.0
        rs = self._orig_resize_scale

        piece_bgr = self.piece_images[pid]
        alpha_raw = self.piece_alphas[pid]
        h_img, w_img = piece_bgr.shape[:2]

        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
        img_pts = []
        for cx, cy in corners:
            ddx = cx - ic[0]
            ddy = cy - ic[1]
            ox = ddx * cos_r - ddy * sin_r + ic[0] + translation[0]
            oy = ddx * sin_r + ddy * cos_r + ic[1] + translation[1]
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
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX * 2 + 1, ERODE_PX * 2 + 1))
        mask_eroded = cv2.erode(mask_raw.astype(np.uint8), erode_kernel) > 0
        piece_gray = cv2.cvtColor(piece_final, cv2.COLOR_BGR2GRAY)

        paste_x = (img_min_x - min_x + margin) * gen_scale * rs
        paste_y = ((img_min_y - min_y + margin) * gen_scale + header_h) * rs

        score, dx, dy, angle = _ncc_search(
            piece_gray, mask_eroded, (paste_x, paste_y), (out_w, out_h), self.target_aligned)
        return score, dx, dy, angle

    def _draw_progress(self, iteration, output_dir):
        th, tw = self.target_aligned.shape[:2]
        canvas = self.target_aligned.copy()

        canvas_info = self._orig_canvas_info
        if not canvas_info:
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

                if pid not in self.piece_images or pid not in self._orig_transforms:
                    continue

                rotation, translation, ic = self._orig_transforms[pid]

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
                    t = self._compute_transform(pid, ori, gx, gy)
                    if t is not None:
                        self._orig_transforms[pid] = t
                    changed = True
                    print(f"  + {pid} at ({gx},{gy}) ori={ori} NCC={best_score:.4f}")

            self._draw_progress(iteration, output_dir)

        elapsed = time.time() - t0
        print(f"\nDone connectivity phase: {self.board.placed_count}/{self.pw * self.ph} in "
              f"{iteration} iters ({elapsed:.1f}s)")

        used_pids = self._get_used_pids()
        all_pids = set(self.connectivity.keys())
        missing = sorted(p for p in all_pids if p not in used_pids)

        if missing:
            print(f"\n{'='*60}")
            print(f"Relaxed search (threshold=0.7, no connectivity constraint)")
            print(f"{'='*60}")
            relaxed_threshold = 0.7
            changed_relaxed = True
            while changed_relaxed:
                changed_relaxed = False
                used_pids = self._get_used_pids()
                missing_now = sorted(p for p in all_pids if p not in used_pids)
                if not missing_now:
                    break
                empty_positions = self._find_adjacent_empty()
                if not empty_positions:
                    break

                for pid in missing_now:
                    if pid not in self.piece_images:
                        continue
                    best_score = 0
                    best_pos = None
                    for gx, gy in empty_positions:
                        if self.board.get(gx, gy) is not None:
                            continue
                        for ori in range(4):
                            score, dx, dy, angle = self._try_piece(pid, ori, gx, gy)
                            if score > best_score:
                                best_score = score
                                best_pos = (gx, gy, ori, dx, dy, angle)

                    if best_score >= relaxed_threshold and best_pos:
                        gx, gy, ori, dx, dy, angle = best_pos
                        fits = self.ps_raw.get(pid, [[], [], [], []])
                        self.board.place(pid, fits, gx, gy, ori)
                        used_pids.add(pid)
                        self.ncc_data[pid] = {
                            'dx': dx, 'dy': dy, 'angle': angle, 'score': best_score,
                        }
                        t = self._compute_transform(pid, ori, gx, gy)
                        if t is not None:
                            self._orig_transforms[pid] = t
                        changed_relaxed = True
                        print(f"  + {pid} at ({gx},{gy}) ori={ori} NCC={best_score:.4f} [relaxed]")
                        break

                if changed_relaxed:
                    self._draw_progress(iteration + 100, output_dir)

            used_pids = self._get_used_pids()
            still_missing = sorted(p for p in all_pids if p not in used_pids)
            if still_missing:
                print(f"\n--- Still missing ({len(still_missing)}), max NCC below 0.7 ---")
                for pid in still_missing:
                    if pid not in self.piece_images:
                        print(f"  #{pid}: no color image")
                        continue
                    best_score = 0
                    best_pos = None
                    for gy in range(self.ph):
                        for gx in range(self.pw):
                            if self.board.get(gx, gy) is not None:
                                continue
                            for ori in range(4):
                                score, _, _, _ = self._try_piece(pid, ori, gx, gy)
                                if score > best_score:
                                    best_score = score
                                    best_pos = (gx, gy, ori)
                    if best_pos:
                        print(f"  #{pid}: max NCC={best_score:.4f} at ({best_pos[0]},{best_pos[1]}) ori={best_pos[2]}")
                    else:
                        print(f"  #{pid}: no valid position found")

        print(f"\nFinal: {self.board.placed_count}/{self.pw * self.ph}")

        self._draw_transparent(output_dir)

        return self.board

    def _draw_transparent(self, output_dir):
        canvas_info = self._orig_canvas_info
        if not canvas_info:
            return
        min_x = canvas_info['min_x']
        min_y = canvas_info['min_y']
        margin = max(canvas_info['max_x'] - min_x, canvas_info['max_y'] - min_y) * 0.05
        header_h = 60
        gen_scale = 1.0
        rs = self._orig_resize_scale

        canvas_w = int((canvas_info['max_x'] - min_x + 2 * margin) * gen_scale * rs)
        canvas_h = int(((canvas_info['max_y'] - min_y + 2 * margin) * gen_scale + header_h) * rs)
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

        for gy in range(self.ph):
            for gx in range(self.pw):
                cell = self.board.get(gx, gy)
                if cell is None:
                    continue
                pid, _, ori = cell
                if pid not in self.piece_images or pid not in self._orig_transforms:
                    continue

                rotation, translation, ic = self._orig_transforms[pid]
                piece_bgr = self.piece_images[pid]
                alpha_raw = self.piece_alphas[pid]
                h_img, w_img = piece_bgr.shape[:2]

                cos_r = math.cos(rotation)
                sin_r = math.sin(rotation)
                corners = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
                img_pts = []
                for cx, cy in corners:
                    ddx = cx - ic[0]
                    ddy = cy - ic[1]
                    ox = ddx * cos_r - ddy * sin_r + ic[0] + translation[0]
                    oy = ddx * sin_r + ddy * cos_r + ic[1] + translation[1]
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

                nd = self.ncc_data.get(pid, {})
                ddx = nd.get('dx', 0)
                ddy = nd.get('dy', 0)
                angle = nd.get('angle', 0)

                if abs(angle) > 0.01:
                    M_rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), angle, 1.0)
                    piece_final = cv2.warpAffine(piece_final, M_rot, (out_w, out_h),
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                    alpha_final = cv2.warpAffine(alpha_final, M_rot, (out_w, out_h),
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                paste_x = (img_min_x - min_x + margin) * gen_scale * rs
                paste_y = ((img_min_y - min_y + margin) * gen_scale + header_h) * rs

                fx = int(paste_x) + int(ddx)
                fy = int(paste_y) + int(ddy)

                px1 = max(0, fx)
                py1 = max(0, fy)
                px2 = min(canvas_w, fx + out_w)
                py2 = min(canvas_h, fy + out_h)
                sx1 = px1 - fx
                sy1 = py1 - fy
                sx2 = sx1 + (px2 - px1)
                sy2 = sy1 + (py2 - py1)
                if sx2 <= sx1 or sy2 <= sy1:
                    continue

                a_region = alpha_final[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
                a3 = np.stack([a_region] * 3, axis=2)
                a4 = a_region

                existing = canvas[py1:py2, px1:px2]
                existing_alpha = existing[:, :, 3].astype(np.float32) / 255.0

                out_a = a_region + existing_alpha * (1 - a_region)
                out_a = np.clip(out_a, 0, 1)
                out_a3 = np.stack([out_a] * 3, axis=2)

                src_rgb = piece_final[sy1:sy2, sx1:sx2].astype(np.float32)
                dst_rgb = existing[:, :, :3].astype(np.float32)
                dst_a = existing_alpha[:, :, np.newaxis]

                out_rgb = (src_rgb * a3 + dst_rgb * dst_a * (1 - a3)) / np.maximum(out_a3, 1e-6)

                canvas[py1:py2, px1:px2, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
                canvas[py1:py2, px1:px2, 3] = np.clip(out_a * 255, 0, 255).astype(np.uint8)

        used_pids = self._get_used_pids()
        all_pids = set(self.connectivity.keys())
        missing = sorted(p for p in all_pids if p not in used_pids)

        if not missing:
            path = os.path.join(output_dir, 'puzzle_transparent.png')
            cv2.imwrite(path, canvas)
            print(f"  Saved {path} (complete, no missing pieces)")
            return

        cell_sz = 120
        cols_missing = min(len(missing), 5)
        missing_w = cols_missing * (cell_sz + 10) + 20
        missing_h = ((len(missing) - 1) // cols_missing + 1) * (cell_sz + 30) + 20

        total_w = canvas_w + 20 + missing_w
        total_h = max(canvas_h, missing_h)
        vis = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        vis[:, :, :] = 30

        vis[:canvas_h, :canvas_w] = canvas[:, :, :3]

        for i, pid in enumerate(missing):
            col = i % cols_missing
            row = i // cols_missing
            x0 = canvas_w + 20 + col * (cell_sz + 10) + 10
            y0 = row * (cell_sz + 30) + 10

            cv2.putText(vis, f"#{pid}", (x0 + 5, y0 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if pid in self.piece_images:
                bgr = self.piece_images[pid].copy()
                alpha = self.piece_alphas[pid]
                bgr[alpha < 128] = 30
                h, w = bgr.shape[:2]
                scale = min((cell_sz - 4) / w, (cell_sz - 4) / h)
                rw, rh = int(w * scale), int(h * scale)
                bgr = cv2.resize(bgr, (rw, rh))
                px = x0 + (cell_sz - rw) // 2
                py = y0 + 22 + (cell_sz - rh) // 2
                vis[py:py + rh, px:px + rw] = bgr

        path = os.path.join(output_dir, 'puzzle_transparent.png')
        cv2.imwrite(path, canvas)
        print(f"  Saved {path}")

        path2 = os.path.join(output_dir, 'puzzle_result_with_missing.png')
        cv2.imwrite(path2, vis)
        print(f"  Saved {path2} ({len(missing)} missing: {missing})")

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
    parser.add_argument('--target', default=None, help='Path to target image (optional if target_aligned.png exists)')
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
