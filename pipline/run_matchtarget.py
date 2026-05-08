import os
import sys
import json
import math
import argparse
import re

import cv2
import numpy as np
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src'))

from common.config import DEDUPED_DIR, SOLUTION_DIR


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
                meta = json.load(f)
            pw = meta.get('width', 0)
            ph = meta.get('height', 0)

    grid_path = os.path.join(solution_dir, 'solution_grid.txt')
    with open(grid_path, 'r') as f:
        grid_text = f.read()

    placed = {}
    arrow_map = {'^': 0, '>': 1, 'v': 2, '<': 3}
    lines = [l.strip() for l in grid_text.split('\n') if l.strip() and not l.strip().startswith('--')]
    row = 0
    for line in lines:
        tokens = line.split()
        col = 0
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            m = re.match(r'^(\d+)([\^v<>])$', tok)
            if m:
                pid = int(m.group(1))
                ori = arrow_map[m.group(2)]
                placed[pid] = {
                    'gx': col, 'gy': row,
                    'orientation': ori,
                }
                col += 1
            elif tok == '-':
                col += 1
            else:
                col += 1
            i += 1
        row += 1

    if pw == 0:
        pw = max(p['gx'] for p in placed.values()) + 1 if placed else 0
    if ph == 0:
        ph = max(p['gy'] for p in placed.values()) + 1 if placed else 0

    return pw, ph, placed


def _order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    top = ordered[ordered[:, 1] <= center[1]]
    bot = ordered[ordered[:, 1] > center[1]]
    if len(top) < 2 or len(bot) < 2:
        return pts
    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(bot[:, 0])]
    return np.array([top[0], top[-1], bot[-1], bot[0]], dtype=np.float32)


def _rotate_image(img, rotation):
    if rotation == 0:
        return img.copy()
    elif rotation == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == 3:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img.copy()


def _flip_image(img, flip):
    if flip:
        return cv2.flip(img, 1)
    return img.copy()


def _compute_hsv_hist(img_bgr, mask=None):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if mask is None:
        mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255
    hist = cv2.calcHist([hsv], [0, 1], mask, [36, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _hist_correlation(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)


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
        if piece_color_dir:
            self.color_dir = piece_color_dir
        elif output_root:
            self.color_dir = os.path.join(output_root, '2_piece_colors')
        else:
            self.color_dir = self._find_dir('2_piece_colors')
        if deduped_dir:
            self.deduped_dir = deduped_dir
        elif output_root:
            self.deduped_dir = os.path.join(output_root, DEDUPED_DIR)
        else:
            self.deduped_dir = self._find_dir(DEDUPED_DIR)

        self.target_rectified = None
        self.best_rotation = 0
        self.best_flip = False
        self.match_results = {}
        self.global_transform = None

        self._piece_imgs = {}
        self._piece_masks = {}
        self._load_piece_images()

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

    def _load_piece_images(self):
        if not os.path.exists(self.color_dir):
            print(f"  WARNING: piece color dir not found: {self.color_dir}")
            return
        for pid in self.placed:
            path = os.path.join(self.color_dir, f'piece_{pid}.png')
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 4:
                        alpha = img[:, :, 3]
                        mask = (alpha > 30).astype(np.uint8) * 255
                        bgr = img[:, :, :3]
                    else:
                        bgr = img
                        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
                    ys, xs = np.where(mask > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        x1, x2 = xs.min(), xs.max()
                        y1, y2 = ys.min(), ys.max()
                        bgr = bgr[y1:y2 + 1, x1:x2 + 1]
                        mask = mask[y1:y2 + 1, x1:x2 + 1]
                    self._piece_imgs[pid] = bgr
                    self._piece_masks[pid] = mask
        print(f"  Loaded {len(self._piece_imgs)} piece images from {self.color_dir}")

    def _detect_puzzle_corners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        best = contours[0]
        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.02 * peri, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            return _order_corners(corners)

        hull = cv2.convexHull(best)
        peri_h = cv2.arcLength(hull, True)
        approx_h = cv2.approxPolyDP(hull, 0.02 * peri_h, True)
        if len(approx_h) == 4:
            corners = approx_h.reshape(4, 2).astype(np.float32)
            return _order_corners(corners)

        x, y, w, h = cv2.boundingRect(best)
        margin = 0.02 * max(w, h)
        corners = np.array([
            [x - margin, y - margin],
            [x + w + margin, y - margin],
            [x + w + margin, y + h + margin],
            [x - margin, y + h + margin],
        ], dtype=np.float32)
        return corners

    def rectify_target(self):
        print("\n--- Phase 1: Target Rectification ---")
        corners = self._detect_puzzle_corners(self.target_raw)
        h, w = self.target_raw.shape[:2]

        target_w = max(w, self.pw * 100)
        target_h = max(h, self.ph * 100)

        if corners is not None:
            print(f"  Detected corners: {corners.tolist()}")
            dst = np.array([
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1],
            ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(corners, dst)
            self.target_rectified = cv2.warpPerspective(
                self.target_raw, M, (target_w, target_h)
            )
            print(f"  Rectified to {target_w}x{target_h}")
        else:
            print("  Corner detection failed, using raw image with bounding rect")
            self.target_rectified = cv2.resize(self.target_raw, (target_w, target_h))

        return self.target_rectified

    def _align_orientation(self):
        print("\n--- Phase 2: Orientation Detection ---")
        img = self.target_rectified
        h, w = img.shape[:2]
        cell_w = w / self.pw
        cell_h = h / self.ph

        best_score = -1
        best_rot = 0
        best_fl = False

        sample_pids = list(self.placed.keys())
        if len(sample_pids) > 20:
            step = len(sample_pids) // 20
            sample_pids = sample_pids[::step]

        for rot in range(4):
            for flip in [False, True]:
                test_img = _rotate_image(img, rot)
                if flip:
                    test_img = _flip_image(test_img, True)
                th, tw = test_img.shape[:2]
                tcw = tw / self.pw
                tch = th / self.ph

                total_score = 0.0
                count = 0
                for pid in sample_pids:
                    if pid not in self._piece_imgs:
                        continue
                    info = self.placed[pid]
                    gx, gy = info['gx'], info['gy']

                    x1 = int(gx * tcw)
                    y1 = int(gy * tch)
                    x2 = int((gx + 1) * tcw)
                    y2 = int((gy + 1) * tch)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(tw, x2), min(th, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    cell_img = test_img[y1:y2, x1:x2]
                    cell_hist = _compute_hsv_hist(cell_img)

                    piece_img = self._piece_imgs[pid]
                    piece_mask = self._piece_masks.get(pid)
                    if piece_mask is not None:
                        piece_hist = _compute_hsv_hist(piece_img, mask=piece_mask)
                    else:
                        piece_hist = _compute_hsv_hist(piece_img)

                    score = _hist_correlation(cell_hist, piece_hist)
                    total_score += max(0, score)
                    count += 1

                if count > 0:
                    avg_score = total_score / count
                else:
                    avg_score = 0

                print(f"  rot={rot}, flip={flip}: avg_score={avg_score:.4f} ({count} pieces)")

                if avg_score > best_score:
                    best_score = avg_score
                    best_rot = rot
                    best_fl = flip

        self.best_rotation = best_rot
        self.best_flip = best_fl
        print(f"  Best orientation: rotation={best_rot}, flip={best_fl}, score={best_score:.4f}")

        self.target_rectified = _rotate_image(self.target_rectified, best_rot)
        if best_fl:
            self.target_rectified = _flip_image(self.target_rectified, True)

        return best_rot, best_fl

    def _match_single_piece(self, pid, info):
        if pid not in self._piece_imgs:
            return None

        gx, gy = info['gx'], info['gy']
        orientation = info['orientation']

        img = self.target_rectified
        th, tw = img.shape[:2]
        cell_w = tw / self.pw
        cell_h = th / self.ph

        x1 = int(gx * cell_w)
        y1 = int(gy * cell_h)
        x2 = int((gx + 1) * cell_w)
        y2 = int((gy + 1) * cell_h)

        margin_ratio = 0.3
        mx = int(cell_w * margin_ratio)
        my = int(cell_h * margin_ratio)
        sx1 = max(0, x1 - mx)
        sy1 = max(0, y1 - my)
        sx2 = min(tw, x2 + mx)
        sy2 = min(th, y2 + my)

        search_region = img[sy1:sy2, sx1:sx2]
        if search_region.size == 0:
            return None

        piece_img = self._piece_imgs[pid]
        piece_mask = self._piece_masks.get(pid)
        rotated_piece = np.rot90(piece_img, k=orientation)
        rotated_mask = np.rot90(piece_mask, k=orientation) if piece_mask is not None else None

        target_cell_h = y2 - y1
        target_cell_w = x2 - x1
        ph_img, pw_img = rotated_piece.shape[:2]

        scale = min(target_cell_w * 1.1 / pw_img, target_cell_h * 1.1 / ph_img, 2.0)
        if abs(scale - 1.0) > 0.05:
            new_w = max(1, int(pw_img * scale))
            new_h = max(1, int(ph_img * scale))
            rotated_piece = cv2.resize(rotated_piece, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if rotated_mask is not None:
                rotated_mask = cv2.resize(rotated_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            ph_img, pw_img = new_h, new_w

        sr_h, sr_w = search_region.shape[:2]

        if pw_img >= sr_w or ph_img >= sr_h:
            region = img[max(0, y1):min(th, y2), max(0, x1):min(tw, x2)]
            if region.size == 0:
                return None
            piece_resized = cv2.resize(rotated_piece, (region.shape[1], region.shape[0]))
            if rotated_mask is not None:
                mask_resized = cv2.resize(rotated_mask, (region.shape[1], region.shape[0]))
                ncc_score = self._compute_masked_ncc(region, piece_resized, mask_resized)
            else:
                ncc_score = self._compute_ncc(region, piece_resized)
            return {
                'score': ncc_score,
                'ncc_score': ncc_score,
                'hist_score': 0.0,
                'raw_offset': (0.0, 0.0),
                'ncc_map': None,
                'search_origin': (sx1, sy1),
            }

        if rotated_mask is not None:
            ncc_map = cv2.matchTemplate(
                search_region, rotated_piece, cv2.TM_CCORR_NORMED,
                mask=rotated_mask
            )
        else:
            ncc_map = cv2.matchTemplate(
                search_region, rotated_piece, cv2.TM_CCORR_NORMED
            )

        _, max_val, _, max_loc = cv2.minMaxLoc(ncc_map)

        piece_center_x = gx * cell_w + cell_w / 2
        piece_center_y = gy * cell_h + cell_h / 2
        match_center_x = sx1 + max_loc[0] + pw_img / 2
        match_center_y = sy1 + max_loc[1] + ph_img / 2
        dx = match_center_x - piece_center_x
        dy = match_center_y - piece_center_y

        region_at_match = search_region[
            max_loc[1]:max_loc[1] + ph_img,
            max_loc[0]:max_loc[0] + pw_img
        ]
        if rotated_mask is not None:
            hist_target = _compute_hsv_hist(region_at_match, mask=rotated_mask)
            hist_piece = _compute_hsv_hist(rotated_piece, mask=rotated_mask)
        else:
            hist_target = _compute_hsv_hist(region_at_match)
            hist_piece = _compute_hsv_hist(rotated_piece)
        hist_score = max(0, _hist_correlation(hist_target, hist_piece))

        ncc_score = max(0, max_val)
        combined = 0.6 * ncc_score + 0.4 * hist_score

        return {
            'score': combined,
            'ncc_score': ncc_score,
            'hist_score': hist_score,
            'raw_offset': (dx, dy),
            'ncc_map': ncc_map,
            'search_origin': (sx1, sy1),
            'piece_size': (pw_img, ph_img),
        }

    def _compute_ncc(self, img_a, img_b):
        if img_a.shape != img_b.shape:
            min_h = min(img_a.shape[0], img_b.shape[0])
            min_w = min(img_a.shape[1], img_b.shape[1])
            img_a = img_a[:min_h, :min_w]
            img_b = img_b[:min_h, :min_w]
        ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ga = (ga - ga.mean()) / (ga.std() + 1e-8)
        gb = (gb - gb.mean()) / (gb.std() + 1e-8)
        ncc = np.mean(ga * gb)
        return float(max(0, ncc))

    def _compute_masked_ncc(self, img_a, img_b, mask):
        if img_a.shape[:2] != img_b.shape[:2]:
            min_h = min(img_a.shape[0], img_b.shape[0])
            min_w = min(img_a.shape[1], img_b.shape[1])
            img_a = img_a[:min_h, :min_w]
            img_b = img_b[:min_h, :min_w]
            mask = mask[:min_h, :min_w]
        ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = (mask > 127).astype(np.float32) / 255.0
        if m.sum() < 10:
            return self._compute_ncc(img_a, img_b)
        ma = ga * m
        mb = gb * m
        mean_a = ma.sum() / m.sum()
        mean_b = mb.sum() / m.sum()
        da = (ga - mean_a) * m
        db = (gb - mean_b) * m
        std_a = np.sqrt((da ** 2).sum() / m.sum() + 1e-8)
        std_b = np.sqrt((db ** 2).sum() / m.sum() + 1e-8)
        ncc = (da * db).sum() / (m.sum() * std_a * std_b + 1e-8)
        return float(max(0, ncc))

    def match_all_pieces(self):
        print("\n--- Phase 3: Per-Piece Matching ---")
        results = {}
        pids = sorted(self.placed.keys())
        total = len(pids)

        for idx, pid in enumerate(pids):
            info = self.placed[pid]
            result = self._match_single_piece(pid, info)
            if result is not None:
                results[pid] = result
            if (idx + 1) % 10 == 0 or idx == total - 1:
                print(f"  Matched {idx + 1}/{total} pieces")

        self.match_results = results

        scores = [r['score'] for r in results.values()]
        if scores:
            print(f"  Score stats: min={min(scores):.3f}, max={max(scores):.3f}, "
                  f"mean={np.mean(scores):.3f}, median={np.median(scores):.3f}")

        return results

    def _subpixel_refine(self, ncc_map):
        if ncc_map is None or ncc_map.size < 9:
            return (0.0, 0.0), 0.0

        _, _, _, max_loc = cv2.minMaxLoc(ncc_map)
        mx, my = max_loc

        if mx <= 0 or mx >= ncc_map.shape[1] - 1 or my <= 0 or my >= ncc_map.shape[0] - 1:
            return (float(mx), float(my)), float(ncc_map[my, mx])

        dx = (ncc_map[my, mx - 1] - ncc_map[my, mx + 1]) / \
             (2.0 * (ncc_map[my, mx - 1] - 2 * ncc_map[my, mx] + ncc_map[my, mx + 1]) + 1e-10)
        dy = (ncc_map[my - 1, mx] - ncc_map[my + 1, mx]) / \
             (2.0 * (ncc_map[my - 1, mx] - 2 * ncc_map[my, mx] + ncc_map[my + 1, mx]) + 1e-10)

        dx = np.clip(dx, -0.5, 0.5)
        dy = np.clip(dy, -0.5, 0.5)

        refined_x = mx + dx
        refined_y = my + dy

        peak_val = float(ncc_map[my, mx])
        return (refined_x, refined_y), peak_val

    def _consistency_check(self, offsets):
        grid = {}
        for pid, info in self.placed.items():
            if pid in offsets:
                grid[(info['gx'], info['gy'])] = offsets[pid]

        consistent = {}
        for pid, info in self.placed.items():
            if pid not in offsets:
                continue
            gx, gy = info['gx'], info['gy']
            dx, dy = offsets[pid]

            neighbors = []
            for ngx, ngy in [(gx - 1, gy), (gx + 1, gy), (gx, gy - 1), (gx, gy + 1)]:
                if (ngx, ngy) in grid:
                    neighbors.append(grid[(ngx, ngy)])

            if not neighbors:
                consistent[pid] = 1.0
                continue

            avg_nx = np.mean([n[0] for n in neighbors])
            avg_ny = np.mean([n[1] for n in neighbors])
            diff = math.sqrt((dx - avg_nx) ** 2 + (dy - avg_ny) ** 2)
            max_diff = max(
                math.sqrt((n[0] - avg_nx) ** 2 + (n[1] - avg_ny) ** 2)
                for n in neighbors
            ) if neighbors else 0

            if max_diff < 1e-6:
                consistency = 1.0
            else:
                consistency = max(0, 1.0 - diff / (max_diff + 1e-6))
            consistent[pid] = consistency

        return consistent

    def _fit_global_transform(self, offsets, threshold=0.7):
        src_pts = []
        dst_pts = []
        for pid, info in self.placed.items():
            if pid not in offsets:
                continue
            result = self.match_results.get(pid)
            if result is None or result['score'] < threshold:
                continue
            gx, gy = info['gx'], info['gy']
            dx, dy = offsets[pid]
            src_pts.append([gx, gy])
            dst_pts.append([gx + dx, gy + dy])

        if len(src_pts) < 3:
            return None

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        M, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,
                                           ransacReprojThreshold=2.0)
        return M

    def refine_positions(self, threshold=0.7):
        print(f"\n--- Phase 4: Position Refinement (threshold={threshold}) ---")

        refined_offsets = {}
        for pid, result in self.match_results.items():
            if result.get('ncc_map') is not None:
                (rx, ry), peak = self._subpixel_refine(result['ncc_map'])
                so = result['search_origin']
                ps = result.get('piece_size', (0, 0))

                img = self.target_rectified
                th, tw = img.shape[:2]
                cell_w = tw / self.pw
                cell_h = th / self.ph
                info = self.placed[pid]
                piece_center_x = info['gx'] * cell_w + cell_w / 2
                piece_center_y = info['gy'] * cell_h + cell_h / 2

                match_center_x = so[0] + rx + ps[0] / 2
                match_center_y = so[1] + ry + ps[1] / 2
                dx = match_center_x - piece_center_x
                dy = match_center_y - piece_center_y
                refined_offsets[pid] = (dx, dy)
            else:
                refined_offsets[pid] = result.get('raw_offset', (0.0, 0.0))

        consistencies = self._consistency_check(refined_offsets)

        self.global_transform = self._fit_global_transform(refined_offsets, threshold)
        if self.global_transform is not None:
            print(f"  Global affine transform fitted: {self.global_transform.tolist()}")

        refined_count = 0
        for pid, result in self.match_results.items():
            score = result['score']
            refined = score >= threshold
            confidence = score * consistencies.get(pid, 0.5)
            result['refined_offset'] = refined_offsets.get(pid, (0.0, 0.0))
            result['confidence'] = confidence
            result['refined'] = refined
            if refined:
                refined_count += 1

        print(f"  Refined {refined_count}/{len(self.match_results)} pieces")

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
                'raw_offset': [round(float(result.get('raw_offset', (0, 0))[0]), 2),
                               round(float(result.get('raw_offset', (0, 0))[1]), 2)],
                'refined_offset': [round(float(result.get('refined_offset', (0, 0))[0]), 2),
                                   round(float(result.get('refined_offset', (0, 0))[1]), 2)],
                'confidence': round(float(result.get('confidence', 0)), 4),
                'refined': bool(result.get('refined', False)),
            }

        report_path = os.path.join(out, 'target_match_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved: {report_path}")

        if self.global_transform is not None:
            transform_data = {
                'matrix': self.global_transform.tolist(),
                'type': 'affine',
                'rotation': self.best_rotation,
                'flip': self.best_flip,
            }
            transform_path = os.path.join(out, 'target_refined_transform.json')
            with open(transform_path, 'w') as f:
                json.dump(transform_data, f, indent=2)
            print(f"  Transform saved: {transform_path}")

        return report

    def generate_visual(self, output_dir=None):
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)

        img = self.target_rectified.copy()
        th, tw = img.shape[:2]
        cell_w = tw / self.pw
        cell_h = th / self.ph

        for pid, result in self.match_results.items():
            info = self.placed[pid]
            gx, gy = info['gx'], info['gy']
            score = result.get('score', 0)

            x1 = int(gx * cell_w)
            y1 = int(gy * cell_h)
            x2 = int((gx + 1) * cell_w)
            y2 = int((gy + 1) * cell_h)

            if score >= 0.7:
                color = (0, int(255 * min(1, score)), 0)
            elif score >= 0.4:
                color = (0, 200, 200)
            else:
                color = (0, 0, int(255 * min(1, 1 - score)))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if result.get('refined', False):
                ro = result.get('refined_offset', (0, 0))
                arrow_scale = 3.0
                ex = int(cx + ro[0] * arrow_scale)
                ey = int(cy + ro[1] * arrow_scale)
                cv2.arrowedLine(img, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)

            label = f"{pid}:{score:.2f}"
            font_scale = max(0.3, min(0.8, cell_w / 200))
            cv2.putText(img, label, (x1 + 2, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        vis_path = os.path.join(out, 'target_match_visual.png')
        cv2.imwrite(vis_path, img)
        print(f"  Visual saved: {vis_path}")
        return vis_path

    def run(self, refine_threshold=0.7):
        self.rectify_target()
        self._align_orientation()
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
            'raw_offset': r.get('raw_offset', (0, 0)),
            'refined_offset': r.get('refined_offset', (0, 0)),
            'confidence': r.get('confidence', 0),
            'refined': r.get('refined', False),
        }
    result['global_transform'] = matcher.global_transform.tolist() if matcher.global_transform is not None else None
    result['orientation'] = matcher.best_rotation
    result['flip'] = matcher.best_flip
    return result


def main():
    parser = argparse.ArgumentParser(description='Match puzzle solution against target image')
    parser.add_argument('--target', required=True, help='Path to target image')
    parser.add_argument('--solution', required=True, help='Path to solution directory (containing solution_grid.txt)')
    parser.add_argument('--output', default=None, help='Output directory for results')
    parser.add_argument('--output-root', default=None, help='Root output directory (e.g., output/puzzle_new) containing 2_piece_colors/')
    parser.add_argument('--refine-threshold', type=float, default=0.7, help='Score threshold for position refinement')
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
