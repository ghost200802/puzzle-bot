import os
import sys
import json
import math
import argparse
import re

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
                meta = json.load(f)
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

        margin = max(data_w, data_h) * 0.05
        header_h = 60

        canvas_w = data_w + 2 * margin
        canvas_h = data_h + 2 * margin + header_h
        max_size = 12000
        if max(canvas_w, canvas_h) > max_size:
            gen_scale = max_size / max(canvas_w, canvas_h)
        else:
            gen_scale = 1.0

        grid_x = margin * gen_scale
        grid_y = margin * gen_scale + header_h
        grid_w = data_w * gen_scale
        grid_h = data_h * gen_scale

        rs = self._resize_scale
        self._grid_offset_x = grid_x * rs
        self._grid_offset_y = grid_y * rs
        self._cell_w = grid_w * rs / self.pw
        self._cell_h = grid_h * rs / self.ph

        print(f"  Grid layout (analytical): offset=({self._grid_offset_x:.1f},{self._grid_offset_y:.1f}), "
              f"cell={self._cell_w:.1f}x{self._cell_h:.1f}")

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

    def _match_single_piece(self, pid, info):
        gx, gy = info['gx'], info['gy']

        x1 = int(self._grid_offset_x + gx * self._cell_w)
        y1 = int(self._grid_offset_y + gy * self._cell_h)
        x2 = int(self._grid_offset_x + (gx + 1) * self._cell_w)
        y2 = int(self._grid_offset_y + (gy + 1) * self._cell_h)

        ah, aw = self._assembly_img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(aw, x2), min(ah, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        target_cell = self.target_aligned[y1:y2, x1:x2]
        assembly_cell = self._assembly_img[y1:y2, x1:x2]

        if target_cell.size == 0 or assembly_cell.size == 0:
            return None

        min_h = min(target_cell.shape[0], assembly_cell.shape[0])
        min_w = min(target_cell.shape[1], assembly_cell.shape[1])
        tc = target_cell[:min_h, :min_w]
        ac = assembly_cell[:min_h, :min_w]

        ac_gray = cv2.cvtColor(ac, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if ac_gray.std() < 5:
            return {'score': 0.0, 'ncc_score': 0.0, 'hist_score': 0.0}

        ncc = _ncc_score(tc, ac)
        ncc_score = max(0, ncc)

        tc_hsv = cv2.cvtColor(tc, cv2.COLOR_BGR2HSV)
        ac_hsv = cv2.cvtColor(ac, cv2.COLOR_BGR2HSV)
        mask = np.ones(tc_hsv.shape[:2], dtype=np.uint8) * 255
        hist_tc = cv2.calcHist([tc_hsv], [0, 1], mask, [36, 32], [0, 180, 0, 256])
        hist_ac = cv2.calcHist([ac_hsv], [0, 1], mask, [36, 32], [0, 180, 0, 256])
        cv2.normalize(hist_tc, hist_tc)
        cv2.normalize(hist_ac, hist_ac)
        hist_score = max(0, cv2.compareHist(
            hist_tc.astype(np.float32), hist_ac.astype(np.float32), cv2.HISTCMP_CORREL
        ))

        combined = 0.6 * ncc_score + 0.4 * hist_score
        return {
            'score': combined,
            'ncc_score': ncc_score,
            'hist_score': float(hist_score),
        }

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

        assembly_path = os.path.join(self.output_dir, '_match_assembly.png')
        if not os.path.exists(assembly_path) or self._assembly_img is None:
            print("  No assembly image, skipping visual")
            return

        assembly_bgr = self._assembly_img
        ah, aw = assembly_bgr.shape[:2]

        gray = cv2.cvtColor(assembly_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        mask = mask.astype(np.float32) / 255.0
        mask_3 = np.stack([mask] * 3, axis=2)

        target_bgr = self.target_aligned
        th, tw = target_bgr.shape[:2]

        if (th, tw) != (ah, aw):
            target_bgr = cv2.resize(target_bgr, (aw, ah))

        blended = target_bgr.astype(np.float32) * (1 - mask_3 * 0.6) + assembly_bgr.astype(np.float32) * (mask_3 * 0.6)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        overlay_pil = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGBA))
        draw = ImageDraw.Draw(overlay_pil)

        try:
            font = ImageFont.truetype("arialbd.ttf", max(10, min(20, int(self._cell_w / 8))))
        except Exception:
            font = ImageFont.load_default()

        for pid, result in self.match_results.items():
            info = self.placed[pid]
            gx, gy = info['gx'], info['gy']
            score = result.get('score', 0)

            x1 = int(self._grid_offset_x + gx * self._cell_w)
            y1 = int(self._grid_offset_y + gy * self._cell_h)
            x2 = int(self._grid_offset_x + (gx + 1) * self._cell_w)
            y2 = int(self._grid_offset_y + (gy + 1) * self._cell_h)

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
