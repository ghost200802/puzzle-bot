import os, sys, numpy as np, cv2
from PIL import Image
from scipy.ndimage import label as ndlabel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from common.config import PIECE_BMP_DIR, PHONE_TARGET_PIECE_SIZE
from common.find_islands import save_island_as_bmp

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input', 'puzzles')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'puzzle_new')
BMP_DIR_NAME = PIECE_BMP_DIR

SIZE_RATIO_LOW = 0.35
SIZE_RATIO_HIGH = 1.8
MIN_AREA = 2000


def _touches_border(ys, xs, img_h, img_w, margin=2):
    return (ys.min() < margin or ys.max() >= img_h - margin or
            xs.min() < margin or xs.max() >= img_w - margin)


def _keep_largest_component(binary):
    lbl, n = ndlabel(binary)
    if n <= 1:
        return binary
    best = max(range(1, n + 1), key=lambda j: np.sum(lbl == j))
    return (lbl == best).astype(np.uint8)


def _try_split_oversized(mask, typical_max_dim):
    for kernel_size in range(3, 15, 2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(mask, kernel, iterations=1)
        lbl, n = ndlabel(eroded)
        valid = []
        for i in range(1, n + 1):
            area = np.sum(lbl == i)
            if area < 1000:
                continue
            ys, xs = np.where(lbl == i)
            md = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            if md < typical_max_dim * 1.5:
                valid.append(i)
        if len(valid) >= 2:
            result = []
            for i in valid:
                sub_mask = (lbl == i).astype(np.uint8)
                dilated = cv2.dilate(sub_mask, kernel, iterations=1)
                dilated = (dilated * mask).astype(np.uint8)
                dilated = _keep_largest_component(dilated)
                ys2, xs2 = np.where(dilated == 1)
                if len(ys2) > 0:
                    result.append({
                        'mask': dilated,
                        'area': int(np.sum(dilated)),
                        'max_dim': max(xs2.max() - xs2.min() + 1, ys2.max() - ys2.min() + 1),
                    })
            if len(result) >= 2:
                return result
    return None


def segment_image(image_path, target_size=PHONE_TARGET_PIECE_SIZE):
    """
    Two-pass approach:
      Pass 1: alpha channel -> detect pieces, measure typical size
      Pass 2: crop alpha -> scale up -> Gaussian blur -> threshold -> morph clean
    Uses ALPHA channel (not grayscale) as the source for thresholding,
    since RGBA images have garbage RGB in transparent regions.
    """
    img = Image.open(image_path)
    rgba = np.array(img)

    if rgba.shape[2] == 4:
        alpha = rgba[:, :, 3]
    else:
        gray = np.array(img.convert('L'))
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    img_h, img_w = alpha.shape

    # --- Pass 1: detect pieces using alpha channel ---
    binary_detect = (alpha > 128).astype(np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_detect = cv2.morphologyEx(binary_detect, cv2.MORPH_CLOSE, kernel3)
    binary_detect = cv2.morphologyEx(binary_detect, cv2.MORPH_OPEN, kernel3)

    labeled, num = ndlabel(binary_detect)
    all_components = []
    for i in range(1, num + 1):
        area = np.sum(labeled == i)
        if area < MIN_AREA:
            continue
        ys, xs = np.where(labeled == i)
        if _touches_border(ys, xs, img_h, img_w):
            continue
        all_components.append({
            'label': i,
            'area': area,
            'bbox': (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
            'max_dim': max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1),
        })

    if not all_components:
        return []

    typical_max_dim = int(np.median([p['max_dim'] for p in all_components]))
    typical_area = int(np.median([p['area'] for p in all_components]))
    scale_factor = target_size / typical_max_dim

    print(f"    Components: {len(all_components)}, typical: {typical_max_dim}px/{typical_area}px, scale: {scale_factor:.2f}x")

    # Handle oversized (merged pieces)
    final_components = []
    for comp in all_components:
        h_c = comp['bbox'][3] - comp['bbox'][1]
        w_c = comp['bbox'][2] - comp['bbox'][0]
        ratio = comp['area'] / typical_area
        if ratio > SIZE_RATIO_HIGH or max(w_c, h_c) > typical_max_dim * 1.5:
            comp_mask = (labeled == comp['label']).astype(np.uint8)
            split = _try_split_oversized(comp_mask, typical_max_dim)
            if split:
                for s in split:
                    ys2, xs2 = np.where(s['mask'] == 1)
                    final_components.append({
                        'label': comp['label'],
                        'area': s['area'],
                        'bbox': (xs2.min(), ys2.min(), xs2.max() + 1, ys2.max() + 1),
                        'max_dim': s['max_dim'],
                        'mask': s['mask'],
                    })
                print(f"      Split oversized (area={comp['area']}) -> {len(split)} pieces")
                continue
        final_components.append(comp)

    # Filter too-small
    filtered = []
    for comp in final_components:
        ratio = comp['area'] / typical_area
        if ratio < SIZE_RATIO_LOW:
            print(f"      Skipping small piece (area={comp['area']}, ratio={ratio:.2f}x)")
            continue
        filtered.append(comp)

    print(f"    Final: {len(filtered)} pieces")

    # --- Pass 2: crop alpha, scale up, smooth threshold ---
    pieces = []
    pad = max(3, int(typical_max_dim * 0.05))

    for comp in filtered:
        x0, y0, x1, y1 = comp['bbox']
        py0 = max(0, y0 - pad)
        py1 = min(img_h, y1 + pad)
        px0 = max(0, x0 - pad)
        px1 = min(img_w, x1 + pad)

        # Crop from ALPHA channel (0-255, clean foreground/background)
        alpha_crop = alpha[py0:py1, px0:px1]
        ch, cw = alpha_crop.shape

        new_w = max(int(cw * scale_factor), 10)
        new_h = max(int(ch * scale_factor), 10)

        # Scale up with cubic interpolation
        scaled = cv2.resize(alpha_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Gaussian blur + Otsu threshold (same as run_pipeline.py)
        blur_k = max(3, int(scale_factor * 0.8))
        if blur_k % 2 == 0:
            blur_k += 1
        blurred = cv2.GaussianBlur(scaled, (blur_k, blur_k), 0)

        _, smooth_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        smooth_binary = (smooth_binary > 127).astype(np.uint8)

        # Morphological clean
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smooth_binary = cv2.morphologyEx(smooth_binary, cv2.MORPH_OPEN, kern)
        smooth_binary = cv2.morphologyEx(smooth_binary, cv2.MORPH_CLOSE, kern)

        # Keep only the largest connected component
        smooth_binary = _keep_largest_component(smooth_binary)

        # Fill internal holes
        from scipy.ndimage import binary_fill_holes
        smooth_binary = binary_fill_holes(smooth_binary).astype(np.uint8)

        pieces.append({
            'id': len(pieces) + 1,
            'binary': smooth_binary,
            'origin': (int(px0), int(py0)),
            'centroid': ((px0 + px1) / 2.0, (py0 + py1) / 2.0),
            'area': int(np.sum(smooth_binary)),
            'target_size': (new_w, new_h),
        })

    return pieces


def main():
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    bmp_dir = os.path.join(OUTPUT_DIR, BMP_DIR_NAME)
    os.makedirs(bmp_dir, exist_ok=True)

    all_pieces = []
    global_id = 1

    print("=" * 60)
    print("Segmentation Pipeline (BMP only)")
    print(f"Input: {INPUT_DIR}")
    print(f"BMP output: {bmp_dir}")
    print("=" * 60)

    for img_file in image_files:
        img_path = os.path.join(INPUT_DIR, img_file)
        print(f"\n  Processing: {img_file}")
        pieces = segment_image(img_path)
        for p in pieces:
            p['id'] = global_id
            p['source_file'] = img_file
            all_pieces.append(p)
            global_id += 1

    print(f"\n  Total pieces: {len(all_pieces)}")

    for p in all_pieces:
        pid = p['id']
        bmp_path = os.path.join(bmp_dir, f'piece_{pid}.bmp')
        save_island_as_bmp(p['binary'], bmp_path)

    # Analysis
    print(f"\n{'='*60}")
    print("BMP Analysis")
    print(f"{'='*60}")

    sizes = [(p['id'], p['target_size'][0], p['target_size'][1], p['area'], p['source_file'])
             for p in all_pieces]
    widths = [s[1] for s in sizes]
    heights = [s[2] for s in sizes]
    areas = [s[3] for s in sizes]
    med_w, med_h = np.median(widths), np.median(heights)
    med_area = np.median(areas)
    print(f"  Count: {len(sizes)}")
    print(f"  Median size: {med_w:.0f}x{med_h:.0f}, median area: {med_area:.0f}")
    print(f"  Size range: w=[{min(widths)}-{max(widths)}], h=[{min(heights)}-{max(heights)}]")

    issues = []
    for pid, w, h, area, src in sizes:
        ratio = (w * h) / (med_w * med_h)
        if ratio > 1.8:
            issues.append(f"TOO_LARGE: piece {pid} ({w}x{h}), {ratio:.1f}x median, from {src}")
        elif ratio < 0.35:
            issues.append(f"TOO_SMALL: piece {pid} ({w}x{h}), {ratio:.2f}x median, from {src}")

    if issues:
        print(f"\n  ** Issues ({len(issues)}): **")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"  No issues detected!")

    print(f"\n  Per-source:")
    for img_file in image_files:
        src_pieces = [s for s in sizes if s[4] == img_file]
        print(f"    {img_file}: {len(src_pieces)} pieces")

    # Save grid overview
    inspect_dir = os.path.join(OUTPUT_DIR, 'inspect')
    os.makedirs(inspect_dir, exist_ok=True)
    for img_file in image_files:
        src_pieces = [p for p in all_pieces if p['source_file'] == img_file]
        if not src_pieces:
            continue
        cell_size = 100
        cols = 10
        rows = (len(src_pieces) + cols - 1) // cols
        grid = np.ones((rows * cell_size, cols * cell_size), dtype=np.uint8) * 128
        for idx, p in enumerate(src_pieces):
            r, c = idx // cols, idx % cols
            vis = (p['binary'] * 255).astype(np.uint8)
            vis_resized = cv2.resize(vis, (cell_size - 4, cell_size - 4))
            y0 = r * cell_size + 2
            x0 = c * cell_size + 2
            grid[y0:y0 + cell_size - 4, x0:x0 + cell_size - 4] = vis_resized
        grid_path = os.path.join(inspect_dir, f'grid_{img_file.replace(".png", "")}.png')
        Image.fromarray(grid, mode='L').save(grid_path)
        print(f"  Saved grid overview: {grid_path}")

    print(f"\n  BMPs saved to {bmp_dir}/")


if __name__ == '__main__':
    main()
