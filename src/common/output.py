"""
Human-readable output generation for puzzle solutions.

Provides:
  - Solution grid text output
  - Annotated target image (with piece IDs on grid)
  - HTML piece catalog
"""

import os
import math

import numpy as np
import cv2


def generate_solution_grid(board, output_dir):
    """
    Generate solution_grid.txt from a solved Board object.
    Shows piece IDs with orientation arrows.
    """
    output_path = os.path.join(output_dir, 'solution_grid.txt')
    num_digits = math.floor(math.log(max(board.width * board.height, 1), 10)) + 3
    arrow_chars = '^>v<'

    lines = []
    separator = '  ' + '-' * (num_digits * board.width)
    lines.append(separator)

    for y in range(board.height):
        row_str = '  '
        for x in range(board.width):
            cell = board.get(x, y)
            if cell is None:
                row_str += ' ' * num_digits + '-'
            else:
                piece_id, _, orientation = cell
                ori_str = arrow_chars[orientation]
                row_str += '{:>{}}{}'.format(piece_id, num_digits, ori_str)
        lines.append(row_str)
        lines.append('')

    lines.append(separator)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Solution grid written to {output_path}")
    return output_path


def generate_annotated_target(target_image, board, output_dir):
    """
    Generate annotated_target.png with piece IDs overlaid on the target image.
    target_image: numpy BGR image or path to image file
    board: solved Board object
    """
    if isinstance(target_image, str):
        img = cv2.imread(target_image)
        if img is None:
            print(f"Warning: cannot load target image {target_image}")
            return None
    else:
        img = target_image.copy()

    h, w = img.shape[:2]
    cell_w = w / board.width
    cell_h = h / board.height

    for row in range(board.height):
        for col in range(board.width):
            x1 = int(col * cell_w)
            y1 = int(row * cell_h)
            x2 = int((col + 1) * cell_w)
            y2 = int((row + 1) * cell_h)

            cell = board.get(col, row)
            if cell is not None:
                piece_id, _, _ = cell
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                text = f"#{piece_id}"
                font_scale = max(0.3, min(cell_w, cell_h) / 150.0)
                cv2.putText(img, text, (x1 + 3, y1 + int(cell_h * 0.3)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 0, 0), 1)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 1)

    output_path = os.path.join(output_dir, 'annotated_target.png')
    cv2.imwrite(output_path, img)
    print(f"Annotated target written to {output_path}")
    return output_path


def generate_piece_catalog_html(pieces_info, output_dir):
    """
    Generate piece_catalog.html showing all pieces with IDs.
    pieces_info: list of dicts with keys:
        'id', 'thumbnail_path'(optional), 'is_corner', 'is_edge',
        'solved_position'(optional, (x,y) tuple)
    """
    corners = [p for p in pieces_info if p.get('is_corner')]
    edges = [p for p in pieces_info if p.get('is_edge') and not p.get('is_corner')]
    inner = [p for p in pieces_info if not p.get('is_edge') and not p.get('is_corner')]

    html = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body { font-family: Arial, sans-serif; margin: 20px; }
h2 { color: #333; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, 120px); gap: 10px; }
.piece { border: 1px solid #ccc; padding: 5px; text-align: center; }
.piece img { width: 100px; height: 100px; object-fit: contain; }
.corner { border-color: red; }
.edge { border-color: orange; }
.inner { border-color: green; }
.solved { background: #e8f5e9; }
.unsolved { background: #fff3e0; }
</style></head><body>"""

    html += "<h2>Corners</h2><div class='grid'>"
    for p in corners:
        html += _piece_card(p)
    html += "</div>"

    html += "<h2>Edges</h2><div class='grid'>"
    for p in edges:
        html += _piece_card(p)
    html += "</div>"

    html += "<h2>Inner</h2><div class='grid'>"
    for p in inner:
        html += _piece_card(p)
    html += "</div>"

    html += "</body></html>"

    output_path = os.path.join(output_dir, 'piece_catalog.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Piece catalog written to {output_path}")
    return output_path


def _piece_card(piece):
    pid = piece.get('id', '?')
    css_classes = ['piece']
    if piece.get('is_corner'):
        css_classes.append('corner')
    elif piece.get('is_edge'):
        css_classes.append('edge')
    else:
        css_classes.append('inner')
    if piece.get('solved_position'):
        css_classes.append('solved')
    else:
        css_classes.append('unsolved')

    img_tag = ''
    thumb = piece.get('thumbnail_path')
    if thumb:
        img_tag = f'<img src="{thumb}" alt="Piece {pid}">'

    pos_text = ''
    pos = piece.get('solved_position')
    if pos:
        pos_text = f'<br>Pos: ({pos[0]}, {pos[1]})'

    return (f'<div class="{" ".join(css_classes)}">'
            f'{img_tag}'
            f'<div>#{pid}{pos_text}</div>'
            f'</div>')


def print_solution_summary(board):
    """Print a summary of the solution to stdout."""
    print(f"\nSolution found!")
    print(f"  Board: {board.width} x {board.height}")
    print(f"  Pieces placed: {board.placed_count}")
    print(board)


def generate_assembly_guide(board, output_dir, connectivity=None):
    """
    Generate a step-by-step assembly guide for building the puzzle.

    The guide organizes pieces by placement order:
      1. Corners first (4 pieces)
      2. Edges by side (top, right, bottom, left)
      3. Inner pieces row by row

    For each piece, shows:
      - Step number
      - Piece ID
      - Target position (row, col)
      - Required rotation
      - Which already-placed pieces it connects to (hints)

    Args:
        board: solved Board object
        output_dir: directory to write the guide
        connectivity: optional connectivity graph for neighbor hints

    Returns:
        path to the generated guide file
    """
    if board.placed_count == 0:
        print("No pieces placed, cannot generate assembly guide")
        return None

    steps = []
    step_num = 0

    placed_pieces = []
    for y in range(board.height):
        for x in range(board.width):
            cell = board.get(x, y)
            if cell is not None:
                placed_pieces.append((x, y, cell))

    def _rotation_desc(orientation):
        descs = {
            0: "No rotation",
            1: "Rotate 90deg clockwise",
            2: "Rotate 180deg (upside down)",
            3: "Rotate 90deg counter-clockwise",
        }
        return descs.get(orientation, f"Rotate {orientation * 90}deg")

    def _get_neighbor_info(x, y):
        """Get info about already-placed neighbors."""
        neighbors = []
        for dx, dy, side_name in [
            (-1, 0, 'left'), (1, 0, 'right'),
            (0, -1, 'top'), (0, 1, 'bottom')
        ]:
            nx, ny = x + dx, y + dy
            ncell = board.get(nx, ny)
            if ncell is not None:
                neighbors.append(
                    f"  - {side_name.capitalize()}: Piece #{ncell[0]} "
                    f"(already at row {ny+1}, col {nx+1})"
                )
        return neighbors

    def _classify(x, y):
        is_corner = (x == 0 or x == board.width - 1) and \
                    (y == 0 or y == board.height - 1)
        is_edge = (x == 0 or x == board.width - 1 or
                   y == 0 or y == board.height - 1) and not is_corner
        if is_corner:
            return 'corner'
        elif is_edge:
            return 'edge'
        else:
            return 'inner'

    ordered = sorted(placed_pieces, key=lambda p: (p[1], p[0]))

    groups = {'corner': [], 'edge': [], 'inner': []}
    for x, y, cell in ordered:
        cls = _classify(x, y)
        groups[cls].append((x, y, cell))

    for cls_name in ['corner', 'edge', 'inner']:
        for x, y, cell in groups[cls_name]:
            piece_id, _, orientation = cell
            step_num += 1

            neighbors = _get_neighbor_info(x, y)
            neighbor_text = '\n'.join(neighbors) if neighbors else '  - None (first placement)'

            steps.append({
                'step': step_num,
                'piece_id': piece_id,
                'row': y + 1,
                'col': x + 1,
                'type': cls_name,
                'rotation': _rotation_desc(orientation),
                'neighbors': neighbor_text,
            })

    lines = []
    lines.append("=" * 60)
    lines.append("PUZZLE ASSEMBLY GUIDE")
    lines.append(f"Board: {board.width} x {board.height} = "
                 f"{board.width * board.height} positions")
    lines.append(f"Pieces placed: {board.placed_count}")
    lines.append(f"Missing: {board.width * board.height - board.placed_count}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Legend:")
    lines.append("  Type: corner=Corner piece, edge=Edge piece, inner=Inner piece")
    lines.append("  Rotation: how to orient the piece before placing")
    lines.append("  Neighbors: already-placed pieces this piece connects to")
    lines.append("")
    lines.append("-" * 60)

    current_type = None
    for step in steps:
        if step['type'] != current_type:
            current_type = step['type']
            type_label = current_type.upper()
            if current_type == 'corner':
                type_label += " PIECES (start here)"
            elif current_type == 'edge':
                type_label += " PIECES (border)"
            else:
                type_label += " PIECES (fill in)"
            lines.append("")
            lines.append(f"--- {type_label} ---")

        lines.append("")
        lines.append(f"Step {step['step']:>3d}: Place Piece #{step['piece_id']}")
        lines.append(f"  Position: Row {step['row']}, Column {step['col']}")
        lines.append(f"  Type: {step['type']}")
        lines.append(f"  Rotation: {step['rotation']}")
        lines.append(f"  Connects to:")
        lines.append(step['neighbors'])

    lines.append("")
    lines.append("-" * 60)
    lines.append("ASSEMBLY COMPLETE!")
    lines.append("=" * 60)

    output_path = os.path.join(output_dir, 'assembly_guide.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Assembly guide written to {output_path}")
    return output_path


def generate_assembly_guide_html(board, output_dir, pieces_db=None):
    """
    Generate a visual HTML assembly guide with piece thumbnails and
    step-by-step placement instructions.

    Args:
        board: solved Board object
        output_dir: directory to write the guide
        pieces_db: optional PieceDatabase for thumbnail access

    Returns:
        path to the generated HTML guide file
    """
    if board.placed_count == 0:
        return None

    placed_pieces = []
    for y in range(board.height):
        for x in range(board.width):
            cell = board.get(x, y)
            if cell is not None:
                placed_pieces.append((x, y, cell))

    placed_pieces.sort(key=lambda p: (p[1], p[0]))

    def _classify(x, y):
        is_corner = (x == 0 or x == board.width - 1) and \
                    (y == 0 or y == board.height - 1)
        is_edge = (x == 0 or x == board.width - 1 or
                   y == 0 or y == board.height - 1) and not is_corner
        if is_corner:
            return 'corner'
        elif is_edge:
            return 'edge'
        return 'inner'

    def _rotation_emoji(orientation):
        return ['⬆️', '➡️', '⬇️', '⬅️'][orientation]

    steps_html = []
    step_num = 0
    current_type = None

    for x, y, cell in placed_pieces:
        piece_id, _, orientation = cell
        step_num += 1
        cls = _classify(x, y)

        if cls != current_type:
            current_type = cls
            label = cls.upper()
            if cls == 'corner':
                label += ' - Start Here'
            elif cls == 'edge':
                label += ' - Border'
            else:
                label += ' - Fill In'
            steps_html.append(
                f'<div class="section-header">{label}</div>'
            )

        thumb_src = ''
        if pieces_db and piece_id in pieces_db.pieces:
            pd = pieces_db.pieces[piece_id]
            if pd.color_image is not None:
                import base64
                _, buf = cv2.imencode('.png', pd.color_image)
                thumb_src = f'data:image/png;base64,{base64.b64encode(buf).decode()}'

        neighbors_html = ''
        for dx, dy, name in [(-1, 0, 'left'), (1, 0, 'right'),
                              (0, -1, 'top'), (0, 1, 'bottom')]:
            ncell = board.get(x + dx, y + dy)
            if ncell is not None:
                neighbors_html += (
                    f'<span class="neighbor">{name}: '
                    f'#{ncell[0]}</span> '
                )

        bg_class = 'step-corner' if cls == 'corner' else \
                   'step-edge' if cls == 'edge' else 'step-inner'

        steps_html.append(f'''
<div class="step {bg_class}">
  <div class="step-number">Step {step_num}</div>
  <div class="step-piece">
    {"<img class='piece-thumb' src='" + thumb_src + "'>" if thumb_src else ""}
    <div class="piece-info">
      <span class="piece-id">#{piece_id}</span>
      <span class="piece-type">{cls}</span>
    </div>
  </div>
  <div class="step-details">
    <div>Position: Row {y+1}, Col {x+1}</div>
    <div>Rotation: {_rotation_emoji(orientation)}</div>
    <div class="neighbors">{neighbors_html or "First piece"}</div>
  </div>
</div>''')

    total = board.width * board.height
    progress_pct = board.placed_count / total * 100 if total > 0 else 0

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
.progress {{ background: #e0e0e0; border-radius: 10px; padding: 10px; margin: 10px 0; }}
.progress-bar {{ background: #4caf50; height: 20px; border-radius: 10px; }}
.section-header {{ font-size: 1.3em; font-weight: bold; color: #1565c0;
                   margin: 20px 0 10px; padding: 5px 10px; background: #e3f2fd; border-radius: 5px; }}
.step {{ display: flex; align-items: center; background: white; margin: 5px 0;
          padding: 10px; border-radius: 8px; border-left: 4px solid #ccc; }}
.step-corner {{ border-left-color: #f44336; }}
.step-edge {{ border-left-color: #ff9800; }}
.step-inner {{ border-left-color: #4caf50; }}
.step-number {{ font-size: 1.2em; font-weight: bold; color: #666; min-width: 80px; }}
.step-piece {{ display: flex; align-items: center; min-width: 130px; }}
.piece-thumb {{ width: 60px; height: 60px; object-fit: contain; margin-right: 8px; border-radius: 4px; }}
.piece-id {{ font-weight: bold; font-size: 1.1em; }}
.piece-type {{ color: #888; font-size: 0.85em; margin-left: 5px; }}
.step-details {{ margin-left: 20px; color: #555; }}
.neighbors {{ margin-top: 4px; }}
.neighbor {{ display: inline-block; background: #e8f5e9; padding: 2px 8px;
             border-radius: 10px; margin-right: 5px; font-size: 0.85em; }}
</style></head><body>
<h1>Puzzle Assembly Guide</h1>
<div class="progress">
  <div style="font-weight:bold;">Progress: {board.placed_count}/{total} ({progress_pct:.0f}%)</div>
  <div class="progress-bar" style="width: {progress_pct}%;"></div>
</div>
{''.join(steps_html)}
</body></html>"""

    output_path = os.path.join(output_dir, 'assembly_guide.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML assembly guide written to {output_path}")
    return output_path
