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
