import sys
sys.path.insert(0, 'src')
from common.vector import Vector

for pid in range(1, 11):
    img_path = f'output/puzzle_run/3_vector/piece_{pid}.bmp'
    try:
        p = Vector.from_file(img_path, id=pid)
        p.find_border_raster()
        p.vectorize()
        p.find_four_corners()
        corners = p.corners

        svg_path = f'output/puzzle_run/3_vector/corners_piece_{pid}.svg'
        with open(svg_path, 'w') as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000">\n')
            f.write('<rect width="1000" height="1000" fill="#1a1a2e"/>\n')
            pts = " ".join([f"{v[0]},{v[1]}" for v in p.vertices])
            f.write(f'<polygon points="{pts}" fill="#2d2d44" stroke="#888" stroke-width="1"/>\n')
            colors = ['#ff3333', '#33ff33', '#3399ff', '#ffff33']
            labels = ['C0', 'C1', 'C2', 'C3']
            for ci, (cx, cy) in enumerate(corners):
                cx, cy = int(cx), int(cy)
                f.write(f'<circle cx="{cx}" cy="{cy}" r="8" fill="{colors[ci]}" stroke="white" stroke-width="2"/>\n')
                f.write(f'<text x="{cx+12}" y="{cy-8}" fill="{colors[ci]}" font-size="16" font-family="monospace">{labels[ci]}({cx},{cy})</text>\n')
            f.write(f'<text x="10" y="25" fill="white" font-size="18" font-family="monospace">Piece {pid}</text>\n')
            f.write('</svg>\n')

        cx_str = ", ".join([f"({int(c[0])},{int(c[1])})" for c in corners])
        print(f"Piece_{pid:2d}: {cx_str}")
    except Exception as e:
        print(f"Piece_{pid:2d}: ERROR - {e}")

print(f"\nSVG files saved to: output/puzzle_run/3_vector/corners_piece_1~10.svg")
