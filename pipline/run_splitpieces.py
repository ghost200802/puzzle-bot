#!/usr/bin/env python3
"""
Split puzzle pieces from RGBA/RGB images.

Delegates to src/check/check_segmentation.py which performs:
  - Alpha-channel detection of individual pieces
  - Oversized piece splitting
  - Scale up + Gaussian blur + Otsu threshold (quality enhancement)
  - Morphological cleanup + binary_fill_holes
  - Size filtering

Output structure (under output/puzzle_new/):
  2_piece_bmps/    - binary masks (BMP, for vectorization)
  2_piece_colors/  - color images (RGBA PNG)
  inspect/         - grid overview per source image
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'check'))

from check_segmentation import main

if __name__ == '__main__':
    main()
