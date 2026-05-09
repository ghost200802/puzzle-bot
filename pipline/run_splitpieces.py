#!/usr/bin/env python3
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'check'))

from check_segmentation import main

if __name__ == '__main__':
    main()
