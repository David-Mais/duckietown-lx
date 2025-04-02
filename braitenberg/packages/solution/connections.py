import math

import numpy as np
from typing import Tuple

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    mid = w / 2.0

    # Create row and column indices.
    rows = np.arange(h).reshape(h, 1)   # shape: (h, 1)
    cols = np.arange(w).reshape(1, w)     # shape: (1, w)

    # Vertical weight: increases from 0 at the top to 1 at the bottom.
    vertical_weight = rows / float(h)
    
    # Horizontal directional factor: positive on left, negative on right.
    # At col=0, (mid - 0)/mid = 1; at col=mid, it is 0; at col=w-1, it is negative.
    directional = (cols - mid) / mid
    
    # Gaussian mask centered at the middle to concentrate weights in the center.
    # Adjust sigma to control the width of the central region.
    sigma = w * 0.15  
    center_mask = np.exp(-((cols - mid) ** 2) / (2 * sigma ** 2))
    
    # The final left motor matrix.
    left_matrix = vertical_weight * directional * center_mask
    return left_matrix.astype(np.float32)

def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    return np.fliplr(get_motor_left_matrix(shape)).astype(np.float32)