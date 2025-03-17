from typing import Tuple

import numpy as np

import numpy as np
import math
from typing import Tuple

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    matrix = np.zeros(shape=shape, dtype="float32")
    
    # Parameters (tweak these to adjust behavior)
    bumper_width = 75          # width of the top bumper regions
    bumper_height = 150        # height of the top bumper regions
    mountain_peak_row = 250    # row at which the sloped portion starts
    mid_col = w // 2           # center column
    full_row = int(0.8 * h)    # row at which we switch to the bottom rectangle

    # --- Bumper Regions (upper corners) ---
    # Top-left bumper: assign -1 so detections here reduce left motor speed
    matrix[:bumper_height, :bumper_width] = -1
    # Top-right bumper: assign 1 so detections here have a lesser effect (or boost)
    matrix[:bumper_height, w - bumper_width:] = 1

    # --- Bottom Rectangle ---
    # For rows below 'full_row', assign fixed values: left half gets 1, right half gets -1
    matrix[full_row:, :mid_col] = 1
    matrix[full_row:, mid_col+1:] = -1

    # --- Sloped Portion ---
    # For rows between mountain_peak_row and full_row, compute a gradient based on the row
    row_indices = np.arange(mountain_peak_row, full_row)
    # Normalize row progression: 0 at mountain_peak_row, 1 at full_row
    progression = (row_indices - mountain_peak_row) / (full_row - mountain_peak_row)
    # Compute the width (number of columns) affected from the center
    # As progression increases, more columns on each side become active
    left_widths = np.floor(mid_col * progression).astype(int)
    
    # Fill in the sloped area row by row (vectorized width computation; row-specific indices still require iteration)
    for idx, row in enumerate(row_indices):
        # For left side: from (mid - width) to mid, assign +1
        left_start = mid_col - left_widths[idx]
        matrix[row, left_start:mid_col] = 1
        
        # For right side: from mid+1 to mid+width, assign -1
        right_end = min(mid_col + 1 + left_widths[idx], w)
        matrix[row, mid_col+1:right_end] = -1

    return matrix

def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # The right matrix is the horizontal mirror image of the left matrix
    return np.fliplr(get_motor_left_matrix(shape))
