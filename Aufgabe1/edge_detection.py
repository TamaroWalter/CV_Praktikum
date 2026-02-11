# File with functions for edge detection.
import numpy as np

def sobel(matrix):
  sobel_y = np.array([
    [1, 2, 1,],
    [0, 0, 0,],
    [-1, -2, -1,],
  ])

  sobel_x = np.array([
    [-1, 0, 1,],
    [-2, 0, 2,],
    [-1, 0, 1,],
  ])