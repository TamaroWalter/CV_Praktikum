"""
Edge detection module.
Provides functions for Canny edge detection using Sobel operators.

:author: Tamaro Walter, Simon Schoenhoeffer
:group: Praktikum Gruppe 14
"""
import numpy as np

def sobel(matrix: np.ndarray) -> np.ndarray:
  # Define sobel operator.
  sobel_y: np.ndarray = np.array([
    [1, 2, 1,],
    [0, 0, 0,],
    [-1, -2, -1,],
  ])
  sobel_x: np.ndarray = np.array([
    [-1, 0, 1,],
    [-2, 0, 2,],
    [-1, 0, 1,],
  ])

  rows, cols = matrix.shape
  edge_strengths: np.ndarray = np.zeros((rows-2, cols-2), dtype=np.uint8)

  for y in range(1, rows-1):
    for x in range(1, cols -1):
      # Build a submatrix of the current neighborhood.
      neighbourhood: np.ndarray = np.array([
        [matrix[y-1, x-1], matrix[y-1, x], matrix[y-1, x+1] ],
        [matrix[y, x-1], matrix[y, x], matrix[y, x+1] ],
        [matrix[y+1, x-1], matrix[y+1, x], matrix[y+1, x+1] ],
      ])
      # Compute hx and hy.
      hy: int = calc_correlation_value(sobel_y, neighbourhood)
      hx: int = calc_correlation_value(sobel_x, neighbourhood)
      # Compute the edge strength, rounded to an integer.
      #edge_strengths[y-1, x-1] = min(int(np.sqrt(hx**2 + hy**2)), 255)     -> genauer Wert
      edge_strengths[y-1, x-1] = abs(hx) + abs(hy)                        # -> gerundeter Wert wie in der Uebung
  return edge_strengths



"""
Calculates the single correlation value for two 3x3 matrix
"""
def calc_correlation_value(m1: np.ndarray, m2:np.ndarray) -> int:
  row1: int = m1[0,0]*m2[0,0] + m1[0,1]*m2[0,1] + m1[0,2]*m2[0,2]  
  row2: int = m1[1,0]*m2[1,0] + m1[1,1]*m2[1,1] + m1[1,2]*m2[1,2]
  row3: int = m1[2,0]*m2[2,0] + m1[2,1]*m2[2,1] + m1[2,2]*m2[2,2]
  return (row1 + row2 + row3)
