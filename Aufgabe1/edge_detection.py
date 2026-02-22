"""
Edge detection module.
Provides functions for Canny edge detection using Sobel operators.

:author: Tamaro Walter, Simon Schoenhoeffer
:group: Praktikum Gruppe 14
"""
import numpy as np


"""
Wraps all functions below in one easy to use function.
"""
def calculate_edges(matrix: np.ndarray) -> np.ndarray:
  filtered_matrix = gaussian_filter(matrix)
  strengths, directions = sobel(filtered_matrix)
  max_strengths = non_maximum_suppression(strengths, directions)
  return hysteresis_threshold_operation(max_strengths)


"""
Calculates the edge strengths of an grey-value image with sobel-operator and calculates the edge direction
"""
def sobel(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
  # Pad to preserve dimensions
  padded = np.pad(matrix, 1, mode='reflect')
  edge_strengths: np.ndarray = np.zeros((rows, cols), dtype=np.uint8)
  edge_directions: np.ndarray = np.zeros((rows, cols), dtype=np.uint8)
  
  for y in range(1, rows + 1):
    for x in range(1, cols + 1):
      neighbourhood: np.ndarray = np.array([
        [padded[y-1, x-1], padded[y-1, x], padded[y-1, x+1] ],
        [padded[y, x-1], padded[y, x], padded[y, x+1] ],
        [padded[y+1, x-1], padded[y+1, x], padded[y+1, x+1] ],
      ])
      hy: int = calc_correlation_value(sobel_y, neighbourhood)
      hx: int = calc_correlation_value(sobel_x, neighbourhood)
      edge_strengths[y-1, x-1] = min(int(np.sqrt(hx**2 + hy**2)), 255)

      angle = np.degrees(np.arctan2(hy, hx)) % 180
      if ((angle >=0 and angle< 22.5) or (angle >= 157.5 and angle <180)):
        edge_directions[y-1, x-1] = 1
      elif ((angle >=22.5 and angle< 67.5)):
        edge_directions[y-1, x-1] = 2
      elif ((angle >=67.5 and angle< 112.5)):
        edge_directions[y-1, x-1] = 3
      elif ((angle >=112.5 and angle< 157.5)):
        edge_directions[y-1, x-1] = 4
  return edge_strengths, edge_directions

"""
Executes non maximum suppression on a matrix of edge strengths
"""
def non_maximum_suppression(strengths: np.ndarray, directions: np.ndarray) -> np.ndarray:
  rows, cols = strengths.shape
  for y in range(0, rows):
    for x in range(0, cols):
      match directions[y, x]:
        case 1: # vertical direction, check with neighbours left and right
          neighbour1:int = strengths[y,x-1] if (x>0) else 0
          neighbour2:int = strengths[y,x+1] if (x<cols-1) else 0
        case 2: # bot left to top right direction, check with neighbours top left and bot right
          neighbour1:int = strengths[y-1,x-1] if (y>0 and x>0) else 0
          neighbour2:int = strengths[y+1,x+1] if (y<rows-1 and x<cols-1) else 0
        case 3: # horizontal direction, check with neighbours top and bot
          neighbour1:int = strengths[y-1,x] if (y>0) else 0
          neighbour2:int = strengths[y+1,x] if (y<rows-1) else 0
        case 4: # top left to bot right direction, check with neighbours bot left and top right
          neighbour1:int = strengths[y+1,x-1] if (y<rows-1 and x>0) else 0
          neighbour2:int = strengths[y-1,x+1] if (y>0 and x<cols-1) else 0
      # Update edge strength wir non_maximum_suppression.
      strengths[y,x] = 0 if (neighbour1 > strengths[y,x] or neighbour2 > strengths[y,x]) else strengths[y,x] 
  return strengths

"""
Implements hysteresis threshold operation algorithm from VL4, F.25
"""
def hysteresis_threshold_operation(strengths: np.ndarray) -> np.ndarray:
  # Define the 2 thresholds.
  th: int = np.percentile(strengths, 92)
  tl: int = np.percentile(strengths, 91)
  rows, cols = strengths.shape
  # Step 1: Initialize k(r,c) = 0
  k = np.zeros((rows, cols), dtype=np.uint8)
      
  # Step 2: Marking k(r,c) = 1 for all points (r, c) with s(r, c) ≥ Th
  for y in range(0, rows):
    for x in range(0, cols):
      k[y,x] = 1 if strengths[y,x] >= th else 0

  # Step 3: Mark K(r, c) = 1 for all points (r, c) with Tl ≤ s(r, c) < Th that have at least one neighbor (i′, j′) with K(i′, j′) = 1.
  # Step 4: Repeat until the result becomes stable.
  stable = False
  while (not stable):
    stable = True
    for y in range(0, rows):
      for x in range(0, cols):
        # Check if the pixel is a candidate
        if k[y, x] == 0 and strengths[y, x] >= tl and strengths[y, x] < th:
          # Check all 8 neighbors
          for dy in range(-1, 2):
            for dx in range(-1, 2):
              ny, nx = y + dy, x + dx
              if 0 <= ny < rows and 0 <= nx < cols and k[ny, nx] == 1:
                k[y, x] = 1
                stable = False # it becomes unstable if a change ocurred
                break
            if k[y, x] == 1:
              break

  # Step 5: As a line was taken at the beginning (sobel function) from the original matrix, add a blank line allround the image
  return k


"""
Implements gaussian filter
"""
def gaussian_filter(matrix: np.ndarray) -> np.ndarray:
  # Build the mask
  sigma = 1
  k = int(3 * sigma)
  mask_size = int(2 * k + 1)
  gauss: np.ndarray = np.zeros((mask_size, mask_size), dtype=np.float64)
  for y in range(-k, k + 1):
    for x in range(-k, k + 1):
      gauss[y + k, x + k] = round(g(x, y, sigma)/g(k,k,sigma))
  normfactor = 1 / np.sum(gauss)

  # Pad the image to preserve dimensions
  padded = np.pad(matrix, k, mode='reflect')

  # Apply mask to image matrix
  rows, cols = matrix.shape
  result = np.zeros((rows, cols), dtype=np.uint8)
  for y in range(k, rows + k):
    for x in range(k, cols + k):
      region = padded[(y-k):(y+k+1), (x-k):(x+k+1)]
      result[y - k, x - k] = int(np.sum(region * gauss) * normfactor)

  return result



"""
Helper function. Calculates the single correlation value for two 3x3 matrix
"""
def calc_correlation_value(m1: np.ndarray, m2:np.ndarray) -> float:
  row1: int = m1[0,0]*m2[0,0] + m1[0,1]*m2[0,1] + m1[0,2]*m2[0,2]
  row2: int = m1[1,0]*m2[1,0] + m1[1,1]*m2[1,1] + m1[1,2]*m2[1,2]
  row3: int = m1[2,0]*m2[2,0] + m1[2,1]*m2[2,1] + m1[2,2]*m2[2,2]
  return (row1 + row2 + row3)

"""
Helper function. Calculates the gaussian value for 2D component. 
"""
def g(x:int, y:int, sigma:int) -> float:
  zaehler = np.exp(-1 * ((x**2 + y**2) / (2 * sigma**2)))
  nenner = 2 * np.pi * sigma**2
  return (zaehler/nenner)