"""
Docstring for Aufgabe1.test_edge_detection
Tests the result of the calculate_edges function by comparing it to the .mat file of a picture from BSDS500
"""
import numpy as np

"""
Compares the calculated edges with the ground truth. Both are matrices of the same size with 1 (edge) and zeros (no edge)
"""
def compare_results(edges: np.ndarray, ground_truth: np.ndarray):
  '''Compare 3 things:
  - true-positive: when the ground truth has a 1, the edges has a 1 too?
  - false-negative: when the ground truth has a 1, the edge has a 0
  - false-positive: when the ground truth has a 0, the edge has a 1
  '''
  gt_rows, gt_cols = ground_truth.shape
  e_rows, e_cols = edges.shape
  if (gt_rows != e_rows or e_cols != gt_cols):
    print("ERROR: matrices dont have same dimensions")
    return [0,0,0]
  tp = 0
  tn = 0
  fn = 0
  fp = 0
  for y in range(0, e_rows):
    for x in range(0, e_cols):
      if (ground_truth[y, x] == 1 and edges[y, x] == 1):
        tp += 1          
      if (ground_truth[y, x] == 0 and edges[y, x] == 0):
        tn += 1
      if (ground_truth[y, x] == 1 and edges[y, x] == 0):
        fn += 1
      if (ground_truth[y, x] == 0 and edges[y, x] == 1):
        fp += 1
  
  # Calculate precision ("Of all the edges I detected, how many are actually real edges?")
  precision = tp / (tp+fp)

  # Calculate recall ("Of all the real edges, how many did I actually find?")
  recall = tp / (tp+fn)

  # Calculate Accuracy ("Of all the pixels, how many did I classify correctly?")
  accuracy  = (tp+tn) / (tp +tn+ fp + fn)
  
  # Calculate f1-score:
  score = 2 * ((precision*recall)/(precision+recall))

  return [precision, recall, accuracy, score]
