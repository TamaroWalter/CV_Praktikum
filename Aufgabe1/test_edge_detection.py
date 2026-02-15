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
  print(f"edges shape: {edges.shape}")
  print(f"ground truth shape: {ground_truth.shape}")
  if (gt_rows != e_rows or e_cols != gt_cols):
    print("ERROR: matrices dont have same dimensions")
    return [0,0,0]
  percent_gt = np.sum(ground_truth) / (gt_rows*gt_cols)
  percent_edges = np.sum(edges) / (e_rows*e_cols)
  tp = 0
  fn = 0
  fp = 0
  for y in range(0, e_rows):
    for x in range(0, e_cols):
      if (ground_truth[y, x] == edges[y, x]):
        tp += 1          
      if (ground_truth[y, x] == 1 and edges[y, x] == 0):
        fn += 1
      if (ground_truth[y, x] == 0 and edges[y, x] == 1):
        fp += 1
  
  # Calculate precision ("Of all the edges I detected, how many are actually real edges?")
  precision = tp / (tp+fp)

  # Calculate recall ("Of all the real edges, how many did I actually find?")
  recall = tp / (tp+fn)

  # Calculate score:
  score = 2 * ((precision*recall)/(precision+recall))
  bar_len = 30
  tp_bar = "█" * int(precision * bar_len)
  rc_bar = "█" * int(recall * bar_len)
  f1_bar = "█" * int(score * bar_len)
  '''
  print(f"""
  ╔═══════════════════════════════════════════════════════╗
  ║            🔍 EDGE DETECTION RESULTS                  ║
  ╠═══════════════════════════════════════════════════════╣
  ║                                                       ║
  ║  ✅ True Positives:   {tp:>8}                         ║
  ║  ❌ False Positives:  {fp:>8}                         ║
  ║  ⬛ False Negatives:  {fn:>8}                         ║
  ║                                                       ║
  ╠═══════════════════════════════════════════════════════╣
  ║                                                       ║
  ║  Precision: {precision:>6.2%}  [{tp_bar:<{bar_len}}]  ║
  ║  Recall:    {recall:>6.2%}  [{rc_bar:<{bar_len}}]     ║
  ║  F1 Score:  {score:>6.2%}  [{f1_bar:<{bar_len}}]      ║
  ║                                                       ║
  ╚═══════════════════════════════════════════════════════╝
  """)
  '''
  return [precision, recall, score]
