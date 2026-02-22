import cv2
import numpy as np
import glob
import sys
import os
import scipy.io
import csv
import edge_detection as ed
import test_edge_detection as test

base_path = os.path.join(os.path.dirname(__file__), "..")
test_folder = os.path.join(base_path, "BSDS500", "BSDS500", "data", "images", "test")

# Find all images in the test folder
image_paths = sorted(glob.glob(os.path.join(test_folder, "*.*")))
if len(image_paths) == 0:
    print(f"No images found in {test_folder}")
    sys.exit(1)

print(f"Found {len(image_paths)} images in test folder.\n")
results = []

for i, img_path in enumerate(image_paths):
  image_number = os.path.splitext(os.path.basename(img_path))[0]
  print(f"[{i+1}/{len(image_paths)}] Processing {image_number}", end="\n", flush=True)

  gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  gt_path = glob.glob(os.path.join(base_path, "BSDS500/BSDS500/data/groundTruth/test", f"{image_number}.mat"))

  if gray is None or len(gt_path) == 0:
    print("image or ground truth failed to read, skipping.")
    continue
  
  edges = ed.calculate_edges(gray)
  
  # Calculate mean ground truth and compare results
  mat = scipy.io.loadmat(gt_path[0])
  gt_cell = mat['groundTruth'][0]
  mean_ground_truth = None
  for i, gt in enumerate(gt_cell):
    boundaries = gt['Boundaries'][0, 0]  # binary matrix (0 and 1)
    boundaries = (boundaries > 0).astype(np.uint8)  # Ensure binary
    mean_ground_truth = boundaries if mean_ground_truth is None else np.maximum(mean_ground_truth, boundaries)

  precision, recall, accuracy, score = test.compare_results(edges, mean_ground_truth)
  results.append([image_number, round(precision,3), round(recall, 3), round(accuracy, 3), round(score, 3)])

# Get the total average:
avg_precision = round(np.mean([r[1] for r in results]), 3)
avg_recall = round(np.mean([r[2] for r in results]), 3)
avg_accuracy = round(np.mean([r[3] for r in results]), 3)
avg_score = round(np.mean([r[4] for r in results]), 3)
results.append(["total", avg_precision, avg_recall, avg_accuracy, avg_score])

# Write CSV
with open(os.path.join(os.path.dirname(__file__), "Ergebnisse", "evaluation_results.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_nr", "precision", "recall", "accuracy", "score"])
    writer.writerows(results)
print("Testing complete. See csv for results.\n")
