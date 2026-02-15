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
    print(f"[{i+1}/{len(image_paths)}] Processing {image_number}...", end=" ")

    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("failed to read, skipping.")
        continue

    edges = ed.calculate_edges(gray)

    # Load ground truth
    gt_path = glob.glob(os.path.join(base_path, "BSDS500/BSDS500/data/groundTruth/test", f"{image_number}.mat"))
    if len(gt_path) == 0:
        print("no ground truth, skipping.")
        continue

    mat = scipy.io.loadmat(gt_path[0])
    gt_cell = mat['groundTruth'][0]
    ground_truth = gt_cell[0]['Boundaries'][0, 0]

    if edges.shape != ground_truth.shape:
        print(f"shape mismatch: edges={edges.shape}, gt={ground_truth.shape}, skipping.")
        continue

    precision, recall, score = test.compare_results(edges, ground_truth)
    precision = round(precision, 3)
    recall = round(recall, 3)
    score = round(score, 3)

    results.append([image_number, precision, recall, score])
    print(f"P={precision:.3f} | R={recall:.3f} | F1={score:.3f}")

# Write CSV
output_file = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_nr", "precision", "recall", "score"])
    writer.writerows(results)

# Print summary
if results:
    avg_p = round(np.mean([r[1] for r in results]), 3)
    avg_r = round(np.mean([r[2] for r in results]), 3)
    avg_f1 = round(np.mean([r[3] for r in results]), 3)
    print(f"""
╔══════════════════════════════════════════╗
║           📊 BATCH EVALUATION            ║
╠══════════════════════════════════════════╣
║  Images processed:  {len(results):>5}               ║
║  Avg Precision:     {avg_p:.3f}               ║
║  Avg Recall:        {avg_r:.3f}               ║
║  Avg F1 Score:      {avg_f1:.3f}               ║
╚══════════════════════════════════════════╝
""")

print(f"Results written to {output_file}")