import cv2
import numpy as np
import glob
import sys
import time
import scipy.io
import os
import edge_detection as ed
import test_edge_detection as test
'''
Was ist die Aufgabe?
- Es soll eine Kantenerkennung eines Bildes mit "Canny Edge Detection" implementiert werden.

Genauer Wortlaut:
Implementieren Sie Canny Edge Detection wie in Vorlesung 4 einmal mit und einmal ohne
Bildgl ̈attung. Benutzen Sie fuer die Gradientenberechnung im Canny-Algorithmus einen Operator
Ihrer Wahl (z. B. Sobel). Ueberlegen Sie sich eine Metrik und werten Sie die Qualitaet des Canny-
Edge Detectors sinnvoll anhand der im Datensatz vorhandenen Konturmasken aus

Idee der Implementation:
- Input: eine JPG aus dem BSDS500 set
- Ausgabe: eine Matrix mit den Kanten nach der Hyteresis-Threshold-Operation

Algorithmus:
1. Schritt: Jpg zu grauwerte-Matrix umwandeln   -> check
2. Schritt: Mit Sobel-Operator Kantenstärke ermitteln ->check
3. Schritt: Richtung der Kanten bestimmen -> check
4. Schritt: Non-maximum Suppression anwenden um geeignete Kanten zu finden -> check
5. Schritt: Hysteresis-Threshold-Operation verwenden um Kanten zu identifizieren -> check
6. Schritt: Matrix wieder zu einer Jpg umwandeln, Konturen als schwarze Striche -> check

Wie man dann testen könnte:
1. Schritt: Vom gleichen bild die groundTruth-Matlab datei einlesen und die Matrix mit der eigens berechneten Matrix vergleichen
  1.1 matlab datei einlesen -> check (es sind mehrere versionen desselben bildes vorhanden)
'''

# Start timer.
start = time.time()

# Get image number from user input:
if len(sys.argv) < 2:
  print("Usage: python main.py <image_number>")
  sys.exit(1)

image_number = sys.argv[1]

# Get the image from BSDS500 repo and turn it into a gray value matrix
base_path = os.path.join(os.path.dirname(__file__), "..")
path = glob.glob(os.path.join(base_path, "BSDS500/BSDS500/data/images/**", f"{image_number}.*"), recursive=True)
if len(path) == 0:
  print(f"No image found with number {image_number}")
  sys.exit(1)

print(f"Processing image: {path[0]}")
gray_value_matrix: np.ndarray = cv2.imread(path[0], cv2.IMREAD_GRAYSCALE)

# Compute Canny edge detection.
edges = ed.calculate_edges(gray_value_matrix)

# Output edge file.
output = np.where(edges == 1, 0, 255).astype(np.uint8)
cv2.imwrite(f"edges_{image_number}.jpg", output)

# TESTING against ground truth
# Load ground truth .mat file
gt_start = time.time()
gt_path = glob.glob(os.path.join(base_path, "BSDS500/BSDS500/data/groundTruth/**", f"{image_number}.mat"), recursive=True)
ground_truth = np.zeros((0, 0), dtype=np.uint8)
if len(gt_path) == 0:
  print(f"No ground truth found for {image_number}")
else:
  mat = scipy.io.loadmat(gt_path[0])
  # groundTruth is a 1xN array, each entry contains Boundaries and Segmentation
  # Multiple annotators, so there are multiple ground truths
  gt_cell = mat['groundTruth'][0]
  ground_truth = gt_cell[0]['Boundaries'][0, 0]
  '''
  for i, gt in enumerate(gt_cell):
    boundaries = gt['Boundaries'][0, 0]  # binary matrix (0 and 1)
    ground_truth = boundaries
    gt_output = np.where(boundaries == 1, 0, 255).astype(np.uint8)
    cv2.imwrite(f"gt_{image_number}_{i}.jpg", gt_output)
  '''
cv2.imwrite(f"original_{image_number}.jpg", cv2.imread(path[0]))
print(f"Processing took {(time.time() - start):.2f} seconds")
print("----")
print("---")
test.compare_results(edges, ground_truth)
