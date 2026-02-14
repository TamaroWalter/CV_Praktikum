import cv2
import numpy as np
import glob
import edge_detection as ed
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
3. Schritt: Richtung der Kanten bestimmen (wird glaub ich nicht in vorlesung behandelt)
4. Schritt: Non-maximum Suppression anwenden um geeignete Kanten zu finden
5. Schritt: Hysteresis-Threshold-Operation verwenden um Kanten zu identifizieren
6. Schritt: Matrix wieder zu einer Jpg umwandeln, Konturen als schwarze Striche

Wie man dann testen könnte:
1. Schritt: Vom gleichen bild die groundTruth-Matlab datei einlesen und die Matrix mit der eigens berechneten Matrix vergleichen
'''
path = glob.glob("/home/tamaro/Documents/master_informatik/Praktische_Informatik/ComputerVision/Praktikum/BSDS500/BSDS500/data/images/**/2018.*", recursive=True)
#print(path)

#gray_value_matrix: np.ndarray = cv2.imread(path[0], cv2.IMREAD_GRAYSCALE)
# Test from exercise 1.5
gray_value_matrix: np.ndarray = np.array([
  [3, 1, 5, 7, 2],
  [2, 1, 6, 5, 6],
  [1, 0, 7, 6, 7],
  [0, 1, 2, 6, 5],
  [2, 1, 0, 1, 6],
])
edges = ed.sobel(gray_value_matrix)
print(edges)
'''
expected output:
[[18 22  8]
 [24 28  6]
 [12 34 30]]
'''
