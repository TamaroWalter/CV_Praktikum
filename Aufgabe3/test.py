"""
Test datei. Evaluiert das Modell auf dem Testdatensatz. Berechnet die Precision, Recall und F1-Score für die Kantenextraktion.
"""

import torch
import os
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
import csv
from bsds_dataset import BSDSDataset

if __name__ == "__main__":
  BATCH_SIZE = 8
  # Get relevant paths
  base_path = os.path.join(os.path.dirname(__file__), "..", "BSDS500", "BSDS500", "data")
  test_images = os.path.join(base_path, "images", "test")
  test_gt = os.path.join(base_path, "groundTruth", "test")

  device = "cuda" if torch.cuda.is_available() else "cpu"

  test_dataset = BSDSDataset(test_images, test_gt)
  test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)

  model = UNet(in_channels=3, num_classes=1).to(device)
  model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "unet.pth")))

  model.eval()
  precisions = []
  recalls = []
  f1_scores = []
  results = []
  with torch.no_grad():
    for img, mask, img_names in tqdm(test_dataloader):
      img = img.float().to(device)
      mask = mask.float().to(device)
      y_pred = model(img)
      y_pred = torch.sigmoid(y_pred)
      y_pred_bin = (y_pred > 0.5).cpu().numpy()
      mask_bin = (mask > 0.5).cpu().numpy()

      for i in range(y_pred_bin.shape[0]):
        prediction = y_pred_bin[i, 0]
        gt = mask_bin[i, 0]
        TP = np.sum((prediction == 1) & (gt == 1))
        FP = np.sum((prediction == 1) & (gt == 0))
        FN = np.sum((prediction == 0) & (gt == 1))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * ((precision*recall)/(precision+recall)) if (precision+recall) > 0 else 0
        results.append([img_names[i], precision, recall, f1_score])

  # Get the total average:
  avg_p, avg_r, avg_f1 = [0,0,0]
  if results:
    avg_p = round(np.mean([r[1] for r in results]), 3)
    avg_r = round(np.mean([r[2] for r in results]), 3)
    avg_f1 = round(np.mean([r[3] for r in results]), 3)
    results.append(["total", avg_p, avg_r, avg_f1])
  # Write CSV
  output_file = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")
  with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_nr", "precision", "recall", "score"])
    writer.writerows(results)