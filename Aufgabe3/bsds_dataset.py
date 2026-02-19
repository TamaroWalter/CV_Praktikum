"""
Klasse, die den Datensatz speichert. Speichert Trainings-, Test- und Valierungsbilder.
Speichert pro Bild 1 mal das Bild und 1 mal den Ground truth des bildes (der erste, der da ist)
"""

import cv2
import numpy as np
import glob
import sys
import os
import scipy.io
import csv
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Dataset():

  def __init__(self):
    # Declare the paths
    base_path = os.path.join(os.path.dirname(__file__), "..", "BSDS500", "BSDS500", "data")
    self.test_images = []
    self.test_gt_images = []
    self.train_images = []
    self.train_gt_images = []
    self.val_images = []
    self.val_gt_images = []

    # Get test data
    self.get_images(os.path.join(base_path, "images", "test"),  os.path.join(base_path, "groundTruth", "test"), "test")
    # Get train data
    self.get_images(os.path.join(base_path, "images", "train"),  os.path.join(base_path, "groundTruth", "train"), "train")
    # Get validate data
    self.get_images(os.path.join(base_path, "images", "val"),  os.path.join(base_path, "groundTruth", "val"), "alidat")
    
  def get_images(self, folder, gt_folder, category):
    image_paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    for i, img_path in enumerate(image_paths):
      img_number = os.path.splitext(os.path.basename(img_path))[0]
      image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

      # Get first ground truth from image.
      gt_path = glob.glob(os.path.join(gt_folder, f"{img_number}.mat"))
      mat = scipy.io.loadmat(gt_path[0])
      gt_cell = mat['groundTruth'][0]
      ground_truth = gt_cell[0]['Boundaries'][0, 0]

      # Resize.
      image = image.resize((256, 256), Image.BILINEAR)
      ground_truth = ground_truth.resize((256, 256), Image.NEAREST)

      # Transform to Tensor so the U-net can work with it.
      image = transforms.ToTensor()(ground_truth)
      ground_truth = transforms.ToTensor()(ground_truth)

      # Append to result.
      if category == "test":
        self.test_images.append(image)
        self.test_gt_images.append(ground_truth)
      elif category == "train":
        self.train_images.append(image)
        self.train_gt_images.append(ground_truth)
      elif category == "validate":
        self.val_images.append(image)
        self.val_gt_images.append(ground_truth)