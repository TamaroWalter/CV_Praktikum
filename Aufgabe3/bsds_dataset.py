"""
Klasse, die den Datensatz speichert. Speichert Trainings-, Test- und Valierungsbilder.
Speichert pro Bild 1 mal das Bild und 1 mal den Ground truth des bildes (der erste, der da ist)
"""
import os
import glob
import random
import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class BSDSDataset(Dataset):
    def __init__(self, image_dir, gt_dir, size=(256, 256), augment=False):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*')))
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.mat')))
        self.size = size
        self.augment = augment
        self.transform_img = transforms.Compose([
            transforms.Resize(size, interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def _augment(self, img, mask):
        # Horizontale Spiegelung ist realistisch (Spiegelbild einer Szene)
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        # Kleine Rotationen (+/- 10°) statt 90°-Schritte
        angle = random.uniform(-10, 10)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)
        # Farbveränderungen etwas reduzieren
        img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)(img)
        return img, mask

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')

        img_name = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
        gt_path = os.path.join(os.path.dirname(self.gt_paths[0]), f"{img_name}.mat")
        mat = scipy.io.loadmat(gt_path)
        gt_cell = mat['groundTruth'][0]

        mean_mask = None
        for gt in gt_cell:
            temp = gt["Boundaries"][0, 0]
            temp = (temp > 0).astype(np.uint8)  # Binary mask
            mean_mask = temp if mean_mask is None else np.maximum(mean_mask, temp)

        mask = Image.fromarray((mean_mask * 255).astype(np.uint8))

        # Augmentation vor dem Resizen, damit beide noch als PIL-Objekte vorliegen
        if self.augment:
            img, mask = self._augment(img, mask)

        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        mask = mask.float().clamp(0, 1)

        return img, mask, os.path.basename(self.image_paths[idx])