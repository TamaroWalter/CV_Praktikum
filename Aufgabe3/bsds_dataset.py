"""
Klasse, die den Datensatz speichert. Speichert Trainings-, Test- und Valierungsbilder.
Speichert pro Bild 1 mal das Bild und 1 mal den Ground truth des bildes (der erste, der da ist)
"""

import os
import glob
import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class BSDSDataset(Dataset):
    def __init__(self, image_dir, gt_dir, size=(512, 512)):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*')))
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.mat')))
        self.size = size
        self.transform_img = transforms.Compose([
            transforms.Resize(size, interpolation=Image.BILINEAR),
            transforms.ToTensor()])
        self.transform_mask = transforms.Compose([
            transforms.Resize(size, interpolation=Image.NEAREST),  # or NEAREST for binary
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform_img(img)

        # Load ground truth .mat file
        img_name = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
        gt_path = os.path.join(os.path.dirname(self.gt_paths[0]), f"{img_name}.mat")
        mat = scipy.io.loadmat(gt_path)
        gt_cell = mat['groundTruth'][0]
        # Average all annotators' boundaries
        boundaries = [gt[0][0]['Boundaries'] for gt in gt_cell]
        mean_mask = np.mean(np.stack(boundaries, axis=0), axis=0)
        # Convert to PIL for resizing
        mask = Image.fromarray((mean_mask * 255).astype(np.uint8))
        mask = self.transform_mask(mask)
        mask = mask.float() / mask.max()  # Normalize to 0-1

        return img, mask,  os.path.basename(self.image_paths[idx])
