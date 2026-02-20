import os
import glob
import random
import numpy as np
import cv2
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset, DataLoader


def load_union_boundary_map(mat_path: str) -> np.ndarray:
    """
    Returns union/OR boundary map as uint8 array of shape (H, W) with values 0/1.
    """
    mat = loadmat(mat_path)
    gt = mat["groundTruth"]  # usually shape (1, K)
    # Convert to list of annotator structs:
    annotators = gt[0]

    union = None

    for ann in annotators:
        # ann is usually a numpy.void with fields
        # Commonly: ann["Boundaries"][0,0] is HxW logical/uint8
        b = ann["Boundaries"][0, 0]
        b = (b > 0).astype(np.uint8)
        union = b if union is None else np.maximum(union, b)

    # union = annotators[0]["Boundaries"][0, 0]

    return union  # 0/1


def extract_patch_features_rgb_sobel(img_bgr, gx, gy, y, x, patch):
    assert patch % 2 == 1
    r = patch // 2

    img = img_bgr.astype(np.float32) / 255.0
    img = np.pad(img, ((r,r),(r,r),(0,0)), mode="reflect")

    gx_p = np.pad(gx, ((r,r),(r,r)), mode="reflect")
    gy_p = np.pad(gy, ((r,r),(r,r)), mode="reflect")

    yy = y + r
    xx = x + r

    D = patch*patch*3 + 2*patch*patch
    feats = np.empty((len(y), D), dtype=np.float32)

    for i, (cy, cx) in enumerate(zip(yy, xx)):
        prgb = img[cy-r:cy+r+1, cx-r:cx+r+1, :].reshape(-1)
        pgx  = gx_p[cy-r:cy+r+1, cx-r:cx+r+1].reshape(-1)
        pgy  = gy_p[cy-r:cy+r+1, cx-r:cx+r+1].reshape(-1)
        feats[i] = np.concatenate([prgb, pgx, pgy], axis=0)

    return feats

class SingleImagePixelDataset(Dataset):
    def __init__(self, img_bgr: np.ndarray, union: np.ndarray, coords_yx: np.ndarray, patch: int = 5):
        self.img = img_bgr
        self.union = union
        self.coords = coords_yx.astype(np.int64)
        self.patch = patch

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        y, x = self.coords[i]
        X = extract_patch_features(self.img, np.array([y]), np.array([x]), self.patch)[0]  # (D,)
        ycls = int(self.union[y, x])  # 0/1
        return torch.from_numpy(X), torch.tensor(ycls, dtype=torch.long)



