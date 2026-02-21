"""
Main file fuer Aufgabe 3

:author: Tamaro Walter, Simon Schoenhoeffer
:group: Praktikum Gruppe 14

Was ist die Aufgabe:
- Es soll eine Kanten/Konturerkennung mithilfe eines U-nets 

Genauer Wortlaut:
Trainieren Sie ein U-Net wie in der Vorlesung vorgestellt, das eine Segmentierung der Bilder
durch Erzeugen einer Kontur-Maske generiert.  ̈Ubernehmen Sie die gegebene Aufteilung des
Datensatzes in Trainings-, Validierungs- und Testdaten. Werten Sie die Qualit ̈at des U-Net
sinnvoll anhand der im Datensatz vorhandenen Konturmasken aus.

Idee der Implementierung:
- U-net mithilfe von pytorch bauen
- Bilder aus BSDS500/images/train zu kleineren bildern (256*256) transformieren
- Trainingsloop schreiben und fehler berechnen.
- Evaluation schreiben
"""

'''
Trainingsdatei.
'''
import torch
import os
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from unet import UNet
from bsds_dataset import BSDSDataset

def combined_loss(pred, target, pos_weight):
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred, target)
    pred_sig = torch.sigmoid(pred)
    smooth = 1e-6
    intersection = (pred_sig * target).sum()
    dice = 1 - (2 * intersection + smooth) / (pred_sig.sum() + target.sum() + smooth)
    return bce + dice

LEARNING_RATE = 3e-4
BATCH_SIZE = 4
EPOCHS = 12

# Get relevant paths
base_path = os.path.join(os.path.dirname(__file__), "..", "BSDS500", "BSDS500", "data")
test_images = os.path.join(base_path, "images", "test")
test_gt = os.path.join(base_path, "groundTruth", "test")
train_images = os.path.join(base_path, "images", "train")
train_gt = os.path.join(base_path, "groundTruth", "train")
val_images = os.path.join(base_path, "images", "val")
val_gt = os.path.join(base_path, "groundTruth", "val")

# Result path
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "unet.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = BSDSDataset(train_images, train_gt)
test_dataset = BSDSDataset(test_images, test_gt)
val_dataset = BSDSDataset(val_images, val_gt)

generator = torch.Generator().manual_seed(42)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Calculate pos_weight for BCEWithLogitsLoss
edge_pixels = 0
non_edge_pixels = 0
for _, mask, img_number in train_dataset:
    mask = (mask > 0.5).float()
    edge_pixels += mask.sum().item()
    non_edge_pixels += (mask.numel() - mask.sum().item())

if edge_pixels == 0:
    pos_weight = torch.tensor([1.0], device=device)
else:
    pos_weight = torch.tensor([non_edge_pixels / edge_pixels], device=device) 

model = UNet(in_channels=3, num_classes=1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

results = [] # result of losses for each epoch
for epoch in tqdm(range(EPOCHS)):
    model.train()
    train_running_loss = 0
    for idx, img_mask in enumerate(tqdm(train_dataloader)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)

        y_pred = model(img)
        optimizer.zero_grad()
        loss = combined_loss(y_pred, mask, pos_weight)
        train_running_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / (idx + 1)

    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(val_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            
            y_pred = model(img)
            loss = combined_loss(y_pred, mask, pos_weight)

            val_running_loss += loss.item()

        val_loss = val_running_loss / (idx + 1)

    results.append([epoch+1, train_loss, val_loss])
    scheduler.step()
    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    print("-"*30)

torch.save(model.state_dict(), MODEL_SAVE_PATH)
with open(os.path.join(os.path.dirname(__file__), "losses.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])
    writer.writerows(results)
