import torch
import numpy as np
from PIL import Image
import os
from unet import UNet
from bsds_dataset import BSDSDataset
import torchvision.transforms as T
from scipy.ndimage import maximum_filter
import cv2

def predict_contour(image_path, model_path, output_path, device="cuda"):
    # Bild laden und transformieren
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((512, 512), interpolation=Image.BILINEAR),
        T.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Modell laden
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Vorhersage
    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred_bin = (pred > 0.5).cpu().numpy()[0, 0]  # [1, 1, H, W] -> [H, W]
        
    # Kontur als Bild speichern (schwarz/weiß)
    contour_img = (pred_bin * 255).astype(np.uint8)
    Image.fromarray(contour_img).save(output_path)
    print(f"Kontur gespeichert als {output_path}")

if __name__ == "__main__":
    # Beispielaufruf
    #image_number = "64061"  # z.B. Bildnummer 2018, 3063, 29030, 6046, 64061
    for image_number in ["2018", "3063", "29030", "6046", "64061", "196027", "14092"]:
        base_path = os.path.join(os.path.dirname(__file__), "..", "BSDS500", "BSDS500", "data", "images", "test")
        image_path = os.path.join(base_path, f"{image_number}.jpg")
        model_path = os.path.join(os.path.dirname(__file__), "excel_unet.pth")
        output_path = os.path.join(os.path.dirname(__file__), f"b_edges_{image_number}.png")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predict_contour(image_path, model_path, output_path, device)