# scripts/predict_image.py
import math
import argparse
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

# ---- args ----
parser = argparse.ArgumentParser()
parser.add_argument("--image", "-i", type=str,
                    default="../Datasets/CarDamageBinary/valid/undamaged/",
                    help="Path to an image or a folder to pick one image from")
args = parser.parse_args()

# ---- pick image ----
image_path = args.image
if os.path.isdir(image_path):
    # pick a random image from the folder
    import random
    choices = [f for f in os.listdir(image_path)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_path = os.path.join(image_path, random.choice(choices))

print(f"Using image: {image_path}")

# ---- model ----
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("car_damage_classifier.pth", map_location="cpu"))
model.eval()

# ---- transforms (same as training) ----
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# IMPORTANT: match ImageFolder alphabetical mapping
class_names = ["damaged", "undamaged"]  # 0->damaged, 1->undamaged

# ---- inference ----
img = Image.open(image_path).convert("RGB")
x = transform(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs))
    pred_label = class_names[pred_idx]

print(f"Prediction: {pred_label}")
print(f"Probabilities: damaged={probs[0]:.3f}, undamaged={probs[1]:.3f}")
