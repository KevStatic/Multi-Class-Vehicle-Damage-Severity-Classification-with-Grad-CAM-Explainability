# src/run_gradcam.py

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

from gradcam import GradCAM

# adjust if needed
MODEL_PATH = "models/model.pth"      # or "models/best_model.pth"
OUTPUT_DIR = "results/gradcam"

IMAGE_PATHS = [
    "../data/test/Minor_Damage/0011.jpeg",
    "../data/test/Moderate_Damage/0024.JPEG",
    "../data/test/Severe_Damage/0050.JPEG",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CLASS_NAMES = ["Minor_Damage", "Moderate_Damage", "Severe_Damage"]


def load_model(num_classes=3):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def overlay_heatmap(original_bgr, cam_2d):
    """
    original_bgr: uint8 image (H,W,3)
    cam_2d: float [0,1] array (h,w) -- will be resized
    """
    h, w, _ = original_bgr.shape

    # resize CAM to full image size
    cam_resized = cv2.resize(cam_2d, (w, h))
    cam_uint8 = np.uint8(255 * cam_resized)

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # blend
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)
    return overlay


if __name__ == "__main__":
    model = load_model(num_classes=len(CLASS_NAMES))
    # last conv block in MobileNetV2
    target_layer = model.features[-1][0]
    gradcam = GradCAM(model, target_layer)

    for img_path in IMAGE_PATHS:
        if not os.path.exists(img_path):
            print(f"❌ Missing: {img_path}")
            continue

        # load & preprocess
        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # generate CAM (low-res)
        cam = gradcam.generate(input_tensor)  # 2D, [0,1]

        # original image as BGR uint8, resized to 224x224
        img_cv = cv2.cvtColor(np.array(pil_img.resize((224, 224))), cv2.COLOR_RGB2BGR)

        overlay = overlay_heatmap(img_cv, cam)

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}_gradcam.jpg")
        cv2.imwrite(out_path, overlay)
        print(f"✅ Saved Grad-CAM: {out_path}")
# ---------------------------