import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
from gradcam import GradCAM
import glob

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/model2.pth"  # or best_model.pth
OUTPUT_DIR = "results/user_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Minor_Damage", "Moderate_Damage", "Severe_Damage"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# IMAGE TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# LOAD MODEL
# ==============================


def load_model(num_classes=3):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ==============================
# DAMAGE INTERPRETER
# ==============================
def interpret_damage(class_name):
    if "No" in class_name or "no" in class_name:
        return "Vehicle is not damaged"
    else:
        return "Vehicle is damaged"

# ==============================
# GET LATEST INPUT IMAGE
# ==============================
def get_latest_input_image():
    files = glob.glob("../inputs/*")
    if not files:
        return None
    return max(files, key=os.path.getctime)

# ==============================
# GRAD-CAM OVERLAY FUNCTION
# ==============================
def overlay_heatmap(original_bgr, cam_2d):
    h, w, _ = original_bgr.shape
    cam_resized = cv2.resize(cam_2d, (w, h))
    cam_uint8 = np.uint8(255 * cam_resized)

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)
    return overlay


# ==============================
# MAIN PREDICT + GRADCAM FUNCTION
# ==============================
def predict_and_explain(image_path):
    # Load model
    model = load_model(len(CLASS_NAMES))
    target_layer = model.features[-1][0]
    cam_generator = GradCAM(model, target_layer)

    # Load image
    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Forward pass
    output = model(input_tensor)
    _, pred_idx = torch.max(output, 1)
    predicted_class = CLASS_NAMES[pred_idx.item()]

    # --- Probabilities ---
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
    pred_prob = probs[pred_idx.item()] * 100

    # Damage interpretation
    damage_msg = interpret_damage(predicted_class)

    # Grad-CAM
    cam_2d = cam_generator.generate(input_tensor)
    img_cv = cv2.cvtColor(
        np.array(pil_img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    overlay = overlay_heatmap(img_cv, cam_2d)

    # Save result
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_gradcam.jpg")
    cv2.imwrite(out_path, overlay)

    print("\n===== RESULT =====")
    print(f"Input Image     : {image_path}")
    print(f"Predicted Class : {predicted_class} ({pred_prob:.2f}% confidence)")
    print(f"Damage Status   : {damage_msg}")
    print(f"Grad-CAM Saved  : {out_path}")

    print("\nClass probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        print(f" - {name}: {p*100:.2f}%")



# ==============================
# USER INPUT (ENTRY POINT)
# ==============================
if __name__ == "__main__":
    img_path = get_latest_input_image()

    if img_path is None:
        print("❌ No image found inside /inputs/. Please add an image and try again.")
    else:
        print(f"\nUsing latest image: {img_path}")
        predict_and_explain(img_path)