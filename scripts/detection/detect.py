import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision import transforms
import os

# ===== CONFIG =====
model_path = "damage_detector_frcnn.pth"
confidence_threshold = 0.5

# ===== Load Model =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

print("✅ Model loaded successfully!")

# ===== Image Transform =====
transform = transforms.Compose([
    transforms.ToTensor()
])

# ===== Prediction Function =====
def detect_damage(image_path):
    if not os.path.exists(image_path):
        print("❌ Error: Image file not found.")
        return

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]["boxes"]
    scores = outputs[0]["scores"]
    labels = outputs[0]["labels"]

    # Filter by confidence threshold
    final_boxes = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= confidence_threshold:
            final_boxes.append((box.cpu(), score.cpu()))

    if len(final_boxes) == 0:
        print("⚠ No damage detected above threshold.")
        img.show()
        return

    # Draw boxes
    draw = ImageDraw.Draw(img)
    for box, score in final_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"damage {score:.2f}", fill="red")

    # Show result
    img.show()

    # Save result
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "detected_damage.jpg")
    img.save(save_path)
    print(f"💾 Saved detection result to: {save_path}")

# ===== User Prompt =====
print("\n🔍 Enter image path to run damage detection:")
image_path = input("Image path: ").strip().strip('"')

detect_damage(image_path)
