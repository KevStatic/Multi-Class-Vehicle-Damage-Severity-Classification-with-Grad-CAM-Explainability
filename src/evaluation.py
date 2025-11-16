import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from train import DATA_DIR, BATCH_SIZE, SAVE_PATH, DEVICE  # import if same folder

# ---------------------------
# LOAD DATA (test only)
# ---------------------------
test_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), test_tf)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
class_names = test_set.classes

os.makedirs("results/plots", exist_ok=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
def load_model(num_classes=3):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()

model = load_model(len(class_names))

# ---------------------------
# PREDICTIONS
# ---------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.append(labels.item())
        y_pred.append(preds.item())

# ---------------------------
# CONFUSION MATRIX
# ---------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig_cm, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.savefig("results/plots/confusion_matrix.png", dpi=300)
plt.close()

print("\nConfusion Matrix saved → results/plots/confusion_matrix.png")

# ---------------------------
# CLASSIFICATION REPORT
# ---------------------------
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n")
print(report)

with open("results/plots/classification_report.txt", "w") as f:
    f.write(report)

print("Classification Report saved → results/plots/classification_report.txt")