import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import torch.utils.data
from dataset_detection import CarDamageDetectionDataset
from tqdm import tqdm

# ===== CONFIG =====
root_dir = "../Datasets/CarDamageDetectionCocoDataset/train"
annotation_file = "../Datasets/CarDamageDetectionCocoDataset/train/_annotations.coco.json"
num_classes = 2  # background + damage
num_epochs = 10
batch_size = 4
learning_rate = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.ToTensor(),
])

# ===== DATASET =====
dataset = CarDamageDetectionDataset(root_dir, annotation_file, transforms=transform)

# NOTE: No validation split needed right now (optional)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: tuple(zip(*batch))
)

# ===== MODEL (Faster R-CNN) =====
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# Replace classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

# ===== OPTIMIZER =====
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ===== TRAINING LOOP =====
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        pbar.set_postfix({"batch_loss": losses.item()})

    print(f"Epoch {epoch+1} Loss: {total_loss/len(data_loader):.4f}")

# ===== SAVE MODEL =====
torch.save(model.state_dict(), "damage_detector_frcnn.pth")
print("✅ Faster R-CNN model saved as 'damage_detector_frcnn.pth'!")
