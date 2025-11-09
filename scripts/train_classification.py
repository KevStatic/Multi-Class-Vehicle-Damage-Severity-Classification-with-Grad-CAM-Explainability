# scripts/train_classification.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_classification import CarDamageClassificationDataset
from dataset_custom import CarDamageDataset

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Paths
root_dir = "../Datasets/CarDamageDetectionCocoDataset/train/"
annotation_file = "../Datasets/CarDamageDetectionCocoDataset/train/_annotations.coco.json"

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Dataset & Loader
dataset = CarDamageClassificationDataset(root_dir, annotation_file, transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
train_data = CarDamageDataset(root_dir, annotation_file, transform=transform)

# Model, Loss, Optimizer
model = SimpleCNN(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"[INFO] Number of training samples: {len(train_data)}")
print(f"[INFO] Unique labels: {set([label for _, label in train_data])}")
print(f"[INFO] Example label names: {[train_data.images[i]['file_name'] for i in range(min(3, len(train_data.images)))]}")

# Training loop
for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/5], Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "car_damage_classifier.pth")
print("✅ Model saved successfully.")
