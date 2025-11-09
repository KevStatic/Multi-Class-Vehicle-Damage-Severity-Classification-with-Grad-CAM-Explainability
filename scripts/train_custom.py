# scripts/train_custom.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset_custom import CarDamageDataset
from model_custom import CarDamageCNN
import matplotlib.pyplot as plt

# Paths
data_dir = "../Datasets/CarDamageDetectionCocoDataset/train"
annotation_file = f"{data_dir}/_annotations.coco.json"

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Dataset & DataLoader
dataset = CarDamageDataset(root_dir=data_dir, annotation_file=annotation_file, transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

img, lbl = dataset[0]
print(f"Image shape: {img.shape}, Label: {lbl}")

# Model setup
num_classes = len(set([ann['category_id'] for ann in dataset.annotations]))
model = CarDamageCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Total samples in dataset: {len(dataset)}")

# Training loop
num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "../vehicle_damage_model.pth")

# Plot training loss
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()