import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from model_custom import VehicleDamageModel

# =============================
# CONFIGURATION
# =============================

BASE_DIR = "../Datasets/coco"
TRAIN_IMG_DIR = f"{BASE_DIR}/train"
VAL_IMG_DIR = f"{BASE_DIR}/val"
TRAIN_ANN_FILE = f"{BASE_DIR}/annotations/COCO_train_annos.json"
VAL_ANN_FILE = f"{BASE_DIR}/annotations/COCO_val_annos.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# MAIN TRAINING CODE
# =============================

def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load COCO dataset
    print("Loading datasets...")
    train_dataset = CocoDetection(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transform=transform)
    val_dataset = CocoDetection(VAL_IMG_DIR, VAL_ANN_FILE, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    print("Initializing model...")
    model = VehicleDamageModel(num_classes=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(5):
        model.train()
        for images, targets in train_loader:
            images = torch.stack(images).to(DEVICE)
            # You can adapt target preprocessing here depending on annotations
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, torch.zeros(len(images), dtype=torch.long).to(DEVICE))
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "../vehicle_damage_model_custom.pth")
    print("✅ Training complete. Model saved as vehicle_damage_model_custom.pth")


# Required for Windows multiprocessing
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_model()
