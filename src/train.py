import os
from pyexpat import model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
from copy import deepcopy
from plot_training_curves import plot_curves

# =============================
# CONFIGURATION
# =============================
DATA_DIR = "../data"
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 8
SAVE_PATH = "models/model2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# DATA LOADING
# =============================
def get_dataloaders():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "training"), train_tf)
    val_set   = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), test_tf)
    test_set  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), test_tf)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    sizes = {
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set)
    }

    classes = train_set.classes
    return dataloaders, sizes, classes


# =============================
# MODEL SETUP
# =============================
def create_model(num_classes):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    # freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # 2) Unfreeze LAST block for fine-tuning
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # replace FC head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# =============================
# TRAINING FUNCTION
# =============================

def train_model(model, dataloaders, sizes):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # initialize metric containers
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_acc = 0.0
    best_weights = deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_correct / sizes[phase]

            print(f"{phase.upper()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "train":
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

                # best model checkpointing
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_weights = deepcopy(model.state_dict())
                    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                    torch.save(best_weights, SAVE_PATH)
                    print(f"💾 Saved best model so far (Val Acc: {best_acc:.4f})")

    # load best weights
    model.load_state_dict(best_weights)
    return model, train_losses, val_losses, train_accs, val_accs



# =============================
# TEST FUNCTION
# =============================
def evaluate(model, test_loader, size):
    model.eval()
    corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels).item()

    test_acc = corrects / size
    print(f"\n✔ TEST ACCURACY: {test_acc:.4f}\n")
    return test_acc


# =============================
# MAIN DRIVER
# =============================
if __name__ == "__main__":
    dataloaders, sizes, classes = get_dataloaders()
    print(f"Classes: {classes}")

    model = create_model(len(classes)).to(DEVICE)
    model, train_losses, val_losses, train_accs, val_accs = train_model(model, dataloaders, sizes)
    evaluate(model, dataloaders["test"], sizes["test"])
    plot_curves(train_losses, val_losses, train_accs, val_accs)