import torch
import torch.nn as nn

class VehicleDamageModel(nn.Module):
    def __init__(self, num_classes=2):
        super(VehicleDamageModel, self).__init__()
        
        # --- Feature Extractor (like a mini CNN) ---
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # --- Detection Head ---
        # Output: [batch, 5 + num_classes, H, W]
        # 5 = [x, y, w, h, confidence]
        self.detector = nn.Conv2d(128, 5 + num_classes, 1)

    def forward(self, x):
        x = self.features(x)
        out = self.detector(x)
        return out
