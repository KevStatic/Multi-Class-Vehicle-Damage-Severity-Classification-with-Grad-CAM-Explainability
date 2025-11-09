# scripts/dataset_classification.py

import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CarDamageClassificationDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Store image info
        self.images = data['images']
        self.annotations = data['annotations']

        # Build a mapping from image_id → category_id
        self.image_to_label = {}
        for ann in self.annotations:
            self.image_to_label[ann['image_id']] = ann['category_id']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_info['file_name']}")
            # Skip this sample safely
            return self.__getitem__((idx + 1) % len(self.images))
    
        image = Image.open(img_path).convert("RGB")

        # Default label 0 if not annotated
        label = self.image_to_label.get(img_info['id'], 0)

        if self.transform:
            image = self.transform(image)

        return image, label