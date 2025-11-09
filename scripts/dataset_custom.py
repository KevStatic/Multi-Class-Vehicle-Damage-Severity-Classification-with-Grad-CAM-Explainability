# scripts/dataset_custom.py
import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CarDamageDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform = None):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.trasform = transform

        with open(annotation_file) as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.categories = {cat["id"]: cat["name"] for cat in data["categories"]}

        # Map image_id -> category_id
        self.image_to_label = {}
        for ann in self.annotations:
            self.image_to_label[ann["image_id"]] = ann["category_id"]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        image_id = image_info["id"]
        label_id = self.image_to_label.get(image_id, None)

        if label_id is None:
            raise ValueError(f"No label found for image {image_info['file_name']}")
        
        label = label_id - 1

        if self.trasform:
            image = self.trasform(image)
        
        if idx < 5:
            print(f"Sample {idx}: {image_info['file_name']} -> label {label}")

        return image, label