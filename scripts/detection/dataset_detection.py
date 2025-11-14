import os
import json
import torch
from PIL import Image

class CarDamageDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms

        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        # Map image_id → list of annotations
        self.img_to_anns = {img["id"]: [] for img in self.images}
        for ann in self.annotations:
            self.img_to_anns[ann["image_id"]].append(ann)

        # Class mapping:
        # 0 = background
        # 1 = damaged
        self.num_classes = 2  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        # Get boxes for this image
        anns = self.img_to_anns[img_info["id"]]
        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)   # damaged class

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info["id"]]),
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
