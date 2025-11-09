import os
import random
import shutil

# === CONFIGURATION ===
train_damaged_dir = "../Datasets/CarDamageBinary/train/damaged"
valid_damaged_dir = "../Datasets/CarDamageBinary/valid/damaged"
split_ratio = 0.2  # 20%

# === SETUP ===
os.makedirs(valid_damaged_dir, exist_ok=True)

# Get list of all image files
all_images = [f for f in os.listdir(train_damaged_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle for randomness
random.shuffle(all_images)

# Pick 20% for validation
num_to_move = int(len(all_images) * split_ratio)
val_images = all_images[:num_to_move]

# === MOVE IMAGES ===
for img in val_images:
    src = os.path.join(train_damaged_dir, img)
    dst = os.path.join(valid_damaged_dir, img)
    shutil.move(src, dst)

print(f"✅ Moved {len(val_images)} images ({split_ratio*100:.0f}%) "
      f"from 'train/damaged' → 'valid/damaged'")
print(f"📂 Remaining in train: {len(os.listdir(train_damaged_dir))}")
print(f"📂 Now in valid: {len(os.listdir(valid_damaged_dir))}")
