# create_test_split.py

import os, random, shutil

source_dir = "../../data/validation"   # we will split from validation
dest_dir = "../../data/test"
classes = ["Minor_Damage", "Moderate_Damage", "Severe_Damage"]

test_ratio = 0.33  # takes ~ 1/3 images from validation to build test

os.makedirs(dest_dir, exist_ok=True)

for cls in classes:
    src_cls = os.path.join(source_dir, cls)
    dst_cls = os.path.join(dest_dir, cls)
    os.makedirs(dst_cls, exist_ok=True)

    images = os.listdir(src_cls)
    random.shuffle(images)
    test_count = int(len(images) * test_ratio)

    test_imgs = images[:test_count]

    for img in test_imgs:
        shutil.copy2(os.path.join(src_cls, img), os.path.join(dst_cls, img))

    print(f"{cls}: {test_count} images copied to test set")
