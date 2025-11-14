import json

def merge_annotations(json1, json2, out_json):

    with open(json1, "r") as f:
        coco1 = json.load(f)
    with open(json2, "r") as f:
        coco2 = json.load(f)

    merged = {
        "images": coco1["images"],
        "annotations": coco1["annotations"],
        "categories": coco1["categories"]
    }

    # get max ids
    max_img_id = max(img["id"] for img in merged["images"])
    max_ann_id = max(ann["id"] for ann in merged["annotations"])

    img_offset = max_img_id + 1
    ann_offset = max_ann_id + 1

    # ---- merge categories (safe by name) ----
    cat_map = {c["name"]: c["id"] for c in merged["categories"]}
    next_cat_id = max(cat_map.values()) + 1

    old_to_new_cat = {}

    for cat in coco2["categories"]:
        name = cat["name"]
        if name in cat_map:
            old_to_new_cat[cat["id"]] = cat_map[name]
        else:
            new_id = next_cat_id
            next_cat_id += 1
            cat_map[name] = new_id
            merged["categories"].append({
                "id": new_id,
                "name": name,
                "supercategory": cat.get("supercategory", "")
            })
            old_to_new_cat[cat["id"]] = new_id

    # ---- merge images ----
    for img in coco2["images"]:
        new_img = img.copy()
        new_img["id"] = img["id"] + img_offset
        merged["images"].append(new_img)

    # ---- merge annotations ----
    for ann in coco2["annotations"]:
        new_ann = ann.copy()
        new_ann["id"] = ann["id"] + ann_offset
        new_ann["image_id"] = ann["image_id"] + img_offset
        new_ann["category_id"] = old_to_new_cat[ann["category_id"]]
        merged["annotations"].append(new_ann)

    # save result
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=4)

    print("Merged JSON saved to:", out_json)


# Example run:
# merge_annotations("A/_annotations.coco.json",
#                   "B/_annotations.coco.json",
#                   "merged_annotations.coco.json")

merge_annotations("../../Datasets/CarDamageBinary/train/damaged/_annotations.coco.json", "../../Datasets/CarPartsandCarDamages/train/_annotations.coco.json", "merged_annotations.coco.json")