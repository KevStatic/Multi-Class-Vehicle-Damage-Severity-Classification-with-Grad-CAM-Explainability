# 🚗 Multi-Class Vehicle Damage Severity Classification with Grad-CAM Explainability

This repository contains an **explainable deep learning-based vehicle damage severity classification system** that categorizes input images into **three severity levels**:
- **Minor Damage**
- **Moderate Damage**
- **Severe Damage**

The project uses **MobileNetV2 (Transfer Learning)** for classification and **Grad-CAM** for visual explanation to highlight the regions that influenced the model’s decision — improving **trust**, **interpretability**, and **real-world usability**.

---

## 📌 Key Features

✔ Multi-class vehicle damage severity classification  
✔ Lightweight model — CPU friendly (MobileNetV2)  
✔ Grad-CAM based explainability & heatmap visualization  
✔ Clean modular pipeline (train → evaluate → infer → explain)  
✔ Works with custom user-uploaded images  

---

## 🏋️ Training the Model

```bash
python src/train.py
```
This will:
- Load training & validation datasets

- Train MobileNetV2 with custom classifier head

- Save best model weights in results/

- Generate loss & accuracy plots

---

## 📈 Model Evaluation

```bash 
python src/evaluation.py
```
This script provides:
- Training & validation accuracy curves

- Training & validation loss curves

- Confusion matrix

- Classification report

- Results saved inside ```results/plots/```

---

## 🔍 Run Inference + Grad-CAM Visualization

```bash
python src/inference.py --image path/to/your_image.jpg
```
This will output:
- Predicted severity class

- Grad-CAM heatmap visualization

- Stored under: ```results/gradcam/```

Example:

```bash
python src/inference.py --image samples/car1.jpg
```

---

## 🧠 Model Specification

- Backbone: MobileNetV2 (ImageNet pretrained)

- Final Head: Dense → Dropout (0.3) → Dense (3) → Softmax

- Optimizer: Adam

- Loss Function: Cross-Entropy

- Explainability: Grad-CAM

---

## 📊 Results Summary

| Metric              | Score      |
| ------------------- | ---------- |
| Training Accuracy   | 69.20%     |
| Validation Accuracy | 63.71%     |
| Test Accuracy       | **70.37%** |

Grad-CAM visualizations confirm that the network focuses correctly on damaged regions, improving trustworthiness.

---

## 🚀 Future Improvements

- Add No Damage class

- Balance dataset (Moderate class)

- Test advanced models (EfficientNet, ViT)

- Add damage localization (YOLO / Mask R-CNN)

- Build web or mobile user interface

- Explore cost estimation model

## 📜 License

This project is for academic and research purposes.
Please cite if used or referenced.
