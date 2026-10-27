---
title: Tomato Disease Intelligence
emoji: 🍅
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: "6.30.0"
app_file: app_hf.py
pinned: false
license: mit
short_description: Tomato leaf disease detection with CBAM attention (99.38% accuracy)
tags:
  - pytorch
  - computer-vision
  - image-classification
  - gradcam
  - attention-mechanism
  - plant-disease
  - efficientnet
---

# 🍅 Tomato Disease Intelligence

AI-powered tomato leaf disease detection using **EfficientNet-B0** with configurable attention mechanisms (CBAM, SE, Baseline).

## Model Performance

| Variant | Test Accuracy | Val Loss |
|---------|--------------|----------|
| **CBAM** | **99.38%** | 0.0163 |
| SE | 99.07% | 0.0221 |
| Baseline | 98.15% | 0.0418 |

## Dataset

- **PlantVillage** tomato subset: 18,160 images across 10 disease classes
- SHA-256 deduplicated with zero-leakage stratified splits (70/15/15)

## How to Use

1. Upload a tomato leaf photograph
2. Select an attention variant (CBAM recommended)
3. Click **Analyze Leaf** to get an instant diagnosis with Grad-CAM visualization

## Architecture

```
Input (224×224 RGB)
    ↓
EfficientNet-B0 backbone (pretrained on ImageNet)
    ↓
CBAM Dual-Attention Module
  ├── Channel Attention (GAP + GMP → shared MLP → sigmoid)
  └── Spatial Attention (channel pool → 7×7 conv → sigmoid)
    ↓
Global Average Pooling
    ↓
Dense(256) → ReLU → Dropout(0.4) → Dense(10)
    ↓
Softmax → Disease Class
```

## Classes

1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Spider Mites (Two-spotted)
7. Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Healthy

## Citation

PlantVillage Dataset: Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics.
