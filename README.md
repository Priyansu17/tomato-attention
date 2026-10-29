# 🍅 Tomato Disease Intelligence — Attention Network for Plant Disease Detection

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **99.38% test accuracy** on PlantVillage tomato leaf disease classification using EfficientNet-B0 with CBAM dual-attention mechanism.

---

## 🌐 Live Demo

Run locally and share via Cloudflare Tunnel:

```powershell
# Start the app
.\venv\python.exe -m uvicorn app.server:app --host 0.0.0.0 --port 8000

# (Optional) Share publicly
.\tools\cloudflared.exe tunnel --url http://localhost:8000
```

Then open: `http://localhost:8000`

---

## 📊 Project Overview

This project investigates the effect of **attention mechanisms** on tomato leaf disease classification using a pretrained EfficientNet-B0 backbone. Three variants are compared:

| Variant | Attention | Test Accuracy | Val Loss |
|---------|-----------|--------------|----------|
| **CBAM** | Channel + Spatial (dual) | **99.38%** | 0.0163 |
| SE | Channel only | 99.07% | 0.0221 |
| Baseline | None | 98.15% | 0.0418 |

### Architecture

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
10-class Disease Output
```

---

## 🌿 Disease Classes (10)

| # | Class | Severity |
|---|-------|---------|
| 0 | Bacterial Spot | 🔴 High |
| 1 | Early Blight | 🟡 Medium |
| 2 | Late Blight | 🔴 High |
| 3 | Leaf Mold | 🟡 Medium |
| 4 | Septoria Leaf Spot | 🟡 Medium |
| 5 | Spider Mites (Two-spotted) | 🟡 Medium |
| 6 | Target Spot | 🟡 Medium |
| 7 | Yellow Leaf Curl Virus | 🔴 High |
| 8 | Tomato Mosaic Virus | 🟡 Medium |
| 9 | Healthy | 🟢 None |

---

## 🗂️ Repository Structure

```
tomato-attention/
├── app/
│   ├── server.py           # FastAPI backend
│   └── static/index.html   # Custom 3D Three.js frontend (all CSS+JS inline)
├── src/
│   ├── config.py           # YAML config loader & validator
│   ├── gradcam.py          # Hook-based Grad-CAM implementation
│   ├── models/
│   │   ├── backbone.py     # EfficientNet-B0 wrapper
│   │   ├── attention.py    # Baseline / SE / CBAM modules
│   │   └── net.py          # TomatoAttentionNet full model
│   └── data/
│       ├── transforms.py   # Config-driven augmentation pipeline
│       ├── dataset.py      # Custom PyTorch Dataset
│       ├── dataloaders.py  # DataLoader builder
│       ├── hashing.py      # SHA-256 deduplication
│       └── splitting.py    # Zero-leakage stratified splits
├── checkpoints/
│   ├── best_model_cbam.pt      # CBAM variant weights (52 MB)
│   ├── best_model_se.pt        # SE variant weights (52 MB)
│   └── best_model_baseline.pt  # Baseline variant weights (50 MB)
├── configs/config.yaml     # Single source of truth for all hyperparameters
├── notebooks/
│   ├── tomato_disease_attention_study.ipynb        # Evaluation & Grad-CAM analysis
│   └── data_preprocessing_and_model_training.ipynb # Full pipeline & training records
├── Dockerfile              # Container build (for future cloud deployment)
├── requirements.txt        # Local environment dependencies
└── requirements_hf.txt     # Minimal inference-only dependencies
```

---

## ⚙️ Local Setup

```bash
# Clone
git clone https://github.com/Priyansu17/tomato-attention.git
cd tomato-attention

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate       # Windows
# source venv/bin/activate    # Linux/Mac

# Install CPU PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **Dataset:** Download the PlantVillage tomato subset and place in `data/raw/` with 10 class folders.
> **Checkpoints:** Model weights (`best_model_*.pt`) are included in this repo under `checkpoints/`.

---

## 🚀 Running the App

```powershell
.\venv\python.exe -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` — full 3D web interface with:
- 🎯 CBAM/SE/Baseline variant selector
- 🗺️ Grad-CAM attention saliency maps
- 📋 Clinical Agronomy Dossier (4 tabs: Overview, Symptoms, Treatment, Video)
- 🤖 Farmer Sagar AI assistant
- 📱 Fully mobile-responsive

---

## 🏋️ Training Pipeline

Full pipeline documented in [`notebooks/data_preprocessing_and_model_training.ipynb`](notebooks/data_preprocessing_and_model_training.ipynb):

1. **SHA-256 deduplication** — 18,160 images → 16,012 unique hashes, 1,357 duplicate clusters
2. **Zero-leakage stratified splitting** — 70/15/15 train/val/test, no duplicate cluster crosses splits
3. **Two-phase training** — Phase A: frozen backbone warmup (2 epochs) → Phase B: differential LR fine-tuning (38 epochs)
4. **Cosine Annealing** with warmup, Label Smoothing (λ=0.1), AdamW, AMP mixed precision

---

## 📈 Results

Detailed in [`notebooks/tomato_disease_attention_study.ipynb`](notebooks/tomato_disease_attention_study.ipynb):

- Per-class precision, recall, F1 for all 3 variants
- Confusion matrices
- Grad-CAM spatial attention visualizations
- Ablation study comparison

---

## 📚 Dataset

**PlantVillage** — Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
