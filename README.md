# 🍅 Tomato Disease Intelligence — Attention Network for Plant Disease Detection

[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **99.38% test accuracy** on PlantVillage tomato leaf disease classification using EfficientNet-B0 with CBAM dual-attention mechanism.

---

## 🌐 Live Demo

The app is deployed on Hugging Face Spaces — upload any tomato leaf photo to get an instant diagnosis with Grad-CAM visualization.

👉 **[Try it live on Hugging Face Spaces](https://huggingface.co/spaces)**

---

## 📋 Project Overview

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

## 🦠 Disease Classes (10)

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
│   ├── server.py           # FastAPI backend (local use)
│   └── static/index.html   # Custom 3D Three.js frontend
├── app_hf.py               # Gradio app (Hugging Face Spaces)
├── src/
│   ├── config.py           # YAML config loader & validator
│   ├── gradcam.py          # Hook-based Grad-CAM implementation
│   ├── models/
│   │   ├── backbone.py     # EfficientNet-B0 wrapper
│   │   ├── attention.py    # Baseline / SE / CBAM modules
│   │   └── net.py          # TomatoAttentionNet full model
│   ├── data/
│   │   ├── transforms.py   # Config-driven augmentation pipeline
│   │   ├── dataset.py      # Custom PyTorch Dataset
│   │   ├── dataloaders.py  # DataLoader builder
│   │   ├── hashing.py      # SHA-256 deduplication
│   │   └── splitting.py    # Zero-leakage stratified splits
│   ├── train.py            # Two-phase training engine
│   └── evaluate.py         # Test metrics & confusion matrix
├── configs/config.yaml     # Single source of truth for all hyperparameters
├── notebooks/
│   ├── tomato_disease_attention_study.ipynb        # Evaluation & Grad-CAM analysis
│   └── data_preprocessing_and_model_training.ipynb # Full pipeline & training records
├── requirements.txt        # Local environment dependencies
├── requirements_hf.txt     # Hugging Face Spaces dependencies
└── deploy_to_hf.py         # One-shot HF Spaces deployment script
```

---

## ⚙️ Local Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/tomato-attention.git
cd tomato-attention

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate       # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# For GPU (CUDA 12.6):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> **Dataset:** Download the PlantVillage tomato subset and place it in `data/raw/` with the 10 class folder structure.  
> **Checkpoints:** Trained `.pt` files are distributed separately via Hugging Face Spaces (not stored in this GitHub repo due to size).

---

## 🚀 Running Locally

### Web App (FastAPI + Three.js frontend)
```powershell
.\venv\python.exe -m uvicorn app.server:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000
```

### Gradio App (HF Spaces style)
```powershell
.\venv\python.exe app_hf.py
```

### Jupyter Notebooks
```powershell
.\venv\Scripts\jupyter.exe lab
```

---

## 🧪 Training Pipeline

The full training pipeline is documented in [`notebooks/data_preprocessing_and_model_training.ipynb`](notebooks/data_preprocessing_and_model_training.ipynb):

1. **SHA-256 deduplication** — 18,160 images → 16,012 unique content hashes, 1,357 duplicate clusters identified
2. **Zero-leakage stratified splitting** — 70/15/15 train/val/test, no duplicate cluster crosses split boundaries
3. **Two-phase training** — Phase A: frozen backbone warmup (2 epochs) → Phase B: differential LR fine-tuning (38 epochs)
4. **Cosine Annealing** with warmup, Label Smoothing (ε=0.1), AdamW, AMP mixed precision

---

## 📊 Results

Detailed results are in [`notebooks/tomato_disease_attention_study.ipynb`](notebooks/tomato_disease_attention_study.ipynb):

- Per-class precision, recall, F1 for all 3 variants
- Confusion matrices  
- Grad-CAM spatial attention visualizations
- Ablation study comparison

---

## 📦 Deploy to Hugging Face Spaces

```powershell
pip install huggingface_hub
git lfs install
.\venv\python.exe deploy_to_hf.py
```

---

## 📚 Dataset

**PlantVillage** — Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
