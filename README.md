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

## ⚙️ Setup & Run — Complete Tutorial

### Prerequisites

Make sure you have these installed before starting:

| Tool | Version | Download |
|------|---------|---------|
| **Python** | 3.11 recommended | [python.org](https://www.python.org/downloads/) |
| **Git** | Any recent version | [git-scm.com](https://git-scm.com/) |
| **RAM** | 4 GB minimum | — |

> **GPU is NOT required.** The app runs on CPU. A GPU will speed up inference but is optional.

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Priyansu17/tomato-attention.git
cd tomato-attention
```

---

### Step 2 — Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux / Mac:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

---

### Step 3 — Install Dependencies

**CPU only (recommended for most users):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**GPU (CUDA 12.6, optional):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> ⏳ PyTorch is large (~200 MB for CPU build). This step may take 3–5 minutes depending on your internet speed.

---

### Step 4 — Verify Checkpoints Exist

The trained model weights are included in this repo. Verify they're present:

```bash
# You should see 3 files listed
ls checkpoints/best_model_*.pt          # Linux/Mac
dir checkpoints\best_model_*.pt         # Windows
```

Expected output:
```
best_model_baseline.pt   (~50 MB)
best_model_cbam.pt       (~52 MB)
best_model_se.pt         (~52 MB)
```

> If they are missing, re-clone the repo. These files are committed directly to git.

---

### Step 5 — Run the App

**Windows:**
```powershell
.\venv\python.exe -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

**Linux / Mac:**
```bash
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

### Step 6 — Open the App

Open your browser and go to:
```
http://localhost:8000
```

You'll see the full 3D web interface with:
- 🎯 CBAM / SE / Baseline attention variant selector
- 🗺️ Grad-CAM attention saliency heatmaps
- 📋 Clinical Agronomy Dossier (4 tabs: Overview, Symptoms, Treatment, Video Guide)
- 🤖 Farmer Sagar AI assistant
- 📱 Fully mobile-responsive layout

Upload any tomato leaf photo and click **Analyze** to get a diagnosis!

---

### (Optional) Share Publicly via Cloudflare Tunnel

To generate a public URL anyone can access (no deployment needed):

**Windows:**
```powershell
# Download cloudflared (one-time)
Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "cloudflared.exe"

# Run tunnel (keep app running in another terminal)
.\cloudflared.exe tunnel --url http://localhost:8000
```

**Linux / Mac:**
```bash
# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared

# Run tunnel
./cloudflared tunnel --url http://localhost:8000
```

Copy the `https://xxxx.trycloudflare.com` URL from the output and share it. ✅

> ⚠️ The tunnel URL expires after ~24 hours. Just re-run the command to get a new one.

---

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| `ModuleNotFoundError: No module named 'fastapi'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: checkpoints/best_model_cbam.pt` | Re-clone the repo — checkpoints must be downloaded |
| Port 8000 already in use | Change port: `--port 8080` and open `http://localhost:8080` |
| App loads but prediction fails | Check terminal for error messages; ensure venv is activated |
| Slow inference (~3-5 seconds) | Normal on CPU — use GPU build for faster results |

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
