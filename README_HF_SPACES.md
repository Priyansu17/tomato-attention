---
title: TomatoAI - Tomato Disease Detection
emoji: 🍅
colorFrom: red
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🍅 TomatoAI — Tomato Leaf Disease Detection

An AI-powered tomato leaf disease classifier using **EfficientNet-B0** with three attention variants:
- **CBAM** (Convolutional Block Attention Module)
- **SE** (Squeeze-and-Excitation)
- **Baseline** (no attention)

## Features
- 10-class disease classification with 99.38% accuracy
- Grad-CAM attention saliency maps
- Clinical Agronomy Dossier with treatment protocols
- 4-tier CLIP-based validation (rejects non-tomato images)
- Farmer Sagar AI assistant

## Dataset
PlantVillage Dataset — 18,160 tomato leaf images

## Tech Stack
FastAPI · PyTorch · EfficientNet-B0 · CBAM · Grad-CAM
