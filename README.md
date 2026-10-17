# Tomato Leaf Disease Detection using an Attention Network

**Project complete.** All phases finished: data pipeline, training (3 attention variants),
evaluation, ablation study, Grad-CAM explainability, and Streamlit demo.

> ✅ **Status**
> - Training complete: baseline, SE-attention, and CBAM variants trained on RTX 3050 GPU.
> - Best result: **CBAM — 99.38% accuracy, 99.20% macro F1** on the held-out test set.
> - Grad-CAM overlays generated for all three variants.
> - Streamlit demo app ready (`app/app.py`).
> - The dataset is **not included in the repository** — supply the PlantVillage tomato
>   subset manually under `data/raw/` as described below.

---

## Project Overview

This project studies the effect of **attention modules** on tomato leaf disease
classification. A pretrained CNN backbone extracts features from 224x224 RGB leaf images,
a configurable attention module refines those features, and a small dense head produces a
10-class prediction with a confidence score.

The intended contribution is an **ablation study** comparing:

1. `baseline` — no attention,
2. `se` — squeeze-and-excitation channel attention,
3. `cbam` — convolutional block attention module (channel **and** spatial attention).

Everything is driven from a single configuration file (`configs/config.yaml`).

### Phase roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Project scaffolding: directories, `.clinerules`, `.gitignore`, `requirements.txt`, `configs/config.yaml`, README skeleton | ✅ complete |
| 2 | Data pipeline: hashing/dedup, stratified split, datasets, transforms, DataLoaders (gated sub-phases 2A/2B/2C) | ✅ complete |
| 3 | Models: backbone wrapper, attention modules (baseline / SE / CBAM), classifier head (gated sub-phase 2D) | ✅ complete |
| 4 | Training + validation loop, checkpointing, early stopping, all 3 variants | ✅ complete |
| 5 | Evaluation: metrics, confusion matrix, classification report (test set) | ✅ complete |
| 6 | Ablation study across attention variants | ✅ complete |
| 7 | Grad-CAM explainability | ✅ complete |
| 8 | Streamlit demo application | ✅ complete |

---

## Architecture

Locked architecture (do not change without agreement):

```mermaid
flowchart TD
    A["RGB Tomato Leaf Image (224×224)"] --> B["ImageNet Normalization"]
    B --> C["EfficientNet-B0 Backbone (pretrained)"]
    C --> D{"Attention Module"}
    D -->|"baseline"| E1["Identity (no-op)"]
    D -->|"se"| E2["SE: GlobalAvgPool → FC → ReLU → FC → Sigmoid → Scale"]
    D -->|"cbam"| E3["CBAM: Channel Attn → Spatial Attn (7×7 conv)"]
    E1 --> F["Global Average Pooling"]
    E2 --> F
    E3 --> F
    F --> G["Dense(256) + ReLU"]
    G --> H["Dropout(0.4)"]
    H --> I["Dense(10) → Softmax"]
    I --> J["Predicted Class + Confidence"]
```

The attention module is inserted **after the last backbone feature block**; the insertion
stage is configurable.

---

## CBAM

CBAM (Convolutional Block Attention Module) is applied **after** the last backbone feature
block and is composed of two sequential sub-modules:

**1. Channel attention (first)**

```
F  -->  GlobalAveragePool ---------------------> shared MLP (reduction 16) ----\
                                                                               (+) --> sigmoid --> Mc
F  -->  GlobalMaxPool -------------------------> shared MLP (reduction 16) ----/
```

- Global average pooling **and** global max pooling.
- A **shared MLP** with **reduction ratio 16** (hidden size = C / 16).
- The two descriptors are summed, passed through a sigmoid, and used to re-weight the
  input feature map channel-wise.

**2. Spatial attention (second)**

```
F' --> channel-wise AveragePool ---\
                                    concat --> 7x7 Conv --> sigmoid --> Ms
F' --> channel-wise MaxPool -------/
```

- Channel-wise average pooling and max pooling, concatenated into a 2-channel map.
- A single **7x7 convolution** produces the spatial attention map.
- The map is sigmoid-activated and multiplies the channel-refined features.

Configuration keys: `attention.type`, `attention.insertion_stage`,
`attention.cbam.channel.reduction_ratio` (16),
`attention.cbam.spatial.kernel_size` (7).

Implementation status: **complete** (Phase 2D / 3).

---

## Dataset

- **PlantVillage tomato subset**, 10 classes.
- Class folder names expected by default:

| label id | # | Class folder |
|---|---|---|
| 0 | 1 | `Tomato___Bacterial_spot` |
| 1 | 2 | `Tomato___Early_blight` |
| 2 | 3 | `Tomato___Late_blight` |
| 3 | 4 | `Tomato___Leaf_Mold` |
| 4 | 5 | `Tomato___Septoria_leaf_spot` |
| 5 | 6 | `Tomato___Spider_mites Two-spotted_spider_mite` |
| 6 | 7 | `Tomato___Target_Spot` |
| 7 | 8 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` |
| 8 | 9 | `Tomato___Tomato_mosaic_virus` |
| 9 | 10 | `Tomato___healthy` |

Label ids (used by the model, checkpoints and reports) come **only** from the row order of
`data.class_names` above - never from filesystem, alphabetical or tool-specific sort order.

- The dataset is supplied **manually by the user**. The code will never download it.
- `data/raw/` is treated as **strictly read-only**: nothing is written, re-labelled or
  modified there.
- **Local status:** all 10 class folders are present under `data/raw/` (verified against
  `data.class_names`); the stratified split itself is created in memory in Phase 2.

**Split:** stratified 0.70 / 0.15 / 0.15 (train / validation / test), drive purely by
`data.split` in the config.

**Duplicate handling:** before splitting, exact image duplicates are detected by content
hashing (SHA-256) and grouped so that duplicate images never land in different splits.

> **Warning:** exact hashing only detects **byte-identical** images. It does **not** detect
> every possible near-duplicate — for example re-crops, re-compression, resized copies,
> mild colour/brightness edits, or augmented variants of the same source image can still be
> identical in content while hashing differently. Reported metrics should therefore be read
> as optimistic upper bounds.

---

## Dataset Directory Structure

Expected layout (the exact class-folder names must match `data.class_names`):

```
tomato-attention/
├── data/
│   └── raw/                                  # user-supplied, READ-ONLY, git-ignored
│       ├── Tomato___Bacterial_spot/
│       │   ├── <image files>.JPG
│       │   └── ...
│       ├── Tomato___Early_blight/
│       │   └── ...
│       ├── Tomato___Late_blight/
│       ├── Tomato___Leaf_Mold/
│       ├── Tomato___Septoria_leaf_spot/
│       ├── Tomato___Spider_mites Two-spotted_spider_mite/
│       ├── Tomato___Target_Spot/
│       ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│       ├── Tomato___Tomato_mosaic_virus/
│       └── Tomato___healthy/
```

Notes:

- Both layouts are supported (the loader discovers class folders either way):
  - flat: `data/raw/<class_name>/<image>`
  - split-aware: `data/raw/<split>/<class_name>/<image>` (e.g. an existing `train/` folder)
- Splits are created **in memory** from the flat layout, so the raw folder is never
  re-organised into `train/`, `val/` or `test/` sub-directories.
- Supported image extensions: `.jpg`, `.jpeg`, `.png` (case-insensitive).

### Full project layout (Phase 1)

```
tomato-attention/
├── .clinerules                 # project-wide development rules
├── .gitignore                  # data/, checkpoints/, venv/, caches ignored
├── requirements.txt            # core dependencies (portable, PyTorch-only)
├── README.md                   # this file
├── .vscode/
│   └── settings.json           # pins the interpreter to .\venv\python.exe
├── configs/
│   └── config.yaml             # single source of truth for hyperparameters
├── data/
│   └── raw/                    # user-supplied PlantVillage dataset (READ-ONLY)
├── checkpoints/                # saved model weights (git-ignored)
├── notebooks/                  # Jupyter notebooks (same .\venv kernel)
├── src/
│   ├── __init__.py
│   └── models/
│       └── __init__.py         # backbone / attention / net modules go here later
├── app/                        # Streamlit demo goes here later
└── tests/
    ├── .gitkeep
    └── test_phase1_scaffolding.py   # Phase 1 acceptance checks (pytest)
```

---

## Environment Setup

This project uses the **existing Conda environment at `.\venv`** (Python 3.11.16).
Do **not** create or replace a virtual environment.

### Mandatory invocation rules

| Situation | Command |
|-----------|---------|
| Run any Python command | `.\venv\python.exe <args>` — always explicit |
| Install packages | `.\venv\python.exe -m pip install ...` |
| Verify the interpreter before Python work | `.\venv\python.exe -c "import sys; print(sys.executable); print(sys.version)"` |

- ❌ Do **not** use the base Conda Python (`python` / `python.exe` from `(base)`).
- ❌ Do **not** use a system Python installation (`py`, `python3`, `C:\Python*`, Microsoft
  Store Python).
- ❌ Do **not** create another virtual environment.
- ✅ Always confirm that `.\venv\python.exe` reports **Python 3.11.x** before Python-related
  verification.

Verified interpreter for this repository:

```
sys.executable  ->  c:\Users\sharm\Desktop\tomato-attention\venv\python.exe
sys.version     ->  3.11.16 | packaged by Anaconda, Inc. | (MSC v.1942 64 bit (AMD64))
sys.prefix      ->  c:\Users\sharm\Desktop\tomato-attention\venv
```

```powershell
# Confirm the interpreter (run this before any Python verification)
.\venv\python.exe -c "import sys; print(sys.executable); print(sys.version)"

# This Conda path environment has NO .\venv\Scripts\activate script, so activation is
# not required (and not used by this project): call .\venv\python.exe explicitly instead.
```

For VS Code:

1. Select the interpreter `.\venv\python.exe` (Command Palette → "Python: Select
   Interpreter").
2. Register the Jupyter kernel once so notebooks use the same environment:

```powershell
.\venv\python.exe -m ipykernel install --user --name tomato-attention --display-name "Tomato Attention (venv)"
```

Then pick the **Tomato Attention (venv)** kernel inside any notebook in `notebooks/`.

---

## Installation

Dependencies are declared in `requirements.txt` — install from there, do not install
packages ad hoc.

```powershell
# 1) (Recommended) install the PyTorch build that matches your hardware first
#    CPU only:
.\venv\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#    CUDA (pick the tag matching your driver, e.g. cu126):
# .\venv\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2) then install everything else
.\venv\python.exe -m pip install -r requirements.txt
```

> **PyTorch note:** the correct PyTorch/torchvision wheel depends on your **CUDA vs CPU**
> setup, so `requirements.txt` deliberately does **not** pin a CUDA-specific build and does
> **not** pin exact versions. TensorFlow/Keras are intentionally not used — this project is
> **PyTorch-only**.

---

## Configuration

All hyperparameters live in **`configs/config.yaml`**. Python sources must never hardcode
them.

| Section | Purpose |
|---------|---------|
| `project` | run name / description / phase metadata |
| `seed` | fixed random seed and determinism flags |
| `data` | dataset root, image size, class list, split ratios, duplicate check, DataLoader |
| `model` | backbone, pretrained flag, GAP, Dense(256) + ReLU head, dropout 0.4, ImageNet normalization |
| `attention` | attention type (`baseline`/`se`/`cbam`), insertion stage, CBAM/SE sub-options |
| `training` | batch size, max epochs, loss + label smoothing, freeze/unfreeze, early stopping, checkpointing |
| `optimizer` | AdamW, LR 1e-4, backbone fine-tuning LR 1e-5, weight decay 1e-4 |
| `scheduler` | cosine annealing / step / plateau, warmup |
| `augmentation` | train transforms and eval resize/center-crop |
| `evaluation` | metrics, confusion matrix, classification report |
| `gradcam` | target layer, colormap, overlay alpha, sample count |
| `paths` | output directories for checkpoints, results, logs, figures, Grad-CAM |
| `smoke_test` | tiny sanity run: 1 epoch, small subset, `num_workers: 0` |

Key planned defaults: `image_size: 224`, `num_classes: 10`, `batch_size: 32`,
`max_epochs: 40`, `learning_rate: 1e-4`, `weight_decay: 1e-4`, `dropout: 0.4`,
`label_smoothing: 0.1`, CBAM reduction ratio `16`, CBAM spatial kernel `7`,
`backbone: efficientnet_b0`, `attention: cbam`, `seed: 42`,
split `0.70 / 0.15 / 0.15`, smoke test `enabled: true`.

Implementation status of the config loader: **complete** (Phase 1 / 2).

---

## Data Preparation

1. Download the **PlantVillage tomato subset** yourself (this project never downloads data).
2. Place the 10 class folders under `data/raw/` using the structure shown above.
3. Make sure the folder names match `data.class_names` in `configs/config.yaml` **exactly**
   (including capitalisation and spaces) — or update the config to match your folder names.
4. Do not write anything into `data/raw/`: the pipeline only reads from it. Splits are
   computed in memory.
5. Nothing else is required — no manual resizing and no manual train/val/test folders.

Implementation status of the data pipeline: **complete** (Phases 2A / 2B / 2C).

---

## Training

Planned behaviour (Phase 4 — **not implemented yet**):

- Loss: cross-entropy with **label smoothing 0.1**.
- Optimizer: **AdamW**, LR `1e-4`, weight decay `1e-4`.
- Two-stage fine-tuning controlled by `training.freeze_backbone`:
  - Stage A (first `epochs: 5`): backbone frozen, head trained.
  - Stage B: backbone unfrozen and fine-tuned with a **lower LR** (`1e-5`).
- Scheduler: cosine annealing with warmup (`scheduler.warmup_epochs: 3`).
- Max epochs: `40`, batch size `32`, mixed precision on CUDA.
- Early stopping on `val_loss` (patience 7, min delta 0.0005).
- Best checkpoint saved to `checkpoints/best_model.pt` (git-ignored).

Planned commands (available once Phase 4 lands):

```powershell
# Smoke test only: tiny subset, at most 1 epoch
.\venv\python.exe -m src.train --config configs/config.yaml --smoke-test

# Full training (only with explicit user approval; never run automatically)
.\venv\python.exe -m src.train --config configs/config.yaml
```

> **Rules:** smoke tests use a tiny subset and at most **1 epoch**. Full training is
> **never** executed automatically — explicit user permission is required first.

---

## Evaluation

Planned behaviour (Phase 5 — **not implemented yet**):

- Evaluate the best checkpoint on the held-out **test** split (`evaluation.split`).
- Metrics: `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`,
  `roc_auc_ovr_macro`.
- Artifacts: normalised **confusion matrix**, **classification report**, and per-sample
  predictions saved under `results/` (git-ignored).

Planned command:

```powershell
.\venv\python.exe -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best_model.pt
```

---

## Ablation Study

Controlled comparison of attention variants using identical data, splits, seeds and
training settings — only `attention.type` changes:

| Run | `attention.type` | Description |
|-----|------------------|-------------|
| 1 | `baseline` | no attention (identity) |
| 2 | `se` | squeeze-and-excitation channel attention |
| 3 | `cbam` | channel **+** spatial attention (default) |

Planned: each run uses the same `seed.value`, the same duplicate-deduplicated stratified
split, and the same epoch budget; results are collected into a single comparison table and
figure. `attention.insertion_stage` can be varied as a secondary ablation.

Implementation status: **not implemented yet** (Phase 6).

---

## Grad-CAM

Planned behaviour (Phase 7 — **not implemented yet**):

- Gradient-weighted class activation maps on the last backbone feature stage
  (`gradcam.target_layer: auto`).
- Overlays use the `jet` colormap with alpha `0.5`; `num_samples` images per grid.
- Figures are written to `results/gradcam/` (git-ignored).

Planned command:

```powershell
.\venv\python.exe -m src.gradcam --config configs/config.yaml --checkpoint checkpoints/best_model.pt
```

---

## Streamlit Demo

Planned behaviour (Phase 8 — **not implemented yet**):

- Upload a tomato leaf image, run inference with the trained checkpoint, and display the
  predicted class, the confidence score and a Grad-CAM overlay.

Planned command:

```powershell
.\venv\python.exe -m streamlit run app/app.py
```

---

## Testing

Tests live in `tests/` (pytest) and must always be run with the project environment.

**Implemented now (Phase 1):** `tests/test_phase1_scaffolding.py` covers the directory layout,
`.clinerules` contents, `requirements.txt` dependencies (PyTorch-only), `.gitignore` semantics,
PyYAML parsing plus every important config default, the read-only `data/raw/` contract, the
class-folder match against `data.class_names`, the absence of out-of-scope implementation
files, and the must-run-on-venv rule.

**Planned (later phases):** dataset discovery and duplicate hashing, split ratios /
straddling, attention module tensor shapes (channel + spatial), and a forward-pass shape
check for the full network.

```powershell
.\venv\python.exe -m pytest tests -q
```

---

## Phase 1 Verification (scaffolding checks)

Phase 1 acceptance checks and their verified outcomes are recorded here:

| Check | Result |
|-------|--------|
| A. Project directories exist (`configs/`, `data/raw/`, `checkpoints/`, `notebooks/`, `src/`, `src/models/`, `app/`, `tests/`) | ✅ |
| B. `.clinerules` exists and contains all required development rules | ✅ |
| C. `requirements.txt` exists with the required core dependencies, PyTorch-only | ✅ |
| D. `.gitignore` excludes `venv/`, `.venv/`, `data/`, `checkpoints/` | ✅ |
| E. `configs/config.yaml` parses with PyYAML | ✅ |
| F. YAML contains all required sections and key defaults | ✅ |
| G. `.\venv\python.exe` is Python 3.11.x | ✅ (3.11.16) |
| H. No additional virtual environment created | ✅ |
| I. No dataset downloaded by the project; dataset supplied manually | ✅ (user placed the PlantVillage folders in `data/raw/`; nothing is written there) |
| I2. Supplied class folders match `data.class_names` | ✅ 10/10 folders match exactly |
| J. No full training performed | ✅ |
| K. Dependency check was lightweight/import-oriented only | ✅ |
| M. Tests run on `.\venv\python.exe` (project venv), never base/system Python | ✅ enforced by `test_running_on_project_venv_interpreter` |

Re-run **all** Phase 1 acceptance checks at any time — always with the venv interpreter:

```powershell
.\venv\python.exe -m pytest tests -q
```

The suite parses `configs/config.yaml` with PyYAML itself, and it **fails on purpose** if it
is executed with the base Conda Python or a system Python installation.

---

## Results

All three variants were trained on RTX 3050 (CUDA 12.6), seed 42, EfficientNet-B0 backbone,
70/15/15 stratified split, 18,160 total images, up to 40 epochs with early stopping (patience 7).

### Ablation — Test-Set Metrics

| Variant | Accuracy | F1 Macro | Precision | Recall | ms/img |
|---------|----------|----------|-----------|--------|--------|
| baseline | 0.9923 | 0.9894 | 0.9910 | 0.9880 | 10.88 |
| se       | 0.9934 | 0.9915 | 0.9924 | 0.9908 |  7.05 |
| **cbam** | **0.9938** | **0.9920** | **0.9928** | **0.9913** |  7.56 |

### Training Convergence

| Variant | Best Epoch | Total Epochs | Val Acc (best) | Val Loss (best) |
|---------|------------|--------------|----------------|-----------------|
| baseline | 40 | 40 | 0.9930 | 0.5466 |
| se       | 40 | 40 | 0.9927 | 0.5467 |
| cbam     | 35 | 38 | 0.9919 | 0.5400 |

Full ablation table: [`outputs/ablation.md`](outputs/ablation.md) · [`outputs/ablation.csv`](outputs/ablation.csv)

### Result Artifacts

| Artifact | Location | Status |
|----------|----------|--------|
| Trained checkpoints | `checkpoints/best_model_{baseline,se,cbam}.pt` | ✅ real weights |
| Test metrics | `results/metrics_test_{variant}.json` | ✅ 3 files |
| Confusion matrices | `results/confusion_matrix_test_{variant}.png` | ✅ 3 files |
| Ablation table | `outputs/ablation.csv` + `outputs/ablation.md` | ✅ |
| Grad-CAM grids | `results/gradcam/{variant}/gradcam_val_grid.png` | ✅ 3 grids |

---

## Limitations

- **Near-duplicate leakage is not fully excluded.** The planned duplicate check uses exact
  content hashing (SHA-256); byte-identical images are grouped and kept in a single split,
  but re-crops, re-compression, resizes and mild colour/augmentation edits hash differently
  and can still leak across splits. Reported metrics should be treated as optimistic.
- **Single dataset.** PlantVillage is largely laboratory/controlled imagery; performance on
  real field images with cluttered backgrounds and variable lighting is expected to be lower
  and is not measured here.
- **Single seed per configuration** in the planned experiments, so run-to-run variance is not
  quantified.
- **Class imbalance.** PlantVillage tomato classes are not equally sized; only macro-averaged
  metrics (plus per-class reports) are planned rather than a balanced-sampling strategy.
- **No hyperparameter search.** The configured schedule/LR are reasonable defaults, not
  tuned results.
- **Explainability is qualitative.** Grad-CAM overlays support visual inspection but are not a
  quantitative validation of model reasoning.

---

## Future Work

- Near-duplicate detection (perceptual hashing / embedding similarity) in addition to exact
  hashing, and a stricter group-aware split.
- Additional backbones (ResNet50, MobileNetV3) and attention variants/configurations as a
  wider ablation, driven by `model.backbone` and `attention.type`.
- Multi-seed runs with mean ± standard deviation reporting, and confidence intervals.
- Cross-dataset evaluation on field imagery, plus domain-shift/robustness checks under
  blur, noise, JPEG artefacts and lighting changes.
- Class-imbalance handling (weighted sampling/loss) and test-time augmentation.
- Quantitative explainability checks (e.g. deletion/insertion curves) alongside Grad-CAM.
- Export the trained model for deployment and harden the Streamlit demo (batching, caching,
  error handling).

---

## Repository Rules

The full project-wide development rules live in **`.clinerules`**. Summary:

- Python 3.10+ (this project uses 3.11) and the existing Conda environment at `.\venv` only.
- PyTorch + torchvision, scikit-learn, PyYAML, Streamlit — **PyTorch-only, no TensorFlow**.
- No hardcoded hyperparameters: everything lives in `configs/config.yaml`.
- Type hints and docstrings; reproducible seeds.
- Datasets are never downloaded, never modified — `data/raw/` is read-only.
- `data/` and `checkpoints/` stay out of Git.
- Full training is never run automatically; smoke tests use a tiny subset and ≤ 1 epoch.
- Work proceeds in small phases; the next phase is only started on request.




