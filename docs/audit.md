# Audit Report — Tomato Attention Project Takeover
**Date:** 2026-10-15  
**Auditor:** Antigravity (takeover from Cline)  
**Interpreter:** `.\venv\python.exe` → Python 3.11.16 (Anaconda, CUDA build)

---

## 1. Git Log

```
a42f9b4  Phase 2C: build deterministic PyTorch data pipeline
5613aaa  Phase 2B: hashing duplicate groups and deterministic split
1fe5a84  Phase 2A: dataset discovery, validation and manifest
eaba0de  Phase 1: finalize dataset scaffolding
da0bafc  Phase 1: project scaffolding
```

5 commits. All by Cline. No snapshot commit yet — to be created by the user per the resume instructions.

---

## 2. Component Status Table

| Component | Status | Notes |
|---|---|---|
| **Project structure** | ✅ Done & verified | All dirs present: `app/`, `checkpoints/`, `configs/`, `data/raw/`, `notebooks/`, `results/`, `src/`, `tests/` |
| **`requirements.txt`** | ⚠️ Exists — partial install | File is correct. `torch 2.14.1+cu126`, `torchvision`, `numpy`, `Pillow`, `PyYAML`, `pytest` are installed. **Missing from venv:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `streamlit` |
| **`.venv` / `venv/`** | ✅ Done & verified | `venv/` at project root, Python 3.11.16, CUDA 12.6 build (torch+cu126) |
| **`configs/config.yaml`** | ✅ Done & verified | All required sections and keys present, passes `validate_config()` |
| **`data/raw` check** | ✅ Done & verified | 10 class folders, non-empty. Total 18,160 images (verified via `split_summary.json`) |
| **`src/dataset.py`** (equiv: `src/data/dataset.py`) | ✅ Done & verified | `LeafImageDataset`, `load_split_rows`, `limit_rows` — all tested |
| **`src/models/cbam.py`** (equiv: `src/models/attention.py`) | ✅ Done & verified | `CBAM`, `SEBlock`, `BaselineAttention`, `build_attention` — all tested, spec correct |
| **`src/models/net.py`** | ✅ Done & verified | `TomatoAttentionNet`, `build_model` — full architecture, tested |
| **`src/train.py`** | ❌ Missing | Not yet implemented |
| **`src/evaluate.py`** | ❌ Missing | Not yet implemented |
| **`src/gradcam.py`** | ❌ Missing | Not yet implemented |
| **`app/app.py`** | ❌ Missing | Only `.gitkeep` exists in `app/` |
| **`tests/`** | ✅ Done & verified | 141 tests, 141 passed, 0 failures (run 2026-10-15) |
| **`README.md`** | ⚠️ Partial | Accurate for implemented phases; no Mermaid diagram, Results section is placeholder, some stale text (CBAM section still says "not implemented yet" though it is) |

### Additional files present (beyond the spec):
- `src/data/discovery.py`, `src/data/hashing.py`, `src/data/splitting.py`, `src/data/transforms.py`, `src/data/dataloaders.py` — all implemented and tested
- `src/config.py`, `src/utils/seed.py`, `src/utils/logging_utils.py`, `src/utils/paths.py` — all implemented and tested
- `results/manifests/` — `hashes.csv`, `splits.csv`, `split_summary.json`, `dataset_index.csv` — all present, splits computed

---

## 3. Live Run Results

### 3.1 Python interpreter
```
C:\Users\sharm\Desktop\tomato-attention\venv\python.exe
3.11.16 | packaged by Anaconda, Inc. | (MSC v.1942 64 bit (AMD64))
```

### 3.2 Installed packages (relevant)
| Package | Version | Status |
|---|---|---|
| torch | 2.14.1+cu126 | ✅ |
| torchvision | 0.29.1+cu126 | ✅ |
| numpy | 2.4.6 | ✅ |
| Pillow | 12.3.0 | ✅ |
| PyYAML | 6.0.3 | ✅ |
| pytest | 8.4.2 | ✅ |
| pandas | — | ❌ NOT INSTALLED |
| scikit-learn | — | ❌ NOT INSTALLED |
| matplotlib | — | ❌ NOT INSTALLED |
| seaborn | — | ❌ NOT INSTALLED |
| tqdm | — | ❌ NOT INSTALLED |
| streamlit | — | ❌ NOT INSTALLED |

**CUDA is available** (`torch.cuda.is_available()` → True, CUDA 12.6 build).

### 3.3 Test suite
```
141 passed in 156.46s (0:02:36)
```
**All 141 tests pass.** The `@pytest.mark.realdata` test (`test_one_real_val_batch_passes_through_the_network`) also passed — it loaded a real smoke-test batch through the full model.

### 3.4 Data check
```
10 folders in data/raw/:
  Tomato___Bacterial_spot:               2127 images
  Tomato___Early_blight:                 1000 images
  Tomato___healthy:                      1591 images
  Tomato___Late_blight:                  1909 images
  Tomato___Leaf_Mold:                     952 images
  Tomato___Septoria_leaf_spot:           1771 images
  Tomato___Spider_mites (2-spotted):     1676 images
  Tomato___Target_Spot:                  1404 images
  Tomato___Tomato_mosaic_virus:           373 images
  Tomato___Tomato_Yellow_Leaf_Curl_Virus:5357 images
  TOTAL: 18,160 images
```

Split manifest already computed (seed=42, 70/15/15):
- train: 12,713 | val: 2,724 | test: 2,723
- max abs deviation: 0.000427 (within tolerance 0.01)
- 14 duplicate groups found, 0 groups crossing splits ✅

---

## 4. Spec Compliance Review

### Architecture
| Spec element | Code | Status |
|---|---|---|
| EfficientNet-B0 backbone (configurable) | `src/models/backbone.py` — `_build_efficientnet_b0` | ✅ |
| CBAM: channel-first then spatial | `attention.py` — `forward()` calls `channel_attention` then `spatial_attention` | ✅ |
| CBAM: reduction ratio 16 | Config-driven, enforced in `build_attention` | ✅ |
| CBAM: 7×7 spatial conv | Config-driven, enforced in `CBAM.__init__` | ✅ |
| GAP → Dense(256) + ReLU → Dropout(0.4) → Dense(10) | `net.py` | ✅ |
| Baseline / SE / CBAM switch | `build_attention()` factory | ✅ |
| Backbone switch (resnet50, mobilenetv3) | Config-switchable but raises `ModelConfigError` (not implemented yet, intentional) | ✅ intentional |

### Data Pipeline
| Spec element | Code | Status |
|---|---|---|
| Stratified 70/15/15 split | `splitting.py` | ✅ |
| Fixed seed 42 | `cfg.seed.value = 42` | ✅ |
| Augmentation ONLY on train | `dataloaders.py` — `build_train_transform` for train, `build_eval_transform` for val/test | ✅ |
| Val/test only resize+normalize | `transforms.py` — `build_eval_transform` | ✅ |
| Exact-duplicate check before split | `hashing.py` + `splitting.py`, SHA-256 groups | ✅ |
| Class weights | Config has `class_weights: none` — **not yet computed/applied** (will be needed for training) | ⚠️ deferred to training phase |
| Class names from folder names | Class names from config order only, never filesystem sort | ✅ |

### Training Config (not yet implemented, but config values are correct)
| Spec element | Config value | Status |
|---|---|---|
| AdamW | `optimizer.name: adamw` | ✅ config |
| Cosine schedule | `scheduler.name: cosine_annealing` | ✅ config |
| Label smoothing 0.1 | `training.loss.label_smoothing: 0.1` | ✅ config |
| Frozen-then-unfrozen backbone | `training.freeze_backbone.enabled: true, epochs: 5` | ✅ config |
| Early stopping | `training.early_stopping.enabled: true, patience: 7` | ✅ config |
| Best checkpoint saved | `training.checkpoint.save_best_only: true` | ✅ config |
| No hardcoded hyperparameters | All in config | ✅ |

### Data Leakage & Test-Set Isolation
- **No data leakage:** duplicate groups atomically assigned to one split, no crossing (verified: 0 groups crossing splits).
- **Test set used in model selection?** The training loop does not exist yet. Config sets `training.checkpoint.monitor: val_loss` — monitor is val, not test. ✅ No test set leakage designed in.

---

## 5. Existing Outputs

| Output | Status |
|---|---|
| `checkpoints/` | Empty (no `.pt` files) — no training has been run |
| `results/manifests/hashes.csv` | ✅ Present — 18,160 rows |
| `results/manifests/splits.csv` | ✅ Present — 18,160 rows |
| `results/manifests/split_summary.json` | ✅ Present |
| `results/manifests/dataset_index.csv` | ✅ Present |
| `results/figures/` | Does not exist — not yet created |
| `results/logs/` | Does not exist — not yet created |
| `results/gradcam/` | Does not exist — not yet created |
| `results/predictions/` | Does not exist — not yet created |

**Summary:** All manifest artifacts from Phases 2A/2B/2C are real and reproducible. No training has been performed. No model results exist.

---

## 6. Bugs Found

| # | Severity | Location | Bug | Fix |
|---|---|---|---|---|
| B1 | 🔴 Blocking | `venv/` | `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `streamlit` not installed — all required by evaluation, training and app phases | Run `.\venv\python.exe -m pip install -r requirements.txt` |
| B2 | 🟡 Minor | `README.md` line 123 | Says "Implementation status: **not implemented yet** (Phase 3)" for CBAM — but CBAM **is** implemented | Update README |
| B3 | 🟡 Minor | `README.md` line 328 | Says "Implementation status of the config loader: **not implemented yet** (Phase 2)" — but it is implemented | Update README |
| B4 | 🟡 Minor | `README.md` line 342 | Says "Implementation status of the data pipeline: **not implemented yet** (Phase 2)" — but it is implemented | Update README |
| B5 | 🟡 Minor | `README.md` line 40–41 | Phase roadmap row for "Models" shows status "Phase 3 ✅ complete" but is listed in row 40 as Phase 2/3 confusion | Minor wording clarification |
| B6 | 🔵 Info | `src.zip` / `tests.zip` | Zip archives of src/ and tests/ at project root — likely Cline artefacts; not harmful but should be cleaned up | Delete or git-ignore |
| B7 | 🔵 Info | `backbone.py` | `resnet50` and `mobilenet_v3_large` raise `ModelConfigError` intentionally — but the spec says they should be supported (configurable backbone switch). Only `efficientnet_b0` is implemented so far. | Implement in Step 1 or note as deferred. |

**No logic bugs or data-leakage issues found in the implemented code.**

---

## 7. Proposed Plan

### Step 1 — Fix and complete (before training)
1. **B1: Install missing packages** — `pip install -r requirements.txt` (the packages are declared; just not installed).
2. **B2–B5: Fix README stale text** — update phase status strings.
3. **B6: Remove zip archives** — delete `src.zip` and `tests.zip`.
4. **Implement `src/train.py`** — training loop with AdamW, cosine schedule + warmup, label smoothing, frozen/unfrozen backbone, early stopping, best checkpoint, GradScaler (AMP), grad clip, logging.
5. **Implement `src/evaluate.py`** — load checkpoint, evaluate on test split, compute metrics (accuracy, macro F1, per-class report, confusion matrix, inference time), save artifacts.
6. **Implement `src/gradcam.py`** — Grad-CAM on post-attention features.
7. **Implement `app/app.py`** — Streamlit demo.
8. **Unit tests** — add tests for train smoke test, evaluate, Grad-CAM.

### Checkpoint B (GPU check)
- CUDA 12.6 is available, `torch 2.14.1+cu126` installed.
- **GPU availability: YES** — will confirm GPU model and VRAM before training.

### Step 2 — Train all variants (after approval)
- Run baseline, SE, CBAM with identical seed and settings.
- Evaluate on test set.
- Build `outputs/ablation.csv` and `ablation.md`.
- Generate Grad-CAM examples.
- Finish README (with Mermaid diagram).

---
