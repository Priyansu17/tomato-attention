# Progress Log — Tomato Attention Project

---

## 2026-10-15 — Takeover from Cline (Antigravity)

## 2026-10-15 — Step 1 Complete (Antigravity)

**Commit:** `e8b1dcf`

### Fixes applied
- **B1** — installed all missing packages (pandas, scikit-learn, matplotlib, seaborn, tqdm, streamlit, jupyterlab, ipykernel)
- **B2–B5** — fixed 3 stale README sections that said "not implemented yet"
- **B6** — deleted `src.zip` and `tests.zip` (Cline artefacts)

### New files
- `src/train.py` — full training loop (AdamW, cosine+warmup, label smoothing 0.1, AMP, freeze/unfreeze, early stopping, best/last checkpoint)
- `src/evaluate.py` — test-set evaluation (accuracy/F1/ROC-AUC, confusion matrix, predictions CSV, metrics JSON)
- `src/gradcam.py` — Grad-CAM overlays on post-attention features
- `app/app.py` — Streamlit demo
- `tests/test_train.py` — smoke training tests
- `docs/audit.md`, `docs/progress_log.md`

### Smoke test
Passed: 1 epoch on 48 images, CUDA (RTX 3050 4 GB), 28 s, best checkpoint saved.

### Checkpoint B
GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4.29 GB VRAM, CUDA 12.6.
Estimated training time: ~6–8 h for all 3 variants. Awaiting user approval.

---

## 2026-10-15 — Step 2: Run-name fixes, ablation runner, smoke test (Antigravity)

**Previous context lost due to quota error; resumed on new account.**

### Fixes applied

**Context recovery (Step 2A):**
- Verified `git status`: `src/train.py` and `docs/progress_log.md` had uncommitted changes (the `--run-name` arg addition). No training had run.
- No Python processes running. Only smoke-test checkpoints (`best_model.pt`, `last_model.pt`) existed.

**train.py — per-variant naming (Step 2B):**
- `last_ckpt` was hardcoded as `last_model.pt` — all variants would have overwritten it. Fixed: `last_ckpt` is now derived from `best_ckpt` stem (e.g. `last_model_cbam.pt`).
- `training_history.json` was fixed name — all variants would have overwritten it. Fixed: now `training_history_best_model_{variant}.json`.

**evaluate.py — per-variant naming + architecture fix (Step 2C):**
- Added `--run-name` CLI arg; all outputs (metrics JSON, confusion matrix PNG, predictions CSV) are now namespaced by variant.
- Added checkpoint config merge before model build: reads `attention.type` from the checkpoint's saved config instead of from `configs/config.yaml`. Prevents architecture mismatch when running evaluate for variant X with variant Y's config.

**gradcam.py — per-variant naming + architecture fix (Step 2D):**
- Added `--run-name` CLI arg; Grad-CAM images are saved to `results/gradcam/{variant}/` subdirs.
- Same checkpoint config merge fix applied as in `evaluate.py`.

### New files
- `src/run_ablation.py` — resumable sequential runner (baseline → SE → CBAM). Skips variants whose `results/metrics_test_{variant}.json` exists. Tees all output to `outputs/logs/ablation_{variant}.log`. Supports `--smoke-test` and `--variants`.
- `src/ablation.py` — aggregation script: reads per-variant metrics JSONs and training history JSONs, writes `outputs/ablation.csv` and `outputs/ablation.md`.

### Smoke test (ablation runner, 1 epoch, 48 training images)
Passed in 6.1 min — all 3 variants completed train + eval + Grad-CAM:

| Variant | Accuracy | F1 Macro |
|---------|----------|----------|
| baseline | 0.1590 | 0.0791 |
| se | 0.0779 | 0.0412 |
| cbam | 0.0936 | 0.0443 |

*(1-epoch smoke metrics on untrained weights — not meaningful, just pipeline validation.)*

Intermediate bug found and fixed during smoke test: evaluate.py was building the model from `configs/config.yaml` attention type instead of from the checkpoint's saved config, causing `RuntimeError: Missing key(s) in state_dict` when evaluating SE checkpoint with CBAM config active. Fixed in both evaluate.py and gradcam.py.

---



**Previous agent:** Cline  
**Git snapshot:** 5 commits from Cline (da0bafc through a42f9b4)

### Audit Summary
- Phases 1, 2A, 2B, 2C, 2D all implemented by Cline and verified passing.
- 141 tests pass (0 failures) on `.\venv\python.exe`.
- CUDA available: `torch 2.14.1+cu126`, Python 3.11.16.
- Data: 18,160 images, 10 classes, split manifest computed (seed=42, 70/15/15).
- No training has been performed.
- Missing from venv: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `streamlit`.
- Missing source: `src/train.py`, `src/evaluate.py`, `src/gradcam.py`, `app/app.py`.
- Minor README stale text (CBAM/config/data pipeline sections say "not implemented" but they are).

### Planned next steps (pending user approval of audit)
1. Install missing packages (B1).
2. Fix README stale text (B2–B5).
3. Remove `src.zip` / `tests.zip` (B6).
4. Implement `src/train.py`, `src/evaluate.py`, `src/gradcam.py`, `app/app.py`.
5. Smoke test training pipeline.
6. Checkpoint B: GPU confirmation and training time estimates.
7. After approval: train baseline / SE / CBAM variants, evaluate, ablation table, Grad-CAM, README update.

---
