# Ablation Study — Tomato Leaf Disease Detection

_Generated: 2026-10-17 14:27  (seed 42, EfficientNet-B0 backbone, 70/15/15 split)_

## Test-Set Results

| Variant | Accuracy | F1 Macro | Precision | Recall | ROC-AUC | ms/img |
|---------|----------|----------|-----------|--------|---------|--------|
| baseline | 0.9923 | 0.9894 | 0.9910 | 0.9880 | nan | 10.88 |
| se       | 0.9934 | 0.9915 | 0.9924 | 0.9908 | nan | 7.05 |
| cbam     | 0.9938 | 0.9920 | 0.9928 | 0.9913 | nan | 7.56 |

## Training Convergence

| Variant | Best Epoch | Total Epochs | Val Acc (best) | Val Loss (best) |
|---------|------------|--------------|----------------|-----------------|
| baseline |         40 |           40 | 0.9930 | 0.5466 |
| se       |         40 |           40 | 0.9927 | 0.5467 |
| cbam     |         35 |           38 | 0.9919 | 0.5400 |

## Notes

- **baseline**: EfficientNet-B0 backbone → GAP → Dense(256) → Dropout(0.4) → Dense(10). No attention.
- **se**: Squeeze-and-Excitation channel attention added after backbone features.
- **cbam**: Channel + Spatial attention (CBAM). Channel: reduction=16, avg+max pool. Spatial: 7×7 conv.
- All three variants trained with **identical** seed, split, optimiser, and schedule.
- Grad-CAM images are in `results/gradcam/<variant>/`.

> Results labelled **unverified** if the checkpoint cannot be reproduced from the current code and config.
