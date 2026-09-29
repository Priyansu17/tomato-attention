"""Model package for the tomato leaf disease detection project.

Phase 1 contains no implementation yet. Placed here in later phases:
  * ``backbone``  — pretrained CNN backbone wrapper (EfficientNet-B0, ...),
  * ``attention`` — attention modules (baseline / SE / CBAM),
  * ``net``       — full classification network following the locked architecture.

Locked flow: image -> backbone -> attention -> GAP -> Dense(256)+ReLU ->
Dropout(0.4) -> Dense(10).
"""
