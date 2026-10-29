# =============================================================================
# Dockerfile — TomatoAI (Hugging Face Spaces — CPU Basic, free tier)
#
# HF Spaces Docker SDK runs as a non-root user (UID 1000).
# Port 7860 is the required HF Spaces port.
# =============================================================================

FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (HF Spaces requirement) ─────────────────────────────
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── Install CPU-only PyTorch first (much smaller than the default CUDA build) ─
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining dependencies ───────────────────────────────────────────
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY --chown=appuser:appuser . .

# ── Switch to non-root user ───────────────────────────────────────────────────
USER appuser

# ── Expose HF Spaces port ─────────────────────────────────────────────────────
EXPOSE 7860

# ── Launch FastAPI on port 7860 ───────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
