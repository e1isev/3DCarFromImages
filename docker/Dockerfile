# car2asset — Docker image
# Base: PyTorch 2.2 + CUDA 12.1 (requires nvidia-container-toolkit on host)
# Build: docker build -t car2asset:latest -f docker/Dockerfile .
# Run:   docker run --gpus all -p 8000:8000 car2asset:latest

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="e1isev"
LABEL description="AI-assisted car photo → 3D asset pipeline"

# ── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# ── Python deps ──────────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY blender/ ./blender/
COPY api/ ./api/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e ".[api]"

# ── Output and model cache volumes ────────────────────────────────────────────
RUN mkdir -p /app/outputs /app/models /app/inputs
VOLUME ["/app/outputs", "/app/models", "/app/inputs"]

# ── Runtime ───────────────────────────────────────────────────────────────────
ENV CAR2ASSET_DEVICE=cuda
ENV CAR2ASSET_QUALITY=high
ENV CAR2ASSET_MODE=both
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
