FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ARG USE_FLASHINFER=0
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    DATA_DIR=/data \
    MODELS_DIR=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        python3-pip \
        git curl ca-certificates \
        ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libxcb1 libxcb-xinerama0 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.9.1 + cu128 — must match lingbot-map upstream.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.9.1" "torchvision==0.24.1"

# Core API + infra deps.
RUN pip install \
        "fastapi==0.118.0" \
        "uvicorn[standard]==0.32.0" \
        "websockets==13.1" \
        "python-multipart==0.0.9" \
        "pydantic==2.9.2" \
        "pydantic-settings==2.5.2" \
        "aiosqlite==0.20.0" \
        "sqlalchemy[asyncio]==2.0.36" \
        "aiofiles==24.1.0" \
        "numpy<2.3" \
        "pymeshlab==2023.12.post3" \
        "huggingface_hub==0.26.2"

# lingbot-map (with vis extras: viser/trimesh/onnxruntime for sky mask + glb export).
RUN pip install "lingbot-map[vis] @ git+https://github.com/Robbyant/lingbot-map.git"

# Optional FlashInfer (paged-KV attention). Skip when USE_FLASHINFER=0 (SDPA fallback).
RUN if [ "$USE_FLASHINFER" = "1" ]; then \
        pip install flashinfer-python -i https://flashinfer.ai/whl/cu128/torch2.9/ ; \
    fi

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
