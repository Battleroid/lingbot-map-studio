### Gaussian-splat training worker.
###
### Hosts the real `gsplat` CUDA trainer (`app.processors.gsplat.cuda_trainer`).
### When `gsplat` and `torch.cuda` are both available, GsplatProcessor
### auto-selects GsplatCudaTrainer; otherwise it falls back to the
### simulated trainer with a warn event.
###
### Build: `docker build -t lingbot-studio/worker-gs:latest -f worker/Dockerfile.gs worker`
ARG BASE_IMAGE=lingbot-studio/base:latest
FROM ${BASE_IMAGE}

ENV WORKER_CLASS=gs

# gsplat ships prebuilt wheels against cu121 / torch 2.3.
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.3.1" "torchvision==0.18.1"

RUN pip install \
        "opencv-python-headless==4.10.0.84" \
        "scipy==1.13.1" \
        "einops==0.8.0" \
        "trimesh==4.4.1" \
        "plyfile==1.1"

# Real CUDA gsplat. The PyPI wheel ships prebuilt against torch 2.3 + cu121,
# matching the install above; if the wheel is missing for the host arch it
# source-builds with the CUDA toolkit already present in the base image.
RUN pip install "gsplat==1.4.0"

# Build deps for MonoGS source build (its 3DGS rasterizer ships its own
# CUDA extensions).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ninja-build git \
    && rm -rf /var/lib/apt/lists/*

# MonoGS / Photo-SLAM (Phase 5). The upstream is research-grade with a
# multi-process architecture; clone + `pip install .` source-builds the
# rasterization kernels. Bumping SHA is one line. Note: MonoGS's bundled
# 3DGS rasterizer can coexist with the `gsplat` package above because
# each registers under its own Python module namespace; if you observe
# import-order issues at runtime, ensure `gsplat` is imported before
# `monogs` (the trainer factory in app.processors.gsplat.trainer does
# this implicitly because GsplatCudaTrainer touches gsplat first).
ARG MONOGS_SHA=main
RUN git clone https://github.com/muskie82/MonoGS.git /opt/monogs \
    && cd /opt/monogs \
    && git checkout ${MONOGS_SHA} \
    && pip install --no-cache-dir . \
    && cd / && rm -rf /opt/monogs/.git

CMD ["python", "-m", "app.worker_main", "--worker-class", "gs"]
