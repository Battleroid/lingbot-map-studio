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

# MonoGS / Photo-SLAM (Phase 5). Upstream is a runnable research project
# (multiple top-level dirs: gui/, gaussian_splatting/, submodules/, etc.)
# without proper Python packaging — `pip install .` fails on the
# multi-top-level-package layout.
#
# Workaround: clone the repo, install its CUDA rasterizer submodule
# directly (the only piece with a real setup.py), and add the repo
# root to PYTHONPATH so Python can import `gaussian_splatting.*` from
# wherever the source landed. The wrapper in
# `app.processors.gsplat.monogs_cuda` probes those module paths and
# falls back to the simulated session cleanly when any piece is
# missing — Phase 0's warn event tells the user.
#
# If the diff-gaussian-rasterization submodule build fails (CUDA arch
# mismatch, missing nvcc, etc.) the rest of the image still builds —
# the wrapper's import probe will fail and MonoGS auto-falls back to
# the simulated session.
ARG MONOGS_SHA=main
RUN git clone --recursive https://github.com/muskie82/MonoGS.git /opt/monogs \
    && cd /opt/monogs \
    && git checkout ${MONOGS_SHA} \
    && (pip install --no-cache-dir submodules/diff-gaussian-rasterization \
            || echo "monogs: diff-gaussian-rasterization build failed; CUDA path will fall back to simulated") \
    && (pip install --no-cache-dir submodules/simple-knn \
            || echo "monogs: simple-knn build failed; CUDA path will fall back to simulated") \
    && rm -rf /opt/monogs/.git
ENV PYTHONPATH=/opt/monogs:$PYTHONPATH

CMD ["python", "-m", "app.worker_main", "--worker-class", "gs"]
