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

# CUDA-extension build configuration. The `docker build` runtime has
# no GPU, so torch.cuda.is_available() returns False and naive
# setup.py scripts that probe for runtime CUDA bail with "CUDA not
# found, cannot compile backend!". FORCE_CUDA + an explicit
# TORCH_CUDA_ARCH_LIST work around it for both gsplat (when it
# source-builds — the prebuilt wheel covers most cases) and MonoGS's
# diff-gaussian-rasterization / simple-knn submodules.
# Cover the common gaming/datacenter cards: V100 (7.0), T4 (7.5),
# A100 (8.0), RTX 3000 (8.6), RTX 4000 + L40 (8.9), H100 (9.0).
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

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

# MonoGS upstream pulls in a handful of pure-Python deps that aren't
# in the base image. We've now hit `munch` and `wandb` at runtime;
# the rest of this list is the "next likely candidates" set scraped
# from MonoGS's imports — installing them up front so we stop
# whack-a-mole'ing one missing-import error per scan:
#
#   munch       — config dict wrapper used by `MonoGS/utils/config_utils`.
#   pyyaml      — config-file loading.
#   imageio     — image read/write outside opencv.
#   lpips       — perceptual loss for refinement.
#   wandb       — logging hooks; MonoGS imports it eagerly even when
#                 the run-time `--use_wandb` flag is off, so a missing
#                 install is a hard ImportError on first call.
#   plyfile     — splat I/O (already in the trainer pip install above
#                 but listed here for clarity if that one moves).
#   evo         — trajectory evaluation utilities.
#   matplotlib  — plot generation (some MonoGS code paths import it
#                 unconditionally even when no plot is rendered).
#
# Pin to the major versions MonoGS upstream's requirements file
# nominates; minor-version drift is tolerable because we never call
# these libraries directly.
RUN pip install --no-cache-dir \
        "munch==4.0.0" \
        "pyyaml==6.0.1" \
        "imageio==2.34.2" \
        "lpips==0.1.4" \
        "wandb==0.17.7" \
        "evo==1.28.0" \
        "matplotlib==3.9.2"

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
    && (pip install --no-cache-dir --no-build-isolation submodules/diff-gaussian-rasterization \
            || echo "monogs: diff-gaussian-rasterization build failed; CUDA path will fall back to simulated") \
    && (pip install --no-cache-dir --no-build-isolation submodules/simple-knn \
            || echo "monogs: simple-knn build failed; CUDA path will fall back to simulated") \
    && rm -rf /opt/monogs/.git
ENV PYTHONPATH=/opt/monogs:$PYTHONPATH

CMD ["python", "-m", "app.worker_main", "--worker-class", "gs"]
