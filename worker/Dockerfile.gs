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

CMD ["python", "-m", "app.worker_main", "--worker-class", "gs"]
