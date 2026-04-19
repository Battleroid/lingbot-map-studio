### Gaussian-splat training worker.
###
### Hosts the `gsplat` trainer. Phase 2 ships a stub so the enqueue +
### event flow can be verified; Phase 5 drops in the real `gsplat` CUDA
### wheel and the trainer.
###
### Build: `docker build -t lingbot-studio/worker-gs:latest -f worker/Dockerfile.gs worker`
ARG BASE_IMAGE=lingbot-studio/base:latest
FROM ${BASE_IMAGE}

ENV WORKER_CLASS=gs

# gsplat ships prebuilt wheels against cu121 / torch 2.3 today. Pinning
# matches so Phase 5 can `pip install gsplat==1.x.y` without a source build.
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.3.1" "torchvision==0.18.1"

RUN pip install \
        "opencv-python-headless==4.10.0.84" \
        "scipy==1.13.1" \
        "einops==0.8.0" \
        "trimesh==4.4.1"

# Phase 5 adds: pip install gsplat==<pinned> plyfile viser

CMD ["python", "-m", "app.worker_main", "--worker-class", "gs"]
