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
        "matplotlib==3.9.2" \
        "open3d==0.18.0" \
        "torchmetrics==1.4.0.post0"
        # ^ torchmetrics is imported eagerly by `utils/eval_utils.py:15`
        # (`from torchmetrics.image.lpip import
        # LearnedPerceptualImagePatchSimilarity`). `slam.py` imports
        # `eval_utils` at module top, so a missing torchmetrics is a
        # hard ModuleNotFoundError seven lines into the post-stop
        # subprocess — even though we run with `eval_rendering: False`
        # and never reach the LPIPS code path. Pin to 1.4.0.post0 — the
        # last release in the 1.4.x series that still works against
        # torch 2.3.1; 1.5+ requires torch>=2.4.
        # ^ open3d is the next missing-import we hit after wandb;
        # MonoGS imports it eagerly during scene construction even
        # when the user isn't running a viewer that needs it. ~500 MB
        # uncompressed but mandatory; pinning to 0.18.0 (the last
        # version with prebuilt wheels for Python 3.11 + torch 2.3).

# Build deps for MonoGS source build (its 3DGS rasterizer ships its own
# CUDA extensions).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ninja-build git \
    && rm -rf /var/lib/apt/lists/*

# Explicit CUDA env for setup.py CUDAExtension probes. The base
# image (nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04) installs the
# toolkit at /usr/local/cuda but doesn't always export CUDA_HOME on
# every shell — torch's `CUDAExtension` sniffs that env var to
# locate nvcc and the headers, and falls over silently when it's
# unset. Setting it here means simple-knn / diff-gaussian-
# rasterization both find nvcc deterministically.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

# MonoGS / Photo-SLAM (Phase 5). Upstream is a runnable research project
# (multiple top-level dirs: gui/, gaussian_splatting/, submodules/, etc.)
# without proper Python packaging — `pip install .` fails on the
# multi-top-level-package layout.
#
# Workaround: clone the repo, install its CUDA rasterizer + KNN
# submodules directly (they're the only pieces with real setup.py
# files), and add the repo root to PYTHONPATH so Python can import
# `gaussian_splatting.*` from wherever the source landed.
#
# Build is strict: if either submodule fails to compile we let the
# image build fail with the actual nvcc/g++ error. The previous
# `|| echo "build failed; falls back to simulated"` swallowed real
# compile errors and produced an image that booted fine but blew up
# at first job with `ModuleNotFoundError: No module named 'simple_knn'`
# — the user explicitly called out that simulated splat output is
# useless, so we'd rather catch the build break here.
ARG MONOGS_SHA=main
RUN git clone --recursive https://github.com/muskie82/MonoGS.git /opt/monogs \
    && cd /opt/monogs \
    && git checkout ${MONOGS_SHA} \
    # Patch simple_knn.cu to include <float.h>. Upstream
    # (gitlab.inria.fr/bkerbl/simple-knn) uses `FLT_MAX` without
    # an explicit include; CUDA <12 implicitly pulled it in via
    # crt headers, but CUDA 12.8 (our base image) doesn't, and
    # the build fails with "identifier "FLT_MAX" is undefined" at
    # simple_knn.cu(90) and (154). Inserting `#include <float.h>`
    # at the top of the file is the upstream-issue's standard
    # fix and idempotent if applied twice. Without this patch the
    # post-stop MonoGS jobs blow up at first frame with
    # `ModuleNotFoundError: No module named 'simple_knn'` because
    # the wheel never builds.
    && sed -i '1i #include <float.h>' submodules/simple-knn/simple_knn.cu \
    # Patch upstream slam.py's unconditional GUI import. The bare line
    #   from gui import gui_utils, slam_gui
    # at module top loads `gui/slam_gui.py` even when `Results.use_gui`
    # is False — and slam_gui pulls in a full desktop GUI stack (glfw,
    # imgviz, PyOpenGL) we don't ship in the headless worker. We do
    # need `gui_utils` (it only imports cv2 + open3d + torch, all
    # already in the image), and gui_utils.ParamsGUI is constructed
    # unconditionally by SLAM.__init__; but `slam_gui.run` is only
    # called inside `if self.use_gui:`, so a no-op stand-in works. The
    # sed below splits the import + introduces `types.SimpleNamespace`
    # as a stub. Idempotent: a second run finds no match and is a
    # no-op. Without this patch every post-stop MonoGS job dies with
    # `ModuleNotFoundError: No module named 'glfw'` six lines into
    # slam.py's imports.
    && sed -i 's|^from gui import gui_utils, slam_gui$|from gui import gui_utils\nimport types as _t\nslam_gui = _t.SimpleNamespace(run=lambda *a, **kw: None)|' slam.py \
    # Patch upstream's `dtype=np.unicode_` use in `utils/dataset.py:55`.
    # `np.unicode_` was removed in NumPy 2.0 (NEP 51); upstream MonoGS
    # is research code from a pre-numpy-2 era. The replacement
    # `np.str_` is the official 2.x equivalent and works fine on
    # numpy 1.x too, so the patch is forward+backward compatible.
    # Without this the TUMParser blows up at the first `parse_list`
    # call with `AttributeError: 'np.unicode_' was removed in the
    # NumPy 2.0 release`. Idempotent: second run finds no match.
    && sed -i 's|dtype=np\.unicode_|dtype=np.str_|g' utils/dataset.py \
    # Switch the multiprocessing start method from "spawn" to "fork".
    # Upstream `slam.py:209` forces `mp.set_start_method("spawn")`,
    # which pickles `self.backend` (containing GaussianModel CUDA
    # tensors) to hand it to the spawned child. The child then calls
    # `cudaIpcOpenMemHandle` to reconstruct the tensors — and that
    # IPC handle round-trip is fragile in containerized setups, even
    # with `shm_size: 8gb`. Symptom on first frame:
    #
    #   File "torch/multiprocessing/reductions.py", in rebuild_cuda_tensor
    #     storage = storage_cls._new_shared_cuda(...)
    #   RuntimeError: CUDA error: invalid resource handle
    #
    # `fork` inherits the parent's CUDA context directly — no pickle,
    # no IPC handle round-trip, just shared address space. PyTorch
    # warns against fork+CUDA in general (cuBLAS / cuDNN / NCCL state
    # can be tangled), but for MonoGS's batch backend (single GPU,
    # single host, no NCCL, fresh cuBLAS handles created lazily) it
    # works reliably. We force=True so multiple SLAM runs in the same
    # python process can re-set the method without complaint.
    # Idempotent: second run finds no match.
    && sed -i 's|^    mp.set_start_method("spawn")$|    mp.set_start_method("fork", force=True)  # patched: spawn fails CUDA IPC in container|' slam.py \
    && pip install --no-cache-dir --no-build-isolation submodules/diff-gaussian-rasterization \
    && pip install --no-cache-dir --no-build-isolation submodules/simple-knn \
    && rm -rf /opt/monogs/.git
ENV PYTHONPATH=/opt/monogs:$PYTHONPATH

CMD ["python", "-m", "app.worker_main", "--worker-class", "gs"]
