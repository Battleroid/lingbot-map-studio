from __future__ import annotations

import asyncio
import glob
import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.jobs.schema import JobConfig, JobEvent
from app.pipeline.progress import capture_stdio

log = logging.getLogger(__name__)

# lingbot-map wants a large GPU memory arena; set before first CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PublishFn = Callable[[JobEvent], "asyncio.Future | None"]


def _list_frames(frames_dir: Path) -> list[str]:
    return sorted(glob.glob(str(frames_dir / "*.png")))


def _choose_dtype() -> "Any":
    import torch

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def _build_model(config: JobConfig):
    """Import the correct class (both modules export GCTStream)."""
    if config.mode == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream as ModelCls
    else:
        from lingbot_map.models.gct_stream import GCTStream as ModelCls

    return ModelCls(
        img_size=config.image_size,
        patch_size=config.patch_size,
        enable_3d_rope=config.enable_3d_rope,
        max_frame_num=config.max_frame_num,
        kv_cache_sliding_window=config.kv_cache_sliding_window,
        kv_cache_scale_frames=config.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=config.use_sdpa,
        camera_num_iterations=config.camera_num_iterations,
    )


def _postprocess(predictions: dict, image_shape) -> dict:
    """Match demo.py: pose_enc -> extrinsic/intrinsic, squeeze batch, numpy."""
    import torch

    from lingbot_map.utils.geometry import closed_form_inverse_se3_general
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    pose_enc = predictions["pose_enc"]
    extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_shape[-2:])
    extrinsic_c2w = closed_form_inverse_se3_general(extrinsic_w2c)
    predictions["extrinsic"] = extrinsic_c2w
    predictions["intrinsic"] = intrinsic

    out: dict[str, Any] = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().to("cpu")
            while arr.ndim > 3 and arr.shape[0] == 1 and k not in ("extrinsic", "intrinsic"):
                arr = arr.squeeze(0)
            out[k] = arr.float().numpy() if arr.is_floating_point() else arr.numpy()
        else:
            out[k] = v
    return out


def _run_inference_sync(
    frames: list[str],
    ckpt_path: Path,
    config: JobConfig,
    progress_cb: Callable[[int, int], None],
) -> tuple[dict, tuple[int, ...]]:
    """Synchronous GPU inference. Runs in a worker thread."""
    import torch

    from lingbot_map.utils.load_fn import load_and_preprocess_images

    print(f"Loading {len(frames)} images...")
    images = load_and_preprocess_images(
        frames,
        mode="crop",
        image_size=config.image_size,
        patch_size=config.patch_size,
    )
    if isinstance(images, tuple):
        images = images[0]
    print(f"Preprocessed images to {tuple(images.shape)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _choose_dtype()
    print(f"Device: {device}, dtype: {dtype}")

    print("Building model...")
    model = _build_model(config)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"GPU mem after load: alloc={alloc:.1f}MB")

    # Progress: hook model.forward to count per-frame calls inside the streaming loop.
    orig_forward = model.forward
    frame_count = {"i": 0}
    total = int(images.shape[0] if images.ndim == 4 else images.shape[1])

    def _hooked_forward(*args, **kwargs):
        result = orig_forward(*args, **kwargs)
        frame_count["i"] += 1
        progress_cb(frame_count["i"], total)
        return result

    model.forward = _hooked_forward  # type: ignore[assignment]

    images_dev = images.to(device)
    if images_dev.ndim == 4:
        images_dev = images_dev.unsqueeze(0)

    output_device = torch.device("cpu") if config.offload_to_cpu else None

    print(f"Running {config.mode} inference on {total} frames...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        if config.mode == "windowed":
            predictions = model.inference_windowed(
                images_dev,
                window_size=config.window_size,
                overlap_size=config.overlap_size,
                num_scale_frames=config.num_scale_frames,
                keyframe_interval=config.keyframe_interval,
                output_device=output_device,
            )
        else:
            predictions = model.inference_streaming(
                images_dev,
                num_scale_frames=config.num_scale_frames,
                keyframe_interval=config.keyframe_interval,
                output_device=output_device,
            )

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"GPU peak during inference: {peak:.1f}MB")

    image_shape = tuple(images_dev.shape)
    np_preds = _postprocess(predictions, image_shape)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np_preds, image_shape


async def run_inference(
    job_id: str,
    frames_dir: Path,
    ckpt_path: Path,
    config: JobConfig,
    publish: PublishFn,
) -> dict:
    frames = _list_frames(frames_dir)
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")

    loop = asyncio.get_running_loop()

    def _progress(done: int, total: int) -> None:
        if total <= 0:
            return
        ev = JobEvent(
            job_id=job_id,
            stage="inference",
            message=f"frame {done}/{total}",
            progress=done / total,
            data={"done": done, "total": total},
        )
        try:
            asyncio.run_coroutine_threadsafe(_publish_async(publish, ev), loop)
        except Exception:
            pass

    def _sync_target():
        with capture_stdio(job_id, publish, "inference", loop):
            return _run_inference_sync(frames, ckpt_path, config, _progress)

    predictions, image_shape = await asyncio.to_thread(_sync_target)

    # Cache the raw predictions so re-exports and mesh edits don't need GPU.
    npz_path = frames_dir.parent / "artifacts" / "predictions.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        **{k: v for k, v in predictions.items() if isinstance(v, np.ndarray)},
    )
    await publish(
        JobEvent(
            job_id=job_id,
            stage="inference",
            message=f"cached raw predictions -> {npz_path.name}",
            data={"path": str(npz_path)},
            progress=1.0,
        )
    )
    return predictions


async def _publish_async(publish: PublishFn, event: JobEvent) -> None:
    res = publish(event)
    if asyncio.iscoroutine(res):
        await res


def load_cached_predictions(job_id: str, data_dir: Path) -> dict:
    npz_path = data_dir / "jobs" / job_id / "artifacts" / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No cached predictions for {job_id}")
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}
