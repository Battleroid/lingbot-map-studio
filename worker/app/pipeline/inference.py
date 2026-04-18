from __future__ import annotations

import asyncio
import glob
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.jobs.schema import JobConfig, JobEvent
from app.pipeline.progress import capture_stdio
from app.pipeline.watchdog import VramLimitExceeded, VramWatchState

log = logging.getLogger(__name__)

# lingbot-map wants a large GPU memory arena; set before first CUDA init.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
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
    """Convert pose_enc → extrinsic (c2w, 3×4) + intrinsic (3×3), squeeze
    batch dim, move to numpy.

    closed_form_inverse_se3_general returns 4×4 homogeneous matrices, but
    predictions_to_glb expects 3×4 (R|t) — matching demo.py's postprocess.
    """
    import torch

    from lingbot_map.utils.geometry import closed_form_inverse_se3_general
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    pose_enc = predictions["pose_enc"]
    extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_shape[-2:])
    extrinsic_c2w_h = closed_form_inverse_se3_general(extrinsic_w2c)   # (B, S, 4, 4)
    extrinsic_c2w = extrinsic_c2w_h[..., :3, :]                         # (B, S, 3, 4)
    predictions["extrinsic"] = extrinsic_c2w
    predictions["intrinsic"] = intrinsic

    out: dict[str, Any] = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().to("cpu")
            # Squeeze leading batch dim for ALL tensors (including extrinsic /
            # intrinsic — previously excluded, which left them in (1, S, 3, 4)
            # form and broke predictions_to_glb).
            while arr.ndim > 2 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            out[k] = arr.float().numpy() if arr.is_floating_point() else arr.numpy()
        else:
            out[k] = v
    return out


def _write_partial_ply(
    wp_list: list,        # list of torch tensors (1, Ti, H, W, 3), CPU
    wpc_list: list,       # list of torch tensors (1, Ti, H, W), CPU
    imgs_cpu,             # torch tensor (1, T_total, 3, H, W) CPU, sliced to covered frames
    out_path: Path,
    conf_percentile: float,
) -> tuple[int, int]:
    """Build a colored point cloud from accumulated per-frame outputs and save it.

    Returns (point_count, skipped_count).
    """
    import torch
    import trimesh

    if not wp_list or not wpc_list:
        return 0, 0

    wp = torch.cat(wp_list, dim=1)          # (1, T, H, W, 3)
    wpc = torch.cat(wpc_list, dim=1)        # (1, T, H, W)
    T = wp.shape[1]

    imgs = imgs_cpu[:, :T]                  # (1, T, 3, H, W)
    imgs = imgs.permute(0, 1, 3, 4, 2)      # (1, T, H, W, 3)

    pts = wp[0].reshape(-1, 3).float().numpy()
    c = wpc[0].reshape(-1).float().numpy()
    colors_arr = (imgs[0].reshape(-1, 3).float().numpy() * 255).clip(0, 255).astype(np.uint8)

    # Guard against degenerate cases early in inference.
    if c.size < 3:
        return 0, 0
    pct = max(0.0, min(99.0, float(conf_percentile)))
    thresh = float(np.percentile(c, pct))
    mask = (c >= thresh) & (c > 1e-5)
    kept = int(mask.sum())
    skipped = int(c.size - kept)
    if kept == 0:
        return 0, skipped

    pc = trimesh.PointCloud(vertices=pts[mask], colors=colors_arr[mask])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pc.export(str(out_path))
    return kept, skipped


def _run_inference_sync(
    frames: list[str],
    ckpt_path: Path,
    config: JobConfig,
    progress_cb: Callable[[int, int], None],
    partial_cb: Callable[[int, Path, int], None] | None,
    artifacts_dir: Path,
    vram_state: VramWatchState | None = None,
) -> tuple[dict, tuple[int, ...]]:
    """Synchronous GPU inference. Runs in a worker thread.

    Adds per-frame accumulation of world_points + world_points_conf so a
    background thread can periodically snapshot partial PLYs for the viewer.
    """
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

    # Move images to device once; keep a CPU copy for partial PLY color data.
    images_dev = images.to(device)
    if images_dev.ndim == 4:
        images_dev = images_dev.unsqueeze(0)
    images_cpu_full = images_dev.detach().to("cpu")

    total = int(images_dev.shape[1])

    # Accumulators for partial snapshots
    orig_forward = model.forward
    frame_count = {"i": 0}
    wp_accum: list = []
    wpc_accum: list = []
    last_snapshot = {"i": 0}
    snapshot_active = {"v": False}
    snapshot_every = max(0, int(config.partial_snapshot_every))

    def _spawn_snapshot(frames_done: int) -> None:
        """Kick off a background thread that writes a partial PLY."""
        if snapshot_active["v"] or not partial_cb:
            return
        snapshot_active["v"] = True
        # Freeze the accumulator lists and image slice before starting the
        # thread so inference can keep mutating them safely.
        wp_snap = list(wp_accum)
        wpc_snap = list(wpc_accum)
        imgs_snap = images_cpu_full
        out_path = artifacts_dir / f"partial_{frames_done:06d}.ply"

        def _worker():
            try:
                kept, _ = _write_partial_ply(
                    wp_snap, wpc_snap, imgs_snap, out_path,
                    conf_percentile=config.conf_percentile,
                )
                if kept > 0:
                    partial_cb(frames_done, out_path, kept)
            except Exception as exc:  # noqa: BLE001
                log.warning("partial snapshot failed: %s", exc)
            finally:
                snapshot_active["v"] = False

        threading.Thread(target=_worker, daemon=True, name=f"snap-{frames_done}").start()

    def _hooked_forward(*args, **kwargs):
        if vram_state is not None and vram_state.tripped:
            raise VramLimitExceeded(vram_state.reason or "vram soft-limit exceeded")

        # Figure out how many frames this forward call represents. For the
        # streaming loop the first call is the batched scale-frames pass
        # (processes num_scale_frames frames in one go), then each subsequent
        # call processes one frame. For the windowed mode, every window has
        # its own scale-frames batched pass.
        frames_this_call = 1
        if args:
            inp = args[0]
            if hasattr(inp, "shape") and getattr(inp, "ndim", 0) >= 2:
                frames_this_call = int(inp.shape[1])

        result = orig_forward(*args, **kwargs)

        # Stash world_points + conf on CPU. Use detach().cpu() — small copy,
        # but avoids holding a GPU reference across snapshots.
        if isinstance(result, dict):
            wp = result.get("world_points")
            wpc = result.get("world_points_conf")
            if wp is not None and wpc is not None:
                try:
                    wp_accum.append(wp.detach().to("cpu"))
                    wpc_accum.append(wpc.detach().to("cpu"))
                except Exception:  # noqa: BLE001
                    pass

        frame_count["i"] += frames_this_call
        done = min(frame_count["i"], total)
        progress_cb(done, total)

        # Partial snapshot trigger. We want "first snapshot after ~snapshot_every
        # frames, then every snapshot_every frames after that". Drop the call
        # if a previous snapshot is still running so we don't pile up.
        if (
            snapshot_every > 0
            and partial_cb
            and done < total
            and done - last_snapshot["i"] >= snapshot_every
        ):
            last_snapshot["i"] = done
            _spawn_snapshot(done)

        return result

    model.forward = _hooked_forward  # type: ignore[assignment]

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

    # Ensure the final progress value reflects total frames exactly.
    progress_cb(total, total)

    image_shape = tuple(images_dev.shape)
    np_preds = _postprocess(predictions, image_shape)

    # Release accumulators before freeing the model.
    wp_accum.clear()
    wpc_accum.clear()

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
    vram_state: VramWatchState | None = None,
) -> dict:
    frames = _list_frames(frames_dir)
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")

    loop = asyncio.get_running_loop()
    artifacts_dir = frames_dir.parent / "artifacts"

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

    def _partial(done: int, path: Path, points: int) -> None:
        ev = JobEvent(
            job_id=job_id,
            stage="artifact",
            message=f"partial point cloud @ frame {done}: {points:,} pts",
            data={
                "name": path.name,
                "kind": "partial_ply",
                "frames_done": done,
                "points": points,
            },
        )
        try:
            asyncio.run_coroutine_threadsafe(_publish_async(publish, ev), loop)
        except Exception:
            pass

    def _sync_target():
        with capture_stdio(job_id, publish, "inference", loop):
            return _run_inference_sync(
                frames,
                ckpt_path,
                config,
                _progress,
                _partial,
                artifacts_dir,
                vram_state,
            )

    predictions, image_shape = await asyncio.to_thread(_sync_target)

    npz_path = artifacts_dir / "predictions.npz"
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
