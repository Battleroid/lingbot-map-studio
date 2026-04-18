from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from app.jobs.schema import JobConfig, JobEvent
from app.pipeline.progress import capture_stdio

log = logging.getLogger(__name__)

Format = Literal["glb", "ply", "obj"]


def _apply_sky_mask_to_predictions(predictions: dict, frames_dir: Path) -> dict:
    """Run lingbot-map's sky segmentation over the original frames and zero
    out confidence for sky pixels. Mutates and returns the dict.
    """
    from lingbot_map.vis.sky_segmentation import apply_sky_segmentation

    conf = predictions.get("world_points_conf")
    if conf is None:
        return predictions
    updated = apply_sky_segmentation(
        conf,
        image_folder=str(frames_dir),
        sky_mask_dir=str(frames_dir.parent / "sky_masks"),
    )
    predictions["world_points_conf"] = updated
    return predictions


def _scene_from_predictions(
    predictions: dict,
    config: JobConfig,
    *,
    conf_percentile: Optional[float] = None,
    show_cam: Optional[bool] = None,
    mask_sky: Optional[bool] = None,
    mask_black_bg: Optional[bool] = None,
    mask_white_bg: Optional[bool] = None,
) -> "trimesh.Scene":  # type: ignore[name-defined]
    from lingbot_map.vis.glb_export import predictions_to_glb

    return predictions_to_glb(
        predictions,
        conf_thres=conf_percentile if conf_percentile is not None else config.conf_percentile,
        filter_by_frames="all",
        mask_black_bg=mask_black_bg if mask_black_bg is not None else config.mask_black_bg,
        mask_white_bg=mask_white_bg if mask_white_bg is not None else config.mask_white_bg,
        show_cam=show_cam if show_cam is not None else config.show_cam,
        mask_sky=mask_sky if mask_sky is not None else config.mask_sky,
        prediction_mode="Predicted Pointmap",
    )


def _export_ply_pointcloud(
    predictions: dict,
    config: JobConfig,
    out: Path,
    *,
    conf_percentile: Optional[float],
    max_points: int = 2_000_000,
) -> None:
    """Export the colored point cloud as .ply (drop camera glyphs).

    Uses the same percentile filter predictions_to_glb applies, then writes a
    plain trimesh.PointCloud. Caps the result at `max_points` so the browser
    can actually render it — the PLYLoader + WebGL combo struggles beyond
    ~5M points and a typical full reconstruction easily hits 10M+.
    """
    import trimesh

    world_points = np.asarray(predictions["world_points"])
    conf = np.asarray(predictions["world_points_conf"])
    images = predictions.get("images")

    if world_points.ndim == 5 and world_points.shape[0] == 1:
        world_points = world_points[0]
    if conf.ndim == 4 and conf.shape[0] == 1:
        conf = conf[0]

    pts = world_points.reshape(-1, 3)
    c = conf.reshape(-1)
    thresh = float(np.percentile(c, conf_percentile if conf_percentile is not None else config.conf_percentile))
    mask = (c >= thresh) & (c > 1e-5)

    if images is not None:
        arr = np.asarray(images)
        if arr.ndim == 5 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape[-1] != 3 and arr.ndim == 4:
            arr = np.transpose(arr, (0, 2, 3, 1))
        colors = (arr.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    else:
        colors = np.full((pts.shape[0], 3), 200, dtype=np.uint8)

    pts_m = pts[mask]
    cols_m = colors[mask]

    if pts_m.shape[0] > max_points:
        step = int(pts_m.shape[0] // max_points) + 1
        pts_m = pts_m[::step]
        cols_m = cols_m[::step]
        log.info(
            "ply export downsampled from %d → %d points (step=%d)",
            int(mask.sum()),
            pts_m.shape[0],
            step,
        )

    pc = trimesh.PointCloud(vertices=pts_m, colors=cols_m)
    pc.export(str(out))


async def _heartbeat(
    job_id: str,
    label: str,
    publish,
    interval: float = 5.0,
):
    """Emit a 'still working' event every `interval` seconds until cancelled.

    Long export sub-steps (sky-segmentation over hundreds of frames, GLB
    serialization of large scenes) can go quiet for a minute+ which looks
    like a stall. This keeps the status strip's latest-message fresh.
    """
    start = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        elapsed = time.monotonic() - start
        await publish(
            JobEvent(
                job_id=job_id,
                stage="export",
                message=f"{label} · still working ({elapsed:.0f}s)",
            )
        )


async def export_reconstruction(
    job_id: str,
    frames_dir: Path,
    artifacts_dir: Path,
    predictions: dict,
    config: JobConfig,
    publish,
) -> dict[str, Path]:
    """Produce the initial artifacts for a completed inference run:
    GLB (textured points + camera glyphs), PLY (colored points only).

    Each long sub-step runs under capture_stdio so lingbot-map's internal
    tqdm progress flows into the event stream, and under a heartbeat task
    so the UI sees 'still working' ticks even when sub-steps are quiet.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()

    async def _run_stage(label: str, fn):
        await publish(JobEvent(job_id=job_id, stage="export", message=label))
        hb = asyncio.create_task(_heartbeat(job_id, label, publish))
        try:
            def _wrapped():
                with capture_stdio(job_id, publish, "export", loop):
                    return fn()
            return await asyncio.to_thread(_wrapped)
        finally:
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass

    if config.mask_sky:
        await _run_stage(
            "applying sky segmentation",
            lambda: _apply_sky_mask_to_predictions(predictions, frames_dir),
        )

    def _build_glb() -> Path:
        scene = _scene_from_predictions(predictions, config)
        path = artifacts_dir / "reconstruction.glb"
        scene.export(str(path))
        return path

    def _build_ply() -> Path:
        path = artifacts_dir / "reconstruction.ply"
        _export_ply_pointcloud(predictions, config, path, conf_percentile=None)
        return path

    glb_path = await _run_stage("writing GLB", _build_glb)
    ply_path = await _run_stage("writing PLY", _build_ply)

    # Save the camera trajectory so the viewer can draw the path and play
    # it back. extrinsic is c2w (S, 3, 4) — translation is the last column.
    def _write_camera_path() -> Path:
        from scipy.spatial.transform import Rotation

        ext = np.asarray(predictions.get("extrinsic"))
        if ext.ndim == 4 and ext.shape[0] == 1:
            ext = ext[0]
        if ext.ndim != 3 or ext.shape[-2:] != (3, 4):
            raise RuntimeError(f"unexpected extrinsic shape: {ext.shape}")

        positions = ext[:, :, 3]                # (S, 3)
        rot_mats = ext[:, :, :3]                # (S, 3, 3)
        quats = Rotation.from_matrix(rot_mats).as_quat()  # (S, 4) xyzw

        poses = [
            {
                "position": [float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])],
                "quaternion": [float(quats[i, 0]), float(quats[i, 1]), float(quats[i, 2]), float(quats[i, 3])],
            }
            for i in range(positions.shape[0])
        ]
        out = artifacts_dir / "camera_path.json"
        out.write_text(json.dumps({"fps": float(config.fps), "poses": poses}))
        return out

    try:
        cam_path = await asyncio.to_thread(_write_camera_path)
        await publish(
            JobEvent(
                job_id=job_id,
                stage="artifact",
                message=f"camera path saved: {len(json.loads(cam_path.read_text())['poses'])} poses",
                data={"name": cam_path.name, "kind": "camera_path"},
            )
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to write camera_path.json: %s", exc)

    # Clean up live-preview partial PLYs now that the real reconstruction is
    # in place. Also drop their artifact events from the client's mental model
    # by emitting a "partial_cleanup" event the frontend can act on.
    removed: list[str] = []
    for p in artifacts_dir.glob("partial_*.ply"):
        try:
            p.unlink()
            removed.append(p.name)
        except OSError:
            pass
    if removed:
        await publish(
            JobEvent(
                job_id=job_id,
                stage="artifact",
                message=f"cleaned up {len(removed)} partial snapshot(s)",
                data={"kind": "partial_cleanup", "removed": removed},
            )
        )

    await publish(
        JobEvent(
            job_id=job_id,
            stage="artifact",
            message="reconstruction ready",
            data={"glb": glb_path.name, "ply": ply_path.name},
            progress=1.0,
        )
    )
    return {"glb": glb_path, "ply": ply_path}


async def reexport(
    job_id: str,
    artifacts_dir: Path,
    predictions: dict,
    config: JobConfig,
    fmt: Format,
    *,
    conf_percentile: Optional[float],
    show_cam: Optional[bool],
    mask_sky: Optional[bool],
    mask_black_bg: Optional[bool],
    mask_white_bg: Optional[bool],
    publish,
) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = f"reexport_p{int(conf_percentile if conf_percentile is not None else config.conf_percentile)}"
    out = artifacts_dir / f"{stem}.{fmt}"

    def _build() -> Path:
        scene = _scene_from_predictions(
            predictions,
            config,
            conf_percentile=conf_percentile,
            show_cam=show_cam,
            mask_sky=mask_sky,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
        )
        if fmt == "glb":
            scene.export(str(out))
        elif fmt == "ply":
            _export_ply_pointcloud(predictions, config, out, conf_percentile=conf_percentile)
        elif fmt == "obj":
            scene.export(str(out))
        return out

    await publish(JobEvent(job_id=job_id, stage="export", message=f"re-exporting {fmt}..."))
    path = await asyncio.to_thread(_build)
    await publish(
        JobEvent(
            job_id=job_id,
            stage="artifact",
            message=f"re-export ready: {path.name}",
            data={"name": path.name, "format": fmt},
        )
    )
    return path
