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


def _value_present(predictions: dict, key: str) -> bool:
    """True iff `key` is in `predictions` AND its value isn't None.

    Bare `key in predictions` would say True for `predictions[key] = None`,
    which the upstream `predictions_to_glb` happily accepts and then
    crashes downstream on `.reshape(-1, 3)`. Treat None as absent so the
    synthesis path takes over.
    """
    return predictions.get(key) is not None


def _predictions_summary(predictions: dict) -> dict:
    """`{key: shape_or_type_str}` — for debug messages so users can see
    what their model variant actually emitted when synthesis can't proceed."""
    summary: dict = {}
    for k, v in predictions.items():
        if v is None:
            summary[k] = "None"
        elif hasattr(v, "shape"):
            summary[k] = str(tuple(v.shape))
        else:
            summary[k] = type(v).__name__
    return summary


def _ensure_world_points_for_export(predictions: dict) -> None:
    """Mutate `predictions` so the upstream `predictions_to_glb` has at
    least one of `world_points` / `world_points_from_depth` to consume.

    Some lingbot-map model variants emit `depth` but skip `world_points`
    in their inference output (e.g. certain windowed paths). Upstream's
    `predictions_to_glb` then logs `world_points not found, falling
    back to depth-based points` and immediately KeyError's on
    `world_points_from_depth` because the fallback branch expects that
    derived field, not raw depth.

    Compute it ourselves from `depth + intrinsic + extrinsic` and inject
    so the upstream's depth-fallback branch finds what it needs. No-op
    when either key is already present (real `world_points` always wins).

    Status messages go through `print` (not `log.*`) so they reach the
    user via `capture_stdio` — earlier versions used `log.warning` which
    silently routed nowhere visible and left users staring at a bare
    `KeyError: 'world_points_from_depth'` from the upstream library.
    """
    if _value_present(predictions, "world_points") or _value_present(
        predictions, "world_points_from_depth"
    ):
        return

    # `or` on numpy arrays raises ValueError; use explicit None checks.
    depth = predictions.get("depth")
    if depth is None:
        depth = predictions.get("depth_map")
    intrinsic = predictions.get("intrinsic")
    extrinsic = predictions.get("extrinsic")
    if depth is None or intrinsic is None or extrinsic is None:
        print(
            f"[lingbot] cannot synthesise world_points_from_depth — "
            f"need depth+intrinsic+extrinsic, have {_predictions_summary(predictions)}"
        )
        return

    depth_np = np.asarray(depth)
    K = np.asarray(intrinsic)
    E = np.asarray(extrinsic)

    # Normalise shapes — upstream produces (S, H, W) or (S, H, W, 1) for
    # depth, (S, 3, 3) for K, and (S, 3, 4) c2w for E. _postprocess
    # already strips the leading batch dim.
    if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
        depth_np = depth_np.squeeze(-1)
    if depth_np.ndim != 3:
        print(
            f"[lingbot] depth has unexpected shape {depth_np.shape}; "
            f"skipping world_points synthesis"
        )
        return

    S, H, W = depth_np.shape
    if K.shape != (S, 3, 3) or E.shape[:1] != (S,):
        print(
            f"[lingbot] intrinsic/extrinsic shape mismatch "
            f"(K={K.shape}, E={E.shape}, depth={depth_np.shape}); "
            f"skipping world_points synthesis"
        )
        return

    # Pixel grid (H, W). Same for every frame.
    ys, xs = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )

    out = np.empty((S, H, W, 3), dtype=np.float32)
    for s in range(S):
        z = depth_np[s].astype(np.float32)
        fx, fy = float(K[s, 0, 0]), float(K[s, 1, 1])
        cx, cy = float(K[s, 0, 2]), float(K[s, 1, 2])

        # Camera-space (right/up/forward) coordinates.
        cam_x = (xs - cx) * z / max(fx, 1e-9)
        cam_y = (ys - cy) * z / max(fy, 1e-9)
        cam_z = z

        cam_xyz = np.stack([cam_x, cam_y, cam_z], axis=-1)  # (H, W, 3)

        # c2w extrinsic: world = R · cam + t
        E_s = E[s]
        if E_s.shape == (3, 4):
            R = E_s[:, :3]
            t = E_s[:, 3]
        elif E_s.shape == (4, 4):
            R = E_s[:3, :3]
            t = E_s[:3, 3]
        else:
            print(
                f"[lingbot] extrinsic[{s}] shape {E_s.shape} unrecognised; "
                f"skipping frame"
            )
            out[s] = 0.0
            continue
        out[s] = cam_xyz @ R.T.astype(np.float32) + t.astype(np.float32)

    predictions["world_points_from_depth"] = out
    # Also synthesise a placeholder confidence map if the model didn't emit one.
    # Upstream's depth-fallback branch reads `depth_conf`; without it we'd
    # crash on `np.ones_like(pred_world_points[..., 0])` later if any other
    # callsite expects it. Use ones so every point passes percentile filtering.
    if not _value_present(predictions, "depth_conf"):
        predictions["depth_conf"] = np.ones((S, H, W), dtype=np.float32)
    print(
        f"[lingbot] synthesised world_points_from_depth (S={S}, H={H}, W={W}) "
        f"for export"
    )


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

    _ensure_world_points_for_export(predictions)
    if not (
        _value_present(predictions, "world_points")
        or _value_present(predictions, "world_points_from_depth")
    ):
        # Replace the opaque `KeyError: 'world_points_from_depth'` from upstream
        # with a clear message naming the actual problem (the model checkpoint
        # didn't emit world_points OR depth, so we have no geometry to export).
        raise RuntimeError(
            "lingbot export: model output has neither `world_points` nor "
            "`depth` — cannot build GLB. The checkpoint may have point/depth "
            f"heads disabled. Predictions summary: {_predictions_summary(predictions)}"
        )
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

    # Same model-variant gotcha as _scene_from_predictions: some
    # inference paths emit `depth` but no `world_points`. Synthesise from
    # depth + K + E if needed; upstream's PLY export gets the same
    # treatment as the GLB path.
    _ensure_world_points_for_export(predictions)
    wp = predictions.get("world_points")
    if wp is None:
        wp = predictions.get("world_points_from_depth")
    if wp is None or np.asarray(wp).size == 0:
        # Truly unrecoverable — depth was missing too. Write an empty
        # PLY so downstream callers don't error on a missing file.
        print("[lingbot] ply export: no world_points available; writing empty cloud")
        cloud = trimesh.PointCloud(np.zeros((0, 3), dtype=np.float32))
        cloud.export(str(out))
        return
    world_points = np.asarray(wp)
    # When we synthesised from depth the model didn't emit world_points_conf
    # either; fall back to depth_conf, then to ones. Without this fallback the
    # PLY path would KeyError after the GLB path was already fixed.
    conf_arr = predictions.get("world_points_conf")
    if conf_arr is None:
        conf_arr = predictions.get("depth_conf")
    if conf_arr is None:
        conf_arr = np.ones(world_points.shape[:-1], dtype=np.float32)
    conf = np.asarray(conf_arr)
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
    # Also record each frame's vertical FOV (computed from intrinsic fy +
    # image height) so the playback camera matches the source footage; the
    # default three.js 45° FOV is too narrow for most action/drone cams.
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

        # FOV from intrinsic: fov_y = 2 * atan(H/2 / fy). Image height comes
        # from predictions["images"] shape (S, 3, H, W) or (S, H, W, 3).
        fov_ys: list[float] | None = None
        try:
            intr = np.asarray(predictions.get("intrinsic"))
            imgs = np.asarray(predictions.get("images"))
            if intr is not None and imgs is not None:
                if intr.ndim == 4 and intr.shape[0] == 1:
                    intr = intr[0]
                # Image height
                if imgs.ndim == 4 and imgs.shape[1] == 3:
                    img_h = int(imgs.shape[2])
                elif imgs.ndim == 4 and imgs.shape[-1] == 3:
                    img_h = int(imgs.shape[1])
                else:
                    img_h = None
                if intr.ndim == 3 and intr.shape[-2:] == (3, 3) and img_h:
                    fy = intr[:, 1, 1]
                    fov_ys = [
                        float(np.degrees(2 * np.arctan((img_h / 2) / max(1e-6, f))))
                        for f in fy
                    ]
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to compute per-frame fov: %s", exc)

        poses = []
        for i in range(positions.shape[0]):
            pose = {
                "position": [
                    float(positions[i, 0]),
                    float(positions[i, 1]),
                    float(positions[i, 2]),
                ],
                "quaternion": [
                    float(quats[i, 0]),
                    float(quats[i, 1]),
                    float(quats[i, 2]),
                    float(quats[i, 3]),
                ],
            }
            if fov_ys and i < len(fov_ys):
                pose["fov_y_deg"] = fov_ys[i]
            poses.append(pose)

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
