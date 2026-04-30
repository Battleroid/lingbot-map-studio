"""Pin the MonoGS batch wrapper's workspace + config shape.

The batch wrapper bridges upstream MonoGS's batch-only API (no
streaming `process_frame`) by building a TUM-shaped dataset on disk
and subprocess-running `slam.py`. These tests cover the pure-Python
half (workspace builder + config builder) so the format stays in sync
with upstream's TUMParser / load_dataset assumptions without needing
the actual CUDA stack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_intrinsics(w: int, h: int, fx: float | None = None) -> np.ndarray:
    fx = fx if fx is not None else w * 0.866
    return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float32)


def _write_dummy_frame(path: Path) -> None:
    # Smallest valid PNG so cv2.imread doesn't choke if anyone wires
    # the test up to actually decode. The wrapper itself only stat()s
    # / symlinks; never reads the bytes.
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00"
        b"\x3a\x7e\x9b\x55"
        b"\x00\x00\x00\nIDAT"
        b"\x78\x9c\x62\x00\x00\x00\x00\x05\x00\x01"
        b"\x0d\x0a\x2d\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_build_tum_workspace_emits_expected_files(tmp_path):
    from app.processors.gsplat.monogs_batch import _build_tum_workspace

    frames_src = tmp_path / "src"
    frames_src.mkdir()
    frame_paths = []
    for i in range(8):
        p = frames_src / f"frame_{i:06d}.png"
        _write_dummy_frame(p)
        frame_paths.append(p)

    workspace = tmp_path / "workspace"
    n = _build_tum_workspace(workspace, frame_paths, fps=10.0)
    assert n == 8

    # Three TUM-format text files at the workspace root.
    rgb_txt = (workspace / "rgb.txt").read_text(encoding="utf-8").splitlines()
    depth_txt = (workspace / "depth.txt").read_text(encoding="utf-8").splitlines()
    gt_txt = (workspace / "groundtruth.txt").read_text(encoding="utf-8").splitlines()

    # First few lines are # comment headers (auto-skipped by the
    # upstream TUMParser via np.loadtxt), then 8 entries each.
    rgb_data = [l for l in rgb_txt if not l.startswith("#") and l.strip()]
    depth_data = [l for l in depth_txt if not l.startswith("#") and l.strip()]
    gt_data = [l for l in gt_txt if not l.startswith("#") and l.strip()]
    assert len(rgb_data) == 8
    assert len(depth_data) == 8
    assert len(gt_data) == 8

    # rgb.txt: `<timestamp> rgb/<idx:06d>.png`
    parts = rgb_data[0].split()
    assert len(parts) == 2
    ts0 = float(parts[0])
    assert parts[1].startswith("rgb/")
    # Timestamps spaced by 1/fps.
    ts1 = float(rgb_data[1].split()[0])
    assert abs((ts1 - ts0) - 0.1) < 1e-6

    # groundtruth.txt: identity poses (tx ty tz qx qy qz qw = 0 0 0 0 0 0 1).
    gt_first = gt_data[0].split()
    assert len(gt_first) == 8  # ts + 7
    assert [float(x) for x in gt_first[1:]] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # rgb/ and depth/ directories actually exist + populated.
    assert (workspace / "rgb").is_dir()
    assert (workspace / "depth" / "placeholder.png").exists()
    rgb_files = sorted((workspace / "rgb").iterdir())
    assert len(rgb_files) == 8


def test_build_config_shape_matches_upstream_tum_template():
    """The synthetic config has to satisfy upstream MonoGS's
    `model_params` / `opt_params` / `pipeline_params` keys + the
    `Dataset.Calibration` block that `MonocularDataset.__init__`
    reads. Pin the field set here so an upstream refactor that adds
    a required key surfaces as a CI failure rather than a runtime
    KeyError."""
    from app.processors.gsplat.monogs_batch import _build_config

    K = _make_intrinsics(640, 480, fx=600.0)
    cfg = _build_config(
        intrinsics=K,
        image_shape=(480, 640),
        dataset_path=Path("/scratch/dataset"),
        save_dir=Path("/scratch/output"),
    )

    # Top-level keys upstream's `slam.py` and `dataset.py` read off.
    for key in ("Results", "Dataset", "Training", "opt_params", "model_params", "pipeline_params"):
        assert key in cfg, f"missing top-level config key: {key}"

    # Dataset block: type=tum, sensor_type=monocular, no depth_scale
    # (so MonocularDataset.has_depth=False, never reads our placeholder
    # depth files).
    assert cfg["Dataset"]["type"] == "tum"
    assert cfg["Dataset"]["sensor_type"] == "monocular"
    assert cfg["Dataset"]["dataset_path"] == "/scratch/dataset"
    assert "depth_scale" not in cfg["Dataset"]["Calibration"]
    assert cfg["Dataset"]["Calibration"]["fx"] == 600.0
    assert cfg["Dataset"]["Calibration"]["width"] == 640
    assert cfg["Dataset"]["Calibration"]["height"] == 480
    assert cfg["Dataset"]["Calibration"]["distorted"] is False

    # Training block: required for FrontEnd.set_hyperparams. Spot-check.
    for key in (
        "tracking_itr_num",
        "kf_interval",
        "window_size",
        "single_thread",
        "kf_translation",
        "rgb_boundary_threshold",
        "spherical_harmonics",
    ):
        assert key in cfg["Training"], f"missing Training.{key}"

    # opt_params block: BackEnd.set_hyperparams + GaussianModel.training_setup.
    for key in (
        "iterations",
        "position_lr_init",
        "feature_lr",
        "opacity_lr",
        "scaling_lr",
        "rotation_lr",
        "densify_grad_threshold",
        "lambda_dssim",
    ):
        assert key in cfg["opt_params"], f"missing opt_params.{key}"

    # Results.use_gui must be False — we have no display in the worker.
    assert cfg["Results"]["use_gui"] is False
    assert cfg["Results"]["use_wandb"] is False
    assert cfg["Results"]["save_dir"] == "/scratch/output"


@pytest.mark.asyncio
async def test_run_monogs_batch_raises_when_upstream_missing(tmp_path):
    """When `MONOGS_ROOT/slam.py` doesn't exist (e.g. CPU-only test
    host), the wrapper raises `MonogsBatchUnavailableError` before
    spawning a subprocess — the processor catches this and surfaces
    it as a level=error event with the install hint."""
    from app.processors.gsplat.monogs_batch import (
        MonogsBatchUnavailableError,
        run_monogs_batch,
    )

    frames_src = tmp_path / "src"
    frames_src.mkdir()
    frame_paths = []
    for i in range(5):
        p = frames_src / f"frame_{i:06d}.png"
        _write_dummy_frame(p)
        frame_paths.append(p)

    async def publish(_event):
        return None

    with pytest.raises(MonogsBatchUnavailableError, match="slam.py not found"):
        await run_monogs_batch(
            job_id="test-batch-001",
            frame_paths=frame_paths,
            intrinsics=_make_intrinsics(64, 48),
            image_shape=(48, 64),
            workspace_root=tmp_path / "ws",
            publish=publish,
            monogs_root=tmp_path / "no-such-monogs",
        )


@pytest.mark.asyncio
async def test_run_monogs_batch_raises_when_too_few_frames(tmp_path):
    """MonoGS's monocular initialiser needs more than a couple of
    overlapping views; we bail early rather than launching a subprocess
    that's guaranteed to fail with an opaque error."""
    from app.processors.gsplat.monogs_batch import (
        MonogsBatchUnavailableError,
        run_monogs_batch,
    )

    # Stub a fake `slam.py` so we get past the existence check and hit
    # the frame-count guard.
    fake_root = tmp_path / "fake-monogs"
    fake_root.mkdir()
    (fake_root / "slam.py").write_text("# stub")

    frames_src = tmp_path / "src"
    frames_src.mkdir()
    frame_paths = [frames_src / "f0.png"]
    _write_dummy_frame(frame_paths[0])

    async def publish(_event):
        return None

    with pytest.raises(MonogsBatchUnavailableError, match="too few frames"):
        await run_monogs_batch(
            job_id="test-batch-002",
            frame_paths=frame_paths,
            intrinsics=_make_intrinsics(64, 48),
            image_shape=(48, 64),
            workspace_root=tmp_path / "ws2",
            publish=publish,
            monogs_root=fake_root,
        )
