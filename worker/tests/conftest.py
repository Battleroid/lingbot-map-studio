"""Shared pytest fixtures for worker-side smoke tests.

These tests exercise the plumbing that's CPU-safe — the simulated SLAM
tracker, the simulated gsplat trainer, the FPV preprocessing stages, and
the discriminated-union config schema. Real CUDA backends (DROID /
MASt3R / DPVO / MonoGS CUDA, real gsplat kernels) aren't installed in CI
and live behind a GPU-tagged runner; this suite is designed to run on any
machine with only numpy + opencv available.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest


# Make `import app.*` work without installing the package — the worker uses
# `[tool.setuptools.packages.find]` at build time but the test suite runs
# straight out of the source tree.
_WORKER_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_ROOT))


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point `settings.data_dir` / `settings.models_dir` at a tmp tree.

    The Settings object is instantiated at import time, so we patch the
    attributes in-place rather than reloading the module. Also sets env
    vars so any process that reads them (ffmpeg subprocesses, worker subs)
    picks up the same dirs.
    """
    data = tmp_path / "data"
    models = tmp_path / "models"
    data.mkdir()
    models.mkdir()

    from app.config import settings

    monkeypatch.setattr(settings, "data_dir", data)
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setenv("LINGBOT_DATA_DIR", str(data))
    monkeypatch.setenv("LINGBOT_MODELS_DIR", str(models))
    settings.ensure_dirs()
    yield data


@pytest.fixture
def synthetic_frames(tmp_path: Path) -> Path:
    """Write a tiny sequence of PNG frames that mimic a short clip.

    The scene is a 96×64 grid with a moving bright square + global colour
    tint so the preproc stages (colour-norm, deblur, keyframe scoring,
    rolling-shutter) all have something to chew on. Small enough that the
    whole suite runs in <1s on a laptop.
    """
    import cv2

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    h, w = 64, 96
    n = 16
    for i in range(n):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Slight green tint — exercises grey-world WB.
        img[..., 1] = 60
        # Sweeping square approximates motion for optical-flow-based stages.
        x0 = int((w - 16) * (i / (n - 1))) if n > 1 else 0
        img[16:48, x0:x0 + 16] = (200, 200, 200)
        # Faint noise so Laplacian variance isn't identical across frames.
        noise = np.random.default_rng(i).integers(0, 12, size=(h, w, 3), dtype=np.uint8)
        img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        cv2.imwrite(str(frames_dir / f"{i:06d}.png"), img)
    return frames_dir


@pytest.fixture
def captured_events() -> tuple[list, "CapturePublish"]:
    """Collector for the async publish fn that preproc/processor stages call."""
    events: list = []

    class CapturePublish:
        async def __call__(self, event):
            events.append(event)
            return event

    return events, CapturePublish()


@pytest.fixture(autouse=True)
def _silence_matplotlib_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force Agg backend so stray imports don't open a display."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # Keep OpenCV from swallowing CI runners with thread pools.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
