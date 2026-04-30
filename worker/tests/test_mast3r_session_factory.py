"""Pin the MASt3R-SLAM session resolver. The simulated fallback
was removed in the strict-no-simulated cleanup —
`select_session_cls()` now raises `Mast3rSlamUnavailableError` with
install instructions when its real CUDA stack isn't present. The
`_Mast3rSlamSession` placeholder still lives in the module so the
live-capture wrapper (which has no GPU regardless) can drive its
preview overlay; the worker-slam production path always either
returns the real CUDA session or fails loud.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_select_raises_when_torch_missing(monkeypatch):
    from app.processors.slam.mast3r_slam import (
        Mast3rSlamUnavailableError,
        select_session_cls,
    )

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(Mast3rSlamUnavailableError, match="torch is not installed"):
        select_session_cls()


def test_select_raises_when_cuda_unavailable(monkeypatch):
    from app.processors.slam.mast3r_slam import (
        Mast3rSlamUnavailableError,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(Mast3rSlamUnavailableError, match="torch.cuda.is_available"):
        select_session_cls()


def test_select_raises_when_mast3r_missing(monkeypatch):
    from app.processors.slam.mast3r_slam import (
        Mast3rSlamUnavailableError,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "mast3r_slam", None)
    monkeypatch.setitem(sys.modules, "mast3r_slam.tracker", None)
    with pytest.raises(Mast3rSlamUnavailableError, match="MASt3R-SLAM source"):
        select_session_cls()


def test_processor_make_session_raises_in_production(monkeypatch):
    """The processor's `_make_session` is what the worker-slam runner
    calls. When the real stack is missing it should propagate the
    Unavailable error so the runner marks the job failed with the
    install hint — not silently substitute a placeholder."""
    from app.processors.slam.mast3r_slam import (
        Mast3rSlamUnavailableError,
        Mast3rSlamProcessor,
    )

    monkeypatch.setitem(sys.modules, "torch", None)
    processor = Mast3rSlamProcessor()
    with pytest.raises(Mast3rSlamUnavailableError):
        processor._make_session(ctx=None)


def test_live_capture_wrapper_falls_back_to_simulated(monkeypatch):
    """The live-capture resolver in `live_session.py` catches the
    strict resolver's error and substitutes `_Mast3rSlamSession` so
    the preview keeps moving. The captured frames are queued for
    real reconstruction on the GPU worker after stop, so the
    preview being approximate doesn't taint final output."""
    from app.processors.slam.live_session import _resolve_cls
    from app.processors.slam.mast3r_slam import _Mast3rSlamSession

    monkeypatch.setitem(sys.modules, "torch", None)
    assert _resolve_cls("mast3r_slam") is _Mast3rSlamSession
