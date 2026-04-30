"""Pin the DROID-SLAM session resolver. Same shape as
test_mast3r_session_factory.py — strict-no-fallback in production,
simulated fallback only via the live-capture wrapper."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_select_raises_when_torch_missing(monkeypatch):
    from app.processors.slam.droid_slam import (
        DroidSlamUnavailableError,
        select_session_cls,
    )

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(DroidSlamUnavailableError, match="torch is not installed"):
        select_session_cls()


def test_select_raises_when_cuda_unavailable(monkeypatch):
    from app.processors.slam.droid_slam import (
        DroidSlamUnavailableError,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(DroidSlamUnavailableError, match="torch.cuda.is_available"):
        select_session_cls()


def test_select_raises_when_droid_missing(monkeypatch):
    from app.processors.slam.droid_slam import (
        DroidSlamUnavailableError,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "droid_slam", None)
    with pytest.raises(DroidSlamUnavailableError, match="DROID-SLAM source"):
        select_session_cls()


def test_processor_make_session_raises_in_production(monkeypatch):
    from app.processors.slam.droid_slam import (
        DroidSlamUnavailableError,
        DroidSlamProcessor,
    )

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(DroidSlamUnavailableError):
        DroidSlamProcessor()._make_session(ctx=None)


def test_live_capture_wrapper_falls_back_to_simulated(monkeypatch):
    from app.processors.slam.droid_slam import _DroidSlamSession
    from app.processors.slam.live_session import _resolve_cls

    monkeypatch.setitem(sys.modules, "torch", None)
    assert _resolve_cls("droid_slam") is _DroidSlamSession
