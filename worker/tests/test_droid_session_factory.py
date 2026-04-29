"""Pin the DROID-SLAM session auto-select. Same shape as Phase 2's
test_mast3r_session_factory.py."""

from __future__ import annotations

import sys
from types import SimpleNamespace


def test_select_simulated_when_torch_missing(monkeypatch):
    from app.processors.slam.droid_slam import (
        _DroidSlamSession,
        select_session_cls,
    )

    monkeypatch.setitem(sys.modules, "torch", None)
    assert select_session_cls() is _DroidSlamSession


def test_select_simulated_when_cuda_unavailable(monkeypatch):
    from app.processors.slam.droid_slam import (
        _DroidSlamSession,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert select_session_cls() is _DroidSlamSession


def test_select_simulated_when_droid_missing(monkeypatch):
    from app.processors.slam.droid_slam import (
        _DroidSlamSession,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "droid_slam", None)
    assert select_session_cls() is _DroidSlamSession


def test_processor_make_session_uses_factory():
    from app.processors.slam.droid_slam import (
        _DroidSlamSession,
        DroidSlamProcessor,
    )

    processor = DroidSlamProcessor()
    session = processor._make_session(ctx=None)
    assert isinstance(session, _DroidSlamSession)
