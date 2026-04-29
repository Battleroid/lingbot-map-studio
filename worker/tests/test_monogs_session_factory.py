"""Pin the MonoGS session auto-select. Same shape as the SLAM factory
tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace


def test_select_simulated_when_torch_missing(monkeypatch):
    from app.processors.gsplat.monogs import (
        _MonogsSession,
        select_session_cls,
    )

    monkeypatch.setitem(sys.modules, "torch", None)
    assert select_session_cls() is _MonogsSession


def test_select_simulated_when_cuda_unavailable(monkeypatch):
    from app.processors.gsplat.monogs import (
        _MonogsSession,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert select_session_cls() is _MonogsSession


def test_select_simulated_when_monogs_missing(monkeypatch):
    from app.processors.gsplat.monogs import (
        _MonogsSession,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # The factory probes `gaussian_splatting` first (MonoGS's primary
    # import path now that we install via PYTHONPATH instead of pip)
    # and falls back to `monogs` for legacy forks. Both must be missing
    # for the simulated session to win.
    monkeypatch.setitem(sys.modules, "gaussian_splatting", None)
    monkeypatch.setitem(sys.modules, "monogs", None)
    assert select_session_cls() is _MonogsSession
