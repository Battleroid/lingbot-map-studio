"""Pin the MonoGS session auto-select.

The simulated fallback was removed in the "no fake gsplat output"
fix — `select_session_cls()` raises `MonogsSessionUnavailableError`
when the real CUDA stack is missing instead of silently returning
`_MonogsSession`. The simulated class still exists for tests + for
the live-capture preview path (which catches the error and uses it
explicitly), but the production resolver itself refuses to return
it.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_select_raises_when_torch_missing(monkeypatch):
    from app.processors.gsplat.monogs import (
        MonogsSessionUnavailableError,
        select_session_cls,
    )

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(MonogsSessionUnavailableError, match="torch is not installed"):
        select_session_cls()


def test_select_raises_when_cuda_unavailable(monkeypatch):
    from app.processors.gsplat.monogs import (
        MonogsSessionUnavailableError,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(MonogsSessionUnavailableError, match="torch.cuda.is_available"):
        select_session_cls()


def test_select_raises_when_monogs_missing(monkeypatch):
    from app.processors.gsplat.monogs import (
        MonogsSessionUnavailableError,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # Both probes (`gaussian_splatting` primary, `monogs` legacy) must
    # fail before the resolver gives up.
    monkeypatch.setitem(sys.modules, "gaussian_splatting", None)
    monkeypatch.setitem(sys.modules, "monogs", None)
    with pytest.raises(MonogsSessionUnavailableError, match="MonoGS source"):
        select_session_cls()


def test_live_capture_path_still_falls_back_to_simulated(monkeypatch):
    """The live-capture resolver wraps the strict gsplat selector and
    returns `_MonogsSession` when the real stack is unavailable. The
    api container has no GPU but still needs *some* SLAM session for
    the live preview; the captured frames are re-processed by the
    real backend in worker-gs after stop, so the preview being
    approximate doesn't taint the final output."""
    from app.processors.gsplat.monogs import _MonogsSession
    from app.processors.slam.live_session import _resolve_cls

    # Make the strict resolver fail.
    monkeypatch.setitem(sys.modules, "torch", None)
    cls = _resolve_cls("monogs")
    assert cls is _MonogsSession
