"""Pin the DPVO session resolver. Same shape as the MASt3R / DROID
factory tests — strict-no-fallback in production, simulated
fallback only via the live-capture wrapper."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_select_raises_when_torch_missing(monkeypatch):
    from app.processors.slam.dpvo import DpvoUnavailableError, select_session_cls

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(DpvoUnavailableError, match="torch is not installed"):
        select_session_cls()


def test_select_raises_when_cuda_unavailable(monkeypatch):
    from app.processors.slam.dpvo import DpvoUnavailableError, select_session_cls

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(DpvoUnavailableError, match="torch.cuda.is_available"):
        select_session_cls()


def test_select_raises_when_dpvo_missing(monkeypatch):
    from app.processors.slam.dpvo import DpvoUnavailableError, select_session_cls

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "dpvo", None)
    monkeypatch.setitem(sys.modules, "dpvo.dpvo", None)
    with pytest.raises(DpvoUnavailableError, match="DPVO source"):
        select_session_cls()


def test_processor_make_session_raises_in_production(monkeypatch):
    from app.processors.slam.dpvo import DpvoUnavailableError, DpvoProcessor

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(DpvoUnavailableError):
        DpvoProcessor()._make_session(ctx=None)


def test_live_capture_wrapper_falls_back_to_simulated(monkeypatch):
    from app.processors.slam.dpvo import _DpvoSession
    from app.processors.slam.live_session import _resolve_cls

    monkeypatch.setitem(sys.modules, "torch", None)
    assert _resolve_cls("dpvo") is _DpvoSession
