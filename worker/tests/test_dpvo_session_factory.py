"""Pin the DPVO session auto-select. Same shape as the MASt3R / DROID
factory tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace


def test_select_simulated_when_torch_missing(monkeypatch):
    from app.processors.slam.dpvo import _DpvoSession, select_session_cls

    monkeypatch.setitem(sys.modules, "torch", None)
    assert select_session_cls() is _DpvoSession


def test_select_simulated_when_cuda_unavailable(monkeypatch):
    from app.processors.slam.dpvo import _DpvoSession, select_session_cls

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert select_session_cls() is _DpvoSession


def test_select_simulated_when_dpvo_missing(monkeypatch):
    from app.processors.slam.dpvo import _DpvoSession, select_session_cls

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "dpvo", None)
    monkeypatch.setitem(sys.modules, "dpvo.dpvo", None)
    assert select_session_cls() is _DpvoSession


def test_processor_make_session_uses_factory():
    from app.processors.slam.dpvo import _DpvoSession, DpvoProcessor

    session = DpvoProcessor()._make_session(ctx=None)
    assert isinstance(session, _DpvoSession)
