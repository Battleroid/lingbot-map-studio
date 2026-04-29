"""Pin the MASt3R-SLAM session auto-select. Same shape as the gsplat
trainer-factory tests in test_gsplat_trainer_factory.py — covers:

  * torch isn't installed → simulated session.
  * torch installed but CUDA unavailable → simulated.
  * torch + cuda but `mast3r_slam` package missing → simulated.
  * Mast3rSlamProcessor._make_session() invokes the factory.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace


def test_select_simulated_when_torch_missing(monkeypatch):
    from app.processors.slam.mast3r_slam import (
        _Mast3rSlamSession,
        select_session_cls,
    )

    monkeypatch.setitem(sys.modules, "torch", None)
    assert select_session_cls() is _Mast3rSlamSession


def test_select_simulated_when_cuda_unavailable(monkeypatch):
    from app.processors.slam.mast3r_slam import (
        _Mast3rSlamSession,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert select_session_cls() is _Mast3rSlamSession


def test_select_simulated_when_mast3r_missing(monkeypatch):
    from app.processors.slam.mast3r_slam import (
        _Mast3rSlamSession,
        select_session_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "mast3r_slam", None)
    monkeypatch.setitem(sys.modules, "mast3r_slam.tracker", None)
    assert select_session_cls() is _Mast3rSlamSession


def test_processor_make_session_uses_factory():
    from app.processors.slam.mast3r_slam import (
        _Mast3rSlamSession,
        Mast3rSlamProcessor,
    )

    processor = Mast3rSlamProcessor()
    session = processor._make_session(ctx=None)  # ctx unused on simulated path
    # CI without torch/cuda/mast3r_slam returns the simulated session.
    assert isinstance(session, _Mast3rSlamSession)
