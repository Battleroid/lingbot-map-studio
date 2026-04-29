"""Pin the trainer auto-select logic. Three branches to cover:

  * torch isn't installed → simulated.
  * torch is installed but CUDA isn't available → simulated.
  * torch + CUDA + gsplat all importable → CUDA trainer.

CI machines have neither torch nor CUDA, so by default `select_trainer_cls()`
returns the simulated class. We monkeypatch torch into sys.modules to
exercise the other branches without needing a GPU.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_select_trainer_simulated_when_torch_missing(monkeypatch):
    """No torch on PATH → simulated trainer (covers CPU dev boxes that
    haven't installed the worker-gs deps)."""
    from app.processors.gsplat.trainer import (
        SimulatedSplatTrainer,
        select_trainer_cls,
    )

    monkeypatch.setitem(sys.modules, "torch", None)
    assert select_trainer_cls() is SimulatedSplatTrainer


def test_select_trainer_simulated_when_cuda_unavailable(monkeypatch):
    """Torch importable but no GPU passthrough → simulated trainer.
    Covers a worker-gs container without `runtime: nvidia`."""
    from app.processors.gsplat.trainer import (
        SimulatedSplatTrainer,
        select_trainer_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert select_trainer_cls() is SimulatedSplatTrainer


def test_select_trainer_simulated_when_gsplat_missing(monkeypatch):
    """Torch + CUDA both fine but `gsplat` package not installed →
    simulated trainer. Covers a worker-gs image that still has the
    pre-Phase-1 Dockerfile."""
    from app.processors.gsplat.trainer import (
        SimulatedSplatTrainer,
        select_trainer_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # Force a re-import miss for gsplat.rasterization.
    monkeypatch.setitem(sys.modules, "gsplat", None)
    monkeypatch.setitem(sys.modules, "gsplat.rasterization", None)
    assert select_trainer_cls() is SimulatedSplatTrainer


def test_processor_constructor_invokes_auto_select(monkeypatch):
    """GsplatProcessor.__init__ should rebind the instance's trainer_cls
    to whatever select_trainer_cls() returns. CPU CI: stays simulated."""
    from app.processors.gsplat.trainer import (
        GsplatProcessor,
        SimulatedSplatTrainer,
    )

    p = GsplatProcessor()
    # On CI without torch/gsplat the auto-select returns the simulated
    # class — this just pins that the constructor calls into the factory
    # at all (instead of leaving the class-attr default in place silently).
    assert p.trainer_cls is SimulatedSplatTrainer


@pytest.mark.skipif(
    True,
    reason="GPU-only smoke; flip to runtime cuda check on a GPU runner",
)
def test_select_trainer_cuda_when_everything_available():
    """Real-GPU smoke. Skipped by default — flip the marker on a runner
    with torch.cuda + gsplat installed to verify the happy path."""
    from app.processors.gsplat.trainer import select_trainer_cls

    cls = select_trainer_cls()
    assert cls.__name__ == "GsplatCudaTrainer"
