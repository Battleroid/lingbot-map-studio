"""Pin the trainer auto-select logic.

The simulated fallback was removed in the "no fake gsplat output"
fix — `select_trainer_cls()` now raises
`GsplatTrainerUnavailableError` with install instructions whenever
the real CUDA stack isn't present. The simulated class still lives
in the module so tests can drive the pipeline on a CPU box (by
pinning `processor.trainer_cls = SimulatedSplatTrainer` directly),
but the *resolver* refuses to return it.

CI machines have neither torch nor CUDA, so the resolver is the
"raise" path. The CUDA-available branch is GPU-only and stays
skipped by default.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_select_trainer_raises_when_torch_missing(monkeypatch):
    """No torch on PATH → loud error (covers CPU dev boxes that
    haven't installed the worker-gs deps)."""
    from app.processors.gsplat.trainer import (
        GsplatTrainerUnavailableError,
        select_trainer_cls,
    )

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(GsplatTrainerUnavailableError, match="torch is not installed"):
        select_trainer_cls()


def test_select_trainer_raises_when_cuda_unavailable(monkeypatch):
    """Torch importable but no GPU passthrough → loud error.
    Covers a worker-gs container without `runtime: nvidia`."""
    from app.processors.gsplat.trainer import (
        GsplatTrainerUnavailableError,
        select_trainer_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(
        GsplatTrainerUnavailableError, match="torch.cuda.is_available"
    ):
        select_trainer_cls()


def test_select_trainer_raises_when_gsplat_missing(monkeypatch):
    """Torch + CUDA both fine but `gsplat` package not installed →
    loud error. Covers a worker-gs image that still has the
    pre-Phase-1 Dockerfile."""
    from app.processors.gsplat.trainer import (
        GsplatTrainerUnavailableError,
        select_trainer_cls,
    )

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "gsplat", None)
    monkeypatch.setitem(sys.modules, "gsplat.rasterization", None)
    with pytest.raises(GsplatTrainerUnavailableError, match="`gsplat` package"):
        select_trainer_cls()


def test_processor_defers_resolution_to_run(monkeypatch):
    """GsplatProcessor.__init__ no longer calls select_trainer_cls()
    eagerly — the resolver runs inside `run()` so a missing dep
    surfaces as a clean job-failure event instead of crashing the
    worker process at construction. The instance attr stays at the
    class-level default (None) until run() resolves it."""
    from app.processors.gsplat.trainer import GsplatProcessor

    p = GsplatProcessor()
    assert p.trainer_cls is None


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
