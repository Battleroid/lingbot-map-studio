"""Processor registry.

Each concrete processor registers itself here by id. The runner looks up the
processor matching `config.processor` and dispatches to it.

Registration is split in two:

  1. `WORKER_CLASSES` maps each processor id to its worker-class container.
     Lightweight — just a dict of strings. The API container uses this to
     route new jobs (no heavy imports needed).

  2. `load_processor(id)` imports the concrete processor module on demand.
     This matters because SLAM / gsplat backends pull in different torch /
     CUDA wheels than lingbot, so importing all three at once would crash
     whichever container is missing one set. In a worker container, we only
     ever load the processors it knows how to run.
"""

from __future__ import annotations

import importlib
import logging
from typing import Type

from app.jobs.schema import AnyJobConfig, ProcessorId
from app.processors.base import Processor

log = logging.getLogger(__name__)


# Single source of truth for which container runs which processor. Kept
# import-free so the API can consult it without loading torch.
WORKER_CLASSES: dict[str, str] = {
    "lingbot": "lingbot",
    "droid_slam": "slam",
    "mast3r_slam": "slam",
    "dpvo": "slam",
    "monogs": "slam",
    "gsplat": "gs",
}

# Module path for each processor class. Loaded lazily by `load_processor`.
_MODULE_PATHS: dict[str, tuple[str, str]] = {
    "lingbot": ("app.processors.lingbot", "LingbotProcessor"),
    "droid_slam": ("app.processors.slam.stub", "DroidSlamStubProcessor"),
    "mast3r_slam": ("app.processors.slam.stub", "Mast3rSlamStubProcessor"),
    "dpvo": ("app.processors.slam.stub", "DpvoStubProcessor"),
    "monogs": ("app.processors.slam.stub", "MonogsStubProcessor"),
    "gsplat": ("app.processors.gsplat.stub", "GsplatStubProcessor"),
}

# Populated on demand by `load_processor`. Caches the concrete class so we
# don't re-import for every job.
REGISTRY: dict[str, Type[Processor]] = {}


def register(processor_cls: Type[Processor]) -> Type[Processor]:
    """Decorator for processor modules that want to self-register.

    Primarily useful in tests / dev setups where a single container loads
    everything; real deployments rely on `load_processor` to populate the
    registry lazily.
    """
    pid = getattr(processor_cls, "id", None)
    if not pid:
        raise ValueError(f"{processor_cls!r} must set a string `id` class attribute")
    REGISTRY[pid] = processor_cls
    return processor_cls


def load_processor(pid: ProcessorId) -> Type[Processor]:
    """Import and cache the processor class for `pid`."""
    if pid in REGISTRY:
        return REGISTRY[pid]
    entry = _MODULE_PATHS.get(pid)
    if entry is None:
        raise ValueError(f"no processor registered for id={pid!r}")
    module_path, class_name = entry
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"processor {pid!r} lives in {module_path}, but it failed to "
            f"import in this container: {exc}. "
            "check that this worker_class has the right deps baked in."
        ) from exc
    cls = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, Processor):
        raise ValueError(
            f"{module_path}.{class_name} is not a Processor subclass"
        )
    REGISTRY[pid] = cls
    return cls


def resolve(config: AnyJobConfig) -> Processor:
    """Return an instance of the processor matching this config."""
    cls = load_processor(config.processor)
    return cls()


def worker_class_for(config: AnyJobConfig) -> str:
    """The worker-class container that should claim a job with this config.

    Looked up from the static map so the API process doesn't have to import
    heavy processor modules just to enqueue.
    """
    wc = WORKER_CLASSES.get(config.processor)
    if wc is None:
        raise ValueError(f"no worker class mapped for processor {config.processor!r}")
    return wc


def worker_class_from_id(pid: str) -> str:
    """String-keyed variant for callers that have only the raw id string."""
    wc = WORKER_CLASSES.get(pid)
    if wc is None:
        raise ValueError(f"no worker class mapped for processor {pid!r}")
    return wc


def ids_for_worker_class(worker_class: str) -> list[str]:
    """All processor ids handled by this worker-class container."""
    return [pid for pid, wc in WORKER_CLASSES.items() if wc == worker_class]
