"""Processor registry.

Each concrete processor registers itself here by id. The runner looks up the
processor matching `config.processor` and dispatches to it. Adding a new
processor is a two-line change: import the class and drop it into the
registry dict.

Registration is lazy — heavy backends (DROID-SLAM, gsplat, etc) import
torch/CUDA at module load, which would crash the API container that doesn't
have those wheels installed. Each concrete processor module gates its
imports so loading the class shell is cheap.
"""

from __future__ import annotations

from typing import Type

from app.jobs.schema import AnyJobConfig, ProcessorId
from app.processors.base import Processor
from app.processors.lingbot import LingbotProcessor

# Registry keyed by processor id. Populated eagerly at import time for the
# lingbot processor; SLAM and gsplat processors are added in later phases and
# register themselves here when their modules land.
REGISTRY: dict[ProcessorId, Type[Processor]] = {
    "lingbot": LingbotProcessor,
}


def register(processor_cls: Type[Processor]) -> Type[Processor]:
    """Add a processor class to the registry. Decorator-friendly so later
    phases' processor modules can self-register on import."""
    pid = getattr(processor_cls, "id", None)
    if not pid:
        raise ValueError(f"{processor_cls!r} must set a string `id` class attribute")
    REGISTRY[pid] = processor_cls
    return processor_cls


def resolve(config: AnyJobConfig) -> Processor:
    """Return an instance of the processor matching this config."""
    pid = config.processor
    cls = REGISTRY.get(pid)
    if cls is None:
        raise ValueError(
            f"no processor registered for id={pid!r}. "
            f"known ids: {sorted(REGISTRY)}"
        )
    return cls()


def worker_class_for(config: AnyJobConfig) -> str:
    """The worker-class container that should claim a job with this config.

    Used by the API process to route new jobs in Phase 2. Today (Phase 1)
    everything still runs in-process, but the function is already used so
    that the routing table has a single authoritative source.
    """
    cls = REGISTRY.get(config.processor)
    if cls is None:
        raise ValueError(f"no processor registered for id={config.processor!r}")
    return cls.worker_class
