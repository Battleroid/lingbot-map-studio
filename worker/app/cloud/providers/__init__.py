"""Cloud provider adapters.

Each module in this package registers a concrete `CloudProvider` under a
stable id that matches one of the `ExecutionTarget` literals in
`app.jobs.schema`. The dispatcher looks providers up by id; only the ones
whose dependencies are importable in the current process get registered.

The `fake` provider is always available — it runs the remote worker as a
sibling subprocess inside the studio container itself, so CI can cover the
full launch/claim/publish/finalise round trip without touching any cloud
API keys.
"""

from __future__ import annotations

from typing import Dict

from app.cloud.providers.base import CloudProvider, InstanceHandle, InstanceStatus

_REGISTRY: Dict[str, CloudProvider] = {}


def register(provider: CloudProvider) -> None:
    """Install a provider under its `id` attribute. Re-registration wins so
    tests can swap in a double without clearing the registry by hand."""
    _REGISTRY[provider.id] = provider


def get(provider_id: str) -> CloudProvider:
    """Look up a registered provider or raise a clear error.

    The error names the id that was requested rather than a generic KeyError
    so a misconfigured `execution_target` surfaces on submit, not inside the
    dispatcher mid-launch.
    """
    if provider_id not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise KeyError(
            f"no cloud provider registered for execution_target={provider_id!r} "
            f"(registered: {known})"
        )
    return _REGISTRY[provider_id]


def registered_ids() -> list[str]:
    """Sorted list of currently-registered provider ids. Used by the
    `/api/cloud/providers` endpoint to tell the UI which targets are live."""
    return sorted(_REGISTRY)


def clear() -> None:
    """Test-only — wipe the registry between tests."""
    _REGISTRY.clear()


# Fake provider is always importable; it only uses stdlib + the broker.
from app.cloud.providers.fake import FakeProvider  # noqa: E402

register(FakeProvider())

# Real providers self-register on import when their optional deps are
# present. Importing them here is best-effort: if the adapter's SDK isn't
# installed (e.g. `boto3` for AWS), the import fails silently and that
# target simply isn't exposed. This keeps the fake-only test environment
# lean without needing provider-specific feature flags.
for _mod in (
    "app.cloud.providers.runpod",
    "app.cloud.providers.vast",
    "app.cloud.providers.lambda_labs",
    "app.cloud.providers.paperspace",
    "app.cloud.providers.aws_ec2",
    "app.cloud.providers.gcp_gce",
    "app.cloud.providers.azure_vm",
):
    try:
        __import__(_mod)
    except Exception:  # noqa: BLE001
        # Missing SDK or credentials at import time — leave unregistered.
        pass


__all__ = [
    "CloudProvider",
    "InstanceHandle",
    "InstanceStatus",
    "clear",
    "get",
    "register",
    "registered_ids",
]
