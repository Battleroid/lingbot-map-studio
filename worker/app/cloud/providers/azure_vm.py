"""Azure VM adapter.

Mirrors the AWS / GCP adapters against Azure's Compute Management API.
Uses `azure-mgmt-compute` + `azure-identity` when available; registration
is gated on both SDKs being importable.

Design parity with `aws_ec2.py` / `gcp_gce.py`:
- Lazy SDK import with a `client_factory` hook for tests.
- User-data bootstrap via `_ssh_bootstrap.build_bootstrap_script`,
  delivered through Azure's `customData` field (base64-encoded).
- Spot VMs enabled via `priority=Spot` + `evictionPolicy=Delete`.

Credentials: `DefaultAzureCredential` — the same chain the Azure CLI
uses (env, managed identity, az cli, etc.). We don't read keys
directly from our settings.

Subscription: `settings.azure_subscription_id` (or per-spec override).

Image: an `lingbot-remote-worker-*` image we publish to a gallery.
Override via `InstanceSpec.env["AZURE_IMAGE_ID"]`.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Callable, ClassVar, Optional

from app.cloud.providers import register
from app.cloud.providers._ssh_bootstrap import build_bootstrap_script
from app.cloud.providers.base import (
    CloudProvider,
    InstanceHandle,
    InstanceStatus,
    LaunchSpec,
)
from app.config import settings
from app.jobs.schema import InstanceSpec

log = logging.getLogger(__name__)


ClientFactory = Callable[[str], Any]


# Azure VM size mapping for canonical gpu_class.
_VM_SIZE: dict[str, str] = {
    "t4": "Standard_NC4as_T4_v3",
    "v100": "Standard_NC6s_v3",
    "a100-40g": "Standard_NC24ads_A100_v4",
    "a100-80g": "Standard_NC48ads_A100_v4",
    "h100-80g": "Standard_ND96isr_H100_v5",
}

_HOURLY_CENTS: dict[str, int] = {
    "t4": 53,
    "v100": 306,
    "a100-40g": 368,
    "a100-80g": 736,
    "h100-80g": 9800,
}


def _default_client_factory(subscription_id: str) -> Any:
    """Production factory: construct a real `ComputeManagementClient`."""
    from azure.identity import DefaultAzureCredential  # noqa: WPS433
    from azure.mgmt.compute import ComputeManagementClient  # noqa: WPS433

    cred = DefaultAzureCredential()
    return ComputeManagementClient(cred, subscription_id)


class AzureVmProvider(CloudProvider):
    id = "azure-vm"
    display_name = "Azure Virtual Machines (GPU)"

    client_factory: ClassVar[Optional[ClientFactory]] = None

    def _client(self, subscription_id: str) -> Any:
        factory = type(self).client_factory or _default_client_factory
        return factory(subscription_id)

    async def launch(self, spec: LaunchSpec) -> InstanceHandle:
        vm_size = (
            spec.instance_spec.env.get("AZURE_VM_SIZE")
            or _VM_SIZE.get(spec.instance_spec.gpu_class)
        )
        if not vm_size:
            raise RuntimeError(
                f"azure-vm: no VM-size mapping for "
                f"gpu_class={spec.instance_spec.gpu_class!r}. Set "
                "instance_spec.env['AZURE_VM_SIZE'] explicitly."
            )
        image_id = spec.instance_spec.env.get("AZURE_IMAGE_ID") or ""
        if not image_id:
            raise RuntimeError(
                "azure-vm: no image id. Set "
                "instance_spec.env['AZURE_IMAGE_ID']."
            )
        resource_group = spec.instance_spec.env.get("AZURE_RESOURCE_GROUP")
        if not resource_group:
            raise RuntimeError(
                "azure-vm: no resource group. Set "
                "instance_spec.env['AZURE_RESOURCE_GROUP']."
            )
        nic_id = spec.instance_spec.env.get("AZURE_NIC_ID")
        if not nic_id:
            raise RuntimeError(
                "azure-vm: no network interface. Set "
                "instance_spec.env['AZURE_NIC_ID'] to an existing NIC."
            )
        subscription_id = (
            spec.instance_spec.env.get("AZURE_SUBSCRIPTION_ID")
            or settings.azure_subscription_id
        )
        if not subscription_id:
            raise RuntimeError("azure-vm: no subscription id configured.")

        region = spec.instance_spec.region or settings.azure_region_default
        vm_name = f"lingbot-{spec.job_id}"[:64]  # Azure caps at 64 chars
        user_data = build_bootstrap_script(
            spec,
            extra_docker_args=("--gpus all", "--shm-size=8g"),
        )
        custom_data_b64 = base64.b64encode(user_data.encode("utf-8")).decode("ascii")

        parameters: dict[str, Any] = {
            "location": region,
            "hardware_profile": {"vm_size": vm_size},
            "storage_profile": {
                "image_reference": {"id": image_id},
                "os_disk": {
                    "create_option": "FromImage",
                    "disk_size_gb": spec.instance_spec.disk_gb,
                    "delete_option": "Delete",
                },
            },
            "os_profile": {
                "computer_name": vm_name,
                "admin_username": "lingbot",
                "custom_data": custom_data_b64,
                "linux_configuration": {
                    "disable_password_authentication": True,
                    # User must supply an authorized SSH key via env
                    # if they want post-mortem SSH access; we don't
                    # need it to run the job.
                },
            },
            "network_profile": {
                "network_interfaces": [
                    {"id": nic_id, "properties": {"primary": True}}
                ]
            },
            "tags": {
                "lingbot:job_id": spec.job_id,
                "lingbot:worker_class": spec.worker_class,
            },
        }
        if spec.instance_spec.spot:
            parameters["priority"] = "Spot"
            parameters["eviction_policy"] = "Delete"
            parameters["billing_profile"] = {"max_price": -1}  # market

        client = self._client(subscription_id)
        poller = await _maybe_await(
            client.virtual_machines.begin_create_or_update(
                resource_group_name=resource_group,
                vm_name=vm_name,
                parameters=parameters,
            )
        )
        # Don't block on VM boot — the broker's claim feed is the
        # readiness signal. Poller object kept by the SDK.
        del poller

        return InstanceHandle(
            provider_id=self.id,
            instance_id=vm_name,
            region=region,
            gpu_class=spec.instance_spec.gpu_class,
            launched_at=time.time(),
            metadata={
                "resource_group": resource_group,
                "vm_size": vm_size,
                "subscription_id": subscription_id,
                "spot": "1" if spec.instance_spec.spot else "0",
            },
        )

    async def status(self, handle: InstanceHandle) -> InstanceStatus:
        subscription_id = handle.metadata.get("subscription_id") or settings.azure_subscription_id
        resource_group = handle.metadata.get("resource_group", "")
        if not subscription_id or not resource_group:
            return "error"
        client = self._client(subscription_id)
        try:
            instance = await _maybe_await(
                client.virtual_machines.instance_view(
                    resource_group_name=resource_group,
                    vm_name=handle.instance_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "notfound" in msg or "not found" in msg or "404" in msg:
                return "terminated"
            log.warning("azure-vm instance_view failed: %s", exc)
            return "error"
        # instance_view.statuses is a list of statuses; the relevant one
        # has a code like `PowerState/running`.
        statuses = getattr(instance, "statuses", None) or []
        for s in statuses:
            code = str(getattr(s, "code", "")).lower()
            if code.startswith("powerstate/"):
                state = code.split("/", 1)[1]
                if state == "running":
                    return "running"
                if state in {"starting", "creating"}:
                    return "pending"
                if state in {"stopping", "stopped", "deallocating", "deallocated"}:
                    return "terminated"
        return "pending"

    async def terminate(self, handle: InstanceHandle) -> None:
        subscription_id = handle.metadata.get("subscription_id") or settings.azure_subscription_id
        resource_group = handle.metadata.get("resource_group", "")
        if not subscription_id or not resource_group:
            return
        client = self._client(subscription_id)
        try:
            await _maybe_await(
                client.virtual_machines.begin_delete(
                    resource_group_name=resource_group,
                    vm_name=handle.instance_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "notfound" in msg or "not found" in msg or "404" in msg:
                return
            log.warning("azure-vm terminate %s failed: %s", handle.instance_id, exc)

    async def logs(self, handle: InstanceHandle, tail: int = 200) -> str:
        # Azure exposes boot diagnostics but the container logs only
        # live on the VM itself. Broker events are authoritative.
        return ""

    async def estimate_cost(
        self, spec: InstanceSpec, expected_duration_s: float
    ) -> int:
        rate = _HOURLY_CENTS.get(spec.gpu_class, 500)
        if spec.spot:
            # Azure spot is ~10–50% of on-demand depending on region +
            # capacity; 40% is a conservative mid-point.
            rate = int(round(rate * 0.4))
        hours = max(expected_duration_s, 60) / 3600.0
        return int(round(rate * hours))


async def _maybe_await(result: Any) -> Any:
    if hasattr(result, "__await__"):
        return await result
    return result


try:
    import azure.mgmt.compute  # noqa: F401
    import azure.identity  # noqa: F401

    register(AzureVmProvider())
except ImportError:
    pass


__all__ = ["AzureVmProvider"]
