"""Paperspace adapter.

Paperspace exposes two relevant surfaces:

- **Gradient Deployments** (`paperspace-gradient`) — managed runtime for
  long-running workloads. Good fit for SLAM jobs that can take 20–60
  minutes. We submit a deployment spec referencing our remote-worker
  image and let Gradient handle the pod lifecycle + scaling.
- **Paperspace Core VMs** (`paperspace-core`) — raw GPU VMs. Good fit
  for short bursts (gsplat training when cold-start latency matters
  less than hourly cost). We launch a VM, SSH/user-data-bootstrap
  docker, and run the remote-worker image. Same `_ssh_bootstrap` helper
  as Lambda/Vast.

Both share credentials (`settings.paperspace_api_key`). Registration
happens once for each sub-provider when the key is set.

Caveats: Paperspace's API shape has drifted several times. The adapter
targets their v1 REST interface; if their new "core v2" rolls out we'll
add a sibling module rather than rewrite this one.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar, Optional

import httpx

from app.cloud.providers import register
from app.cloud.providers._http_base import ProviderHttp
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


# Paperspace calls their machine sizes things like "P4000", "A4000",
# "A100". The mapping is coarse — matches our canonical gpu_class onto
# the cheapest Paperspace machine type with that GPU. Override via
# `InstanceSpec.env["PAPERSPACE_MACHINE_TYPE"]`.
_MACHINE_TYPE: dict[str, str] = {
    "rtxa4000": "A4000",
    "rtxa5000": "A5000",
    "rtxa6000": "A6000",
    "a100-40g": "A100",
    "a100-80g": "A100-80G",
    "h100-80g": "H100",
    "v100": "V100",
}

# Published on-demand pricing (USD/hr) as of late 2025. Coarse — the UI
# shows it with an "estimated" label, and the real bill is authoritative.
_HOURLY_CENTS: dict[str, int] = {
    "rtxa4000": 76,
    "rtxa5000": 138,
    "rtxa6000": 189,
    "a100-40g": 309,
    "a100-80g": 389,
    "h100-80g": 549,
    "v100": 230,
}


# ---------------------------------------------------------------------------
# Gradient Deployments
# ---------------------------------------------------------------------------


class PaperspaceGradientProvider(CloudProvider):
    """Managed deployment target. One deployment per job — we don't share
    a long-lived deployment across runs (keeps billing attribution clean
    and avoids cross-job worker contamination).
    """

    id = "paperspace-gradient"
    display_name = "Paperspace Gradient"

    transport_override: ClassVar[Optional[httpx.BaseTransport]] = None

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self._http = ProviderHttp(
            base_url=api_base or settings.paperspace_api_base,
            headers={
                "X-Api-Key": api_key if api_key is not None else settings.paperspace_api_key,
                "Content-Type": "application/json",
            },
            transport=PaperspaceGradientProvider.transport_override,
        )

    async def launch(self, spec: LaunchSpec) -> InstanceHandle:
        if (
            PaperspaceGradientProvider.transport_override is not None
            and self._http._transport is None
        ):
            self._http.set_transport(PaperspaceGradientProvider.transport_override)
        client = self._http.client()

        machine_type = (
            spec.instance_spec.env.get("PAPERSPACE_MACHINE_TYPE")
            or _MACHINE_TYPE.get(spec.instance_spec.gpu_class)
        )
        if not machine_type:
            raise RuntimeError(
                f"paperspace-gradient: no machineType mapping for "
                f"gpu_class={spec.instance_spec.gpu_class!r}. Set "
                "instance_spec.env['PAPERSPACE_MACHINE_TYPE'] explicitly."
            )

        env_map = {
            "STUDIO_BROKER_URL": spec.broker_url,
            "STUDIO_JOB_TOKEN": spec.job_token,
            "WORKER_MODE": "remote",
            "WORKER_CLASS": spec.worker_class,
            **spec.env,
        }

        body: dict[str, Any] = {
            "name": f"lingbot-{spec.job_id}",
            "projectId": spec.instance_spec.env.get(
                "PAPERSPACE_PROJECT_ID",
                settings.paperspace_api_key and "default",
            ),
            "spec": {
                "image": spec.image,
                "command": ["python", "-m", "app.worker_main"],
                "env": [{"name": k, "value": v} for k, v in env_map.items()],
                "resources": {
                    "replicas": 1,
                    "instanceType": machine_type,
                },
            },
        }
        resp = await client.post("/deployments", json=body)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"paperspace-gradient launch failed ({resp.status_code}): {resp.text}"
            )
        data = resp.json()
        deployment_id = data.get("id") or data.get("deploymentId")
        if not deployment_id:
            raise RuntimeError(
                f"paperspace-gradient: launch returned no id: {data!r}"
            )
        return InstanceHandle(
            provider_id=self.id,
            instance_id=str(deployment_id),
            region=spec.instance_spec.region,
            gpu_class=spec.instance_spec.gpu_class,
            launched_at=time.time(),
            metadata={"machine_type": machine_type},
        )

    async def status(self, handle: InstanceHandle) -> InstanceStatus:
        client = self._http.client()
        resp = await client.get(f"/deployments/{handle.instance_id}")
        if resp.status_code == 404:
            return "terminated"
        if resp.status_code >= 400:
            return "error"
        data = resp.json()
        state = str(data.get("state") or data.get("status") or "").lower()
        if state in {"ready", "running"}:
            return "running"
        if state in {"provisioning", "pending", "scaling"}:
            return "pending"
        if state in {"stopped", "terminated", "deleted"}:
            return "terminated"
        if state in {"error", "failed"}:
            return "error"
        return "pending"

    async def terminate(self, handle: InstanceHandle) -> None:
        client = self._http.client()
        try:
            resp = await client.delete(f"/deployments/{handle.instance_id}")
            if resp.status_code not in (200, 202, 204, 404):
                log.warning(
                    "paperspace-gradient terminate %s returned %d: %s",
                    handle.instance_id,
                    resp.status_code,
                    resp.text,
                )
        except httpx.HTTPError as exc:
            log.warning(
                "paperspace-gradient terminate %s failed: %s",
                handle.instance_id,
                exc,
            )

    async def logs(self, handle: InstanceHandle, tail: int = 200) -> str:
        client = self._http.client()
        try:
            resp = await client.get(
                f"/deployments/{handle.instance_id}/logs",
                params={"tail": tail},
            )
        except httpx.HTTPError:
            return ""
        if resp.status_code >= 400:
            return ""
        return resp.text

    async def estimate_cost(
        self, spec: InstanceSpec, expected_duration_s: float
    ) -> int:
        rate = _HOURLY_CENTS.get(spec.gpu_class, 100)
        hours = max(expected_duration_s, 60) / 3600.0
        return int(round(rate * hours * spec.gpu_count))


# ---------------------------------------------------------------------------
# Paperspace Core VMs
# ---------------------------------------------------------------------------


class PaperspaceCoreProvider(CloudProvider):
    """Raw VM target. Bootstraps docker via the same user-data script as
    Lambda / Vast, then runs the remote-worker image."""

    id = "paperspace-core"
    display_name = "Paperspace Core"

    transport_override: ClassVar[Optional[httpx.BaseTransport]] = None

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self._http = ProviderHttp(
            base_url=api_base or settings.paperspace_api_base,
            headers={
                "X-Api-Key": api_key if api_key is not None else settings.paperspace_api_key,
                "Content-Type": "application/json",
            },
            transport=PaperspaceCoreProvider.transport_override,
        )

    async def launch(self, spec: LaunchSpec) -> InstanceHandle:
        if (
            PaperspaceCoreProvider.transport_override is not None
            and self._http._transport is None
        ):
            self._http.set_transport(PaperspaceCoreProvider.transport_override)
        client = self._http.client()

        machine_type = (
            spec.instance_spec.env.get("PAPERSPACE_MACHINE_TYPE")
            or _MACHINE_TYPE.get(spec.instance_spec.gpu_class)
        )
        if not machine_type:
            raise RuntimeError(
                f"paperspace-core: no machineType mapping for "
                f"gpu_class={spec.instance_spec.gpu_class!r}"
            )

        body: dict[str, Any] = {
            "region": spec.instance_spec.region or "East Coast (NY2)",
            "machineType": machine_type,
            "size": spec.instance_spec.disk_gb,
            "billingType": "hourly",
            "machineName": f"lingbot-{spec.job_id}",
            "templateId": spec.instance_spec.env.get(
                "PAPERSPACE_TEMPLATE_ID", "tlnrbx2"  # Ubuntu 22.04 GPU default
            ),
            "assignPublicIp": True,
            "startOnCreate": True,
            "scriptId": None,
            "userData": build_bootstrap_script(
                spec,
                extra_docker_args=("--gpus all", "--shm-size=8g"),
            ),
        }
        resp = await client.post("/machines/createSingleMachinePublic", json=body)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"paperspace-core launch failed ({resp.status_code}): {resp.text}"
            )
        data = resp.json()
        machine_id = data.get("id") or data.get("machineId")
        if not machine_id:
            raise RuntimeError(f"paperspace-core: no id in response: {data!r}")
        return InstanceHandle(
            provider_id=self.id,
            instance_id=str(machine_id),
            region=body["region"],
            gpu_class=spec.instance_spec.gpu_class,
            launched_at=time.time(),
            metadata={"machine_type": machine_type},
        )

    async def status(self, handle: InstanceHandle) -> InstanceStatus:
        client = self._http.client()
        resp = await client.get(
            "/machines/getMachinePublic",
            params={"machineId": handle.instance_id},
        )
        if resp.status_code == 404:
            return "terminated"
        if resp.status_code >= 400:
            return "error"
        data = resp.json()
        state = str(data.get("state") or "").lower()
        if state in {"ready", "serviceready"}:
            return "running"
        if state in {"provisioning", "starting", "restarting"}:
            return "pending"
        if state in {"off", "stopped", "terminated"}:
            return "terminated"
        return "error"

    async def terminate(self, handle: InstanceHandle) -> None:
        client = self._http.client()
        try:
            resp = await client.post(
                "/machines/destroyMachine",
                json={"machineId": handle.instance_id, "releasePublicIp": True},
            )
            if resp.status_code not in (200, 202, 204, 404):
                log.warning(
                    "paperspace-core terminate %s returned %d: %s",
                    handle.instance_id,
                    resp.status_code,
                    resp.text,
                )
        except httpx.HTTPError as exc:
            log.warning(
                "paperspace-core terminate %s failed: %s",
                handle.instance_id,
                exc,
            )

    async def logs(self, handle: InstanceHandle, tail: int = 200) -> str:
        # Core VMs don't expose docker logs through the REST API; the
        # remote worker's broker-published events are the authoritative
        # source. Return empty so the UI falls back to the event stream.
        return ""

    async def estimate_cost(
        self, spec: InstanceSpec, expected_duration_s: float
    ) -> int:
        rate = _HOURLY_CENTS.get(spec.gpu_class, 100)
        hours = max(expected_duration_s, 60) / 3600.0
        return int(round(rate * hours * spec.gpu_count))


if settings.paperspace_api_key:
    register(PaperspaceGradientProvider())
    register(PaperspaceCoreProvider())


__all__ = ["PaperspaceGradientProvider", "PaperspaceCoreProvider"]
