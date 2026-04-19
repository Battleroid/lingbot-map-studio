"""GCP Compute Engine adapter.

Mirrors the AWS EC2 adapter against GCE's Instances API. Uses
`google-cloud-compute` when available; registration is gated on the
SDK being importable.

Design parity with `aws_ec2.py`:
- Lazy SDK import.
- `client_factory` hook for test injection.
- User-data bootstrap via `_ssh_bootstrap.build_bootstrap_script`.
- Preemptible (GCE's answer to spot) toggled by `instance_spec.spot`.
- Zones (not regions) are the native scope; we map `instance_spec.region`
  to a zone by appending `-a` when not specified explicitly.

Credentials: standard Application Default Credentials (ADC). Override
path via `settings.gcp_service_account_json`.

Image: a `lingbot-remote-worker-*` family image we publish with docker
+ `nvidia-container-runtime`; override via `AWS_GCP_IMAGE` env on the
spec (`GCP_IMAGE_URI` for precision).
"""

from __future__ import annotations

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


ClientFactory = Callable[[], Any]


# Map canonical gpu_class onto a GCE `accelerator_type` + machine-type
# pair. GCE separates the GPU from the machine: we attach the accelerator
# to an N1 machine for T4/V100 and use the purpose-built A2 / A3 families
# for A100/H100.
_ACCELERATOR_TYPE: dict[str, tuple[str, str, int]] = {
    # gpu_class: (accelerator_type, machine_type, accelerator_count)
    "t4": ("nvidia-tesla-t4", "n1-standard-8", 1),
    "v100": ("nvidia-tesla-v100", "n1-standard-8", 1),
    "a100-40g": ("nvidia-tesla-a100", "a2-highgpu-1g", 1),
    "a100-80g": ("nvidia-a100-80gb", "a2-ultragpu-1g", 1),
    "h100-80g": ("nvidia-h100-80gb", "a3-highgpu-8g", 8),
    "l4": ("nvidia-l4", "g2-standard-8", 1),
}

# Hourly rates in integer cents; coarse and zone-dependent.
_HOURLY_CENTS: dict[str, int] = {
    "t4": 35,
    "v100": 248,
    "a100-40g": 290,
    "a100-80g": 367,
    "h100-80g": 1125,
    "l4": 65,
}


def _default_client_factory() -> Any:
    """Production factory: construct a real `compute_v1.InstancesClient`.

    Uses ADC; if `settings.gcp_service_account_json` is set, loads
    service-account creds from that path instead.
    """
    from google.cloud import compute_v1  # noqa: WPS433

    if settings.gcp_service_account_json:
        from google.oauth2 import service_account  # noqa: WPS433

        creds = service_account.Credentials.from_service_account_file(
            settings.gcp_service_account_json
        )
        return compute_v1.InstancesClient(credentials=creds)
    return compute_v1.InstancesClient()


class GcpGceProvider(CloudProvider):
    id = "gcp-gce"
    display_name = "GCP Compute Engine (GPU)"

    client_factory: ClassVar[Optional[ClientFactory]] = None

    def _client(self) -> Any:
        factory = type(self).client_factory or _default_client_factory
        return factory()

    def _project_id(self, spec_env: dict[str, str]) -> str:
        project = spec_env.get("GCP_PROJECT_ID")
        if not project:
            raise RuntimeError(
                "gcp-gce: no project id. Set instance_spec.env['GCP_PROJECT_ID']."
            )
        return project

    def _zone(self, spec: LaunchSpec) -> str:
        # GCE is zone-scoped; callers can supply either a zone directly
        # (e.g. `us-central1-a`) or a region that we expand into a
        # deterministic zone by appending `-a`.
        explicit = spec.instance_spec.env.get("GCP_ZONE")
        if explicit:
            return explicit
        region = spec.instance_spec.region or settings.gcp_region_default
        return f"{region}-a" if "-" not in region.rsplit("-", 1)[-1] else region

    async def launch(self, spec: LaunchSpec) -> InstanceHandle:
        mapping = _ACCELERATOR_TYPE.get(spec.instance_spec.gpu_class)
        if not mapping:
            raise RuntimeError(
                f"gcp-gce: no accelerator mapping for "
                f"gpu_class={spec.instance_spec.gpu_class!r}."
            )
        accelerator_type, machine_type, accelerator_count = mapping
        image_uri = spec.instance_spec.env.get("GCP_IMAGE_URI")
        if not image_uri:
            raise RuntimeError(
                "gcp-gce: no source image. Set "
                "instance_spec.env['GCP_IMAGE_URI'] to a GCE image URI."
            )

        project = self._project_id(spec.instance_spec.env)
        zone = self._zone(spec)
        user_data = build_bootstrap_script(
            spec,
            extra_docker_args=("--gpus all", "--shm-size=8g"),
        )
        body: dict[str, Any] = {
            "name": f"lingbot-{spec.job_id}",
            "machine_type": f"zones/{zone}/machineTypes/{machine_type}",
            "disks": [
                {
                    "boot": True,
                    "auto_delete": True,
                    "initialize_params": {
                        "source_image": image_uri,
                        "disk_size_gb": spec.instance_spec.disk_gb,
                    },
                }
            ],
            "network_interfaces": [
                {
                    "name": "global/networks/default",
                    "access_configs": [{"name": "External NAT", "type_": "ONE_TO_ONE_NAT"}],
                }
            ],
            "guest_accelerators": [
                {
                    "accelerator_type": (
                        f"zones/{zone}/acceleratorTypes/{accelerator_type}"
                    ),
                    "accelerator_count": accelerator_count,
                }
            ],
            "scheduling": {
                # Required: instances with GPUs can't live-migrate.
                "on_host_maintenance": "TERMINATE",
                "automatic_restart": False,
            },
            "metadata": {
                "items": [{"key": "startup-script", "value": user_data}]
            },
            "labels": {
                "lingbot-job-id": spec.job_id.lower(),
                "lingbot-worker-class": spec.worker_class.lower(),
            },
        }
        if spec.instance_spec.spot:
            body["scheduling"]["preemptible"] = True
            body["scheduling"]["provisioning_model"] = "SPOT"

        client = self._client()
        op = await _maybe_await(
            client.insert(project=project, zone=zone, instance_resource=body)
        )
        # The GCE SDK returns an Operation; for our purposes the instance
        # name + zone is the identity. We don't block on op completion —
        # the broker's claim feed is the readiness signal.
        del op
        return InstanceHandle(
            provider_id=self.id,
            instance_id=body["name"],
            region=zone,
            gpu_class=spec.instance_spec.gpu_class,
            launched_at=time.time(),
            metadata={
                "project": project,
                "zone": zone,
                "machine_type": machine_type,
                "spot": "1" if spec.instance_spec.spot else "0",
            },
        )

    async def status(self, handle: InstanceHandle) -> InstanceStatus:
        client = self._client()
        try:
            inst = await _maybe_await(
                client.get(
                    project=handle.metadata.get("project", ""),
                    zone=handle.metadata.get("zone", handle.region or ""),
                    instance=handle.instance_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "notfound" in msg or "not found" in msg or "404" in msg:
                return "terminated"
            log.warning("gcp-gce get instance failed: %s", exc)
            return "error"
        status = str(getattr(inst, "status", "")).lower()
        if status == "running":
            return "running"
        if status in {"provisioning", "staging", "repairing"}:
            return "pending"
        if status in {"stopping", "stopped", "suspending", "suspended", "terminated"}:
            return "terminated"
        return "error"

    async def terminate(self, handle: InstanceHandle) -> None:
        client = self._client()
        try:
            await _maybe_await(
                client.delete(
                    project=handle.metadata.get("project", ""),
                    zone=handle.metadata.get("zone", handle.region or ""),
                    instance=handle.instance_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "notfound" in msg or "not found" in msg or "404" in msg:
                return
            log.warning("gcp-gce terminate %s failed: %s", handle.instance_id, exc)

    async def logs(self, handle: InstanceHandle, tail: int = 200) -> str:
        # GCE serial console output is available via `get_serial_port_output`
        # but is noisy and rate-limited. Broker events are authoritative.
        return ""

    async def estimate_cost(
        self, spec: InstanceSpec, expected_duration_s: float
    ) -> int:
        rate = _HOURLY_CENTS.get(spec.gpu_class, 500)
        if spec.spot:
            # Spot/preemptible is ~20–30% of on-demand on GCE.
            rate = int(round(rate * 0.3))
        hours = max(expected_duration_s, 60) / 3600.0
        return int(round(rate * hours))


async def _maybe_await(result: Any) -> Any:
    if hasattr(result, "__await__"):
        return await result
    return result


try:
    from google.cloud import compute_v1  # noqa: F401

    register(GcpGceProvider())
except ImportError:
    pass


__all__ = ["GcpGceProvider"]
