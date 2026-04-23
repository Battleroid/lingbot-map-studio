"""Lambda Labs adapter.

Lambda exposes a straightforward REST API — no auction, no offer
search, just `instance-types` + `launch`. Simplest of the R3 tier.

Limitation: Lambda Labs runs VMs, not containers. The onstart script
has to bootstrap docker + pull the remote-worker image, which takes
20–40s cold-start. The adapter writes a short bash user-data script
that installs docker if missing, pulls our image, and runs it with
the dispatcher's env injected.

Credentials: `settings.lambda_labs_api_key` (env `LAMBDA_LABS_API_KEY`).
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


# Lambda instance-type identifiers. Their API calls these
# `instance_type_name`; we map our canonical gpu_class onto the
# cheapest matching type. Users wanting a specific one override via
# `InstanceSpec.env["LAMBDA_INSTANCE_TYPE"]`.
_INSTANCE_TYPE: dict[str, str] = {
    "a100-40g": "gpu_1x_a100",
    "a100-80g": "gpu_1x_a100_sxm4",
    "h100-80g": "gpu_1x_h100_pcie",
    "a10": "gpu_1x_a10",
    "rtx6000ada": "gpu_1x_rtx6000",
    "v100": "gpu_1x_v100",
}

# Lambda's on-demand rates (USD/hr) — hardcoded fallbacks used by the
# cost estimator; real rates come from their website and shift
# occasionally. The adapter's `estimate_cost` is the only consumer so
# "close enough" is good enough.
_HOURLY_CENTS: dict[str, int] = {
    "a100-40g": 129,
    "a100-80g": 179,
    "h100-80g": 249,
    "a10": 75,
    "rtx6000ada": 159,
    "v100": 55,
}


class LambdaLabsProvider(CloudProvider):
    id = "lambda_labs"
    display_name = "Lambda Labs"

    transport_override: ClassVar[Optional[httpx.BaseTransport]] = None

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self._http = ProviderHttp(
            base_url=api_base or settings.lambda_labs_api_base,
            headers={
                "Authorization": f"Bearer {api_key if api_key is not None else settings.lambda_labs_api_key}",
                "Content-Type": "application/json",
            },
            transport=LambdaLabsProvider.transport_override,
        )

    async def launch(self, spec: LaunchSpec) -> InstanceHandle:
        if LambdaLabsProvider.transport_override is not None and self._http._transport is None:
            self._http.set_transport(LambdaLabsProvider.transport_override)
        client = self._http.client()

        instance_type = (
            spec.instance_spec.env.get("LAMBDA_INSTANCE_TYPE")
            or _INSTANCE_TYPE.get(spec.instance_spec.gpu_class)
        )
        if not instance_type:
            raise RuntimeError(
                f"lambda_labs: no instance_type mapping for gpu_class={spec.instance_spec.gpu_class!r}. "
                "set instance_spec.env['LAMBDA_INSTANCE_TYPE'] explicitly."
            )

        body: dict[str, Any] = {
            "region_name": spec.instance_spec.region or "us-east-1",
            "instance_type_name": instance_type,
            "ssh_key_names": [
                spec.instance_spec.env.get("LAMBDA_SSH_KEY_NAME", "studio-default")
            ],
            "quantity": 1,
            "name": f"lingbot-{spec.job_id}",
            "user_data": build_bootstrap_script(
                spec,
                extra_docker_args=("--gpus all", "--shm-size=8g"),
            ),
        }
        resp = await client.post("/instance-operations/launch", json=body)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"lambda launch failed ({resp.status_code}): {resp.text}"
            )
        data = resp.json().get("data") or resp.json()
        ids = data.get("instance_ids") or []
        if not ids:
            raise RuntimeError(f"lambda launch returned no instance_ids: {data!r}")
        return InstanceHandle(
            provider_id=self.id,
            instance_id=str(ids[0]),
            region=body["region_name"],
            gpu_class=spec.instance_spec.gpu_class,
            launched_at=time.time(),
            metadata={"instance_type": instance_type},
        )

    async def status(self, handle: InstanceHandle) -> InstanceStatus:
        client = self._http.client()
        resp = await client.get(f"/instances/{handle.instance_id}")
        if resp.status_code == 404:
            return "terminated"
        if resp.status_code >= 400:
            return "error"
        data = resp.json().get("data") or resp.json()
        status = str(data.get("status") or "").lower()
        if status == "active":
            return "running"
        if status in {"booting", "unhealthy"}:
            return "pending" if status == "booting" else "error"
        if status == "terminated":
            return "terminated"
        return "pending"

    async def terminate(self, handle: InstanceHandle) -> None:
        client = self._http.client()
        try:
            resp = await client.post(
                "/instance-operations/terminate",
                json={"instance_ids": [handle.instance_id]},
            )
            if resp.status_code not in (200, 204, 404):
                log.warning(
                    "lambda terminate %s returned %d: %s",
                    handle.instance_id,
                    resp.status_code,
                    resp.text,
                )
        except httpx.HTTPError as exc:
            log.warning("lambda terminate %s failed: %s", handle.instance_id, exc)

    async def logs(self, handle: InstanceHandle, tail: int = 200) -> str:
        # Lambda doesn't expose container logs via REST; user-data
        # stdout lives in `/var/log/cloud-init-output.log` on the VM,
        # reachable only via SSH. Return empty so the UI falls back to
        # the studio-side event stream.
        return ""

    async def estimate_cost(
        self, spec: InstanceSpec, expected_duration_s: float
    ) -> int:
        rate = _HOURLY_CENTS.get(spec.gpu_class, 100)
        hours = max(expected_duration_s, 60) / 3600.0
        return int(round(rate * hours * spec.gpu_count))


if settings.lambda_labs_api_key:
    register(LambdaLabsProvider())


__all__ = ["LambdaLabsProvider"]
