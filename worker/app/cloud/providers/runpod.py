"""RunPod adapter.

Talks to the RunPod REST v1 API (https://rest.runpod.io/v1). Two
flavours are exposed as separate provider ids:

- `runpod` — the standard `Pod` API. Provision a GPU pod with our
  remote-worker image, hand it a per-job HMAC token + broker URL, let
  it claim the queued job and exit. Best for anything whose wall-time
  dominates the cold-start cost (SLAM runs, longer gsplat trainings).

- `runpod-serverless` — the `Serverless` API. POST a single-shot
  request to a pre-deployed endpoint that runs one `runner.run_job`
  call and returns. Best for short gsplat trainings where the 30-60s
  pod cold-start would eat a non-trivial fraction of total wall-time.

Both adapters share the same token/broker/claim shape so the studio
side doesn't care which one ran the job — the broker sees a normal
claim either way.

Credentials: `settings.runpod_api_key` (env `RUNPOD_API_KEY`). If the
key is empty at import time the module falls back to registering
nothing, so the fake-only CI environment doesn't need to mock this
module out.

The HTTP client is lazily constructed and can be swapped in tests via
`RunPodProvider.transport_override`. Production doesn't touch it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar, Optional

import httpx

from app.cloud.providers import register
from app.cloud.providers.base import (
    CloudProvider,
    InstanceHandle,
    InstanceStatus,
    LaunchSpec,
)
from app.config import settings
from app.jobs.schema import InstanceSpec

log = logging.getLogger(__name__)


# Coarse hourly rates in cents, keyed by canonical gpu_class. These seed
# `estimate_cost` when the live pricing endpoint isn't available / not
# worth an extra round trip (the UI calls `estimate_cost` on a debounce).
# Values are conservative — underestimating cost would make the cap
# guard ineffective. Real rates move; the RunPod website is the source
# of truth.
_DEFAULT_HOURLY_CENTS: dict[str, int] = {
    "rtx3090": 40,
    "rtx4090": 50,
    "rtxa5000": 50,
    "rtxa6000": 80,
    "a40": 60,
    "a100-40g": 180,
    "a100-80g": 240,
    "h100-80g": 450,
    "h100-pcie": 420,
    "l40s": 110,
    "l4": 50,
}

# Map our cross-provider gpu_class strings onto RunPod's gpuTypeId
# strings (the ones the Pod API accepts verbatim). Users can override
# via `InstanceSpec.env["RUNPOD_GPU_TYPE_ID"]` if they want a class
# the map doesn't know.
_GPU_TYPE_ID: dict[str, str] = {
    "rtx3090": "NVIDIA GeForce RTX 3090",
    "rtx4090": "NVIDIA GeForce RTX 4090",
    "rtxa5000": "NVIDIA RTX A5000",
    "rtxa6000": "NVIDIA RTX A6000",
    "a40": "NVIDIA A40",
    "a100-40g": "NVIDIA A100 40GB PCIe",
    "a100-80g": "NVIDIA A100 80GB PCIe",
    "h100-80g": "NVIDIA H100 80GB HBM3",
    "h100-pcie": "NVIDIA H100 PCIe",
    "l40s": "NVIDIA L40S",
    "l4": "NVIDIA L4",
}


class RunPodProvider(CloudProvider):
    id = "runpod"
    display_name = "RunPod (GPU pods)"

    # Tests inject an `httpx.MockTransport`; production leaves this None.
    transport_override: ClassVar[Optional[httpx.BaseTransport]] = None

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.runpod_api_key
        self._api_base = (api_base or settings.runpod_api_base).rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    # --- HTTP plumbing --------------------------------------------------

    def _build_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        kwargs: dict[str, Any] = {
            "base_url": self._api_base,
            "headers": {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            "timeout": 30.0,
        }
        if RunPodProvider.transport_override is not None:
            kwargs["transport"] = RunPodProvider.transport_override
        self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- CloudProvider surface -----------------------------------------

    async def launch(self, spec: LaunchSpec) -> InstanceHandle:
        gpu_type_id = (
            spec.instance_spec.env.get("RUNPOD_GPU_TYPE_ID")
            or _GPU_TYPE_ID.get(spec.instance_spec.gpu_class, spec.instance_spec.gpu_class)
        )
        cloud_type = "SECURE" if not spec.instance_spec.spot else "COMMUNITY"

        # Dispatcher-provided env is the authoritative set for the pod;
        # instance_spec.env is merged on top so user overrides win over
        # our defaults but never over what the dispatcher asserts.
        env_map: dict[str, str] = {
            "STUDIO_BROKER_URL": spec.broker_url,
            "STUDIO_JOB_TOKEN": spec.job_token,
            "WORKER_MODE": "remote",
            "WORKER_CLASS": spec.worker_class,
            **spec.env,
        }
        body = {
            "name": f"lingbot-studio-{spec.job_id}",
            "imageName": spec.image,
            "gpuTypeId": gpu_type_id,
            "gpuCount": spec.instance_spec.gpu_count,
            "volumeInGb": spec.instance_spec.disk_gb,
            "containerDiskInGb": max(20, spec.instance_spec.disk_gb // 2),
            "cloudType": cloud_type,
            "env": [{"key": k, "value": v} for k, v in env_map.items()],
            # Long-running workloads shouldn't be auto-killed if idle.
            "minVcpuCount": 2,
            "minMemoryInGb": 8,
        }
        if spec.instance_spec.region:
            body["dataCenterIds"] = [spec.instance_spec.region]

        client = self._build_client()
        resp = await client.post("/pods", json=body)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"runpod launch failed ({resp.status_code}): {resp.text}"
            )
        data = resp.json()
        pod_id = data.get("id") or data.get("podId")
        if not pod_id:
            raise RuntimeError(f"runpod response missing pod id: {data!r}")
        return InstanceHandle(
            provider_id=self.id,
            instance_id=pod_id,
            region=data.get("dataCenterId") or spec.instance_spec.region,
            gpu_class=spec.instance_spec.gpu_class,
            launched_at=time.time(),
            metadata={
                "gpu_type_id": gpu_type_id,
                "cloud_type": cloud_type,
                "name": body["name"],
            },
        )

    async def status(self, handle: InstanceHandle) -> InstanceStatus:
        client = self._build_client()
        resp = await client.get(f"/pods/{handle.instance_id}")
        if resp.status_code == 404:
            return "terminated"
        if resp.status_code >= 400:
            return "error"
        data = resp.json()
        raw = (data.get("desiredStatus") or data.get("status") or "").upper()
        # RunPod vocabulary: CREATED / RUNNING / EXITED / TERMINATED /
        # FAILED. Map onto our compact status.
        if raw in {"RUNNING", "STARTING"}:
            return "running"
        if raw in {"CREATED", "PENDING"}:
            return "pending"
        if raw in {"FAILED", "ERROR"}:
            return "error"
        return "terminated"

    async def terminate(self, handle: InstanceHandle) -> None:
        client = self._build_client()
        try:
            resp = await client.delete(f"/pods/{handle.instance_id}")
            if resp.status_code in (200, 204, 404):
                return
            log.warning(
                "runpod terminate %s returned %d: %s",
                handle.instance_id,
                resp.status_code,
                resp.text,
            )
        except httpx.HTTPError as exc:
            # Idempotency contract: terminate must never raise.
            log.warning("runpod terminate %s failed: %s", handle.instance_id, exc)

    async def logs(self, handle: InstanceHandle, tail: int = 200) -> str:
        client = self._build_client()
        try:
            resp = await client.get(
                f"/pods/{handle.instance_id}/logs", params={"lines": tail}
            )
        except httpx.HTTPError:
            return ""
        if resp.status_code >= 400:
            return ""
        data = resp.json()
        lines = data.get("logs") or data.get("lines") or []
        if isinstance(lines, list):
            return "\n".join(str(ln) for ln in lines[-tail:])
        return str(lines)[-(tail * 120):]

    async def estimate_cost(
        self, spec: InstanceSpec, expected_duration_s: float
    ) -> int:
        rate = _DEFAULT_HOURLY_CENTS.get(spec.gpu_class, 100)
        hours = max(expected_duration_s, 60) / 3600.0  # min 1-minute billing unit
        cents = int(round(rate * hours * spec.gpu_count))
        # Spot discount: RunPod's community cloud is ~30-50% cheaper.
        if spec.spot:
            cents = int(cents * 0.6)
        return cents


class RunPodServerlessProvider(CloudProvider):
    """Thin variant that posts to a pre-deployed serverless endpoint.

    Endpoint id lives in `InstanceSpec.env["RUNPOD_ENDPOINT_ID"]`; the
    payload is a normal launch env plus a "task" discriminator the
    endpoint template consumes. Serverless pods have sub-second spin-up
    from a warm pool, which is the reason we expose this alongside the
    regular pod path.
    """

    id = "runpod-serverless"
    display_name = "RunPod Serverless"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.runpod_api_key
        # Serverless uses a different subdomain in RunPod's current API.
        self._api_base = (
            api_base
            or settings.runpod_api_base.replace("rest.runpod.io/v1", "api.runpod.ai/v2")
        ).rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    def _build_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        kwargs: dict[str, Any] = {
            "base_url": self._api_base,
            "headers": {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            "timeout": 30.0,
        }
        if RunPodProvider.transport_override is not None:
            kwargs["transport"] = RunPodProvider.transport_override
        self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def launch(self, spec: LaunchSpec) -> InstanceHandle:
        endpoint_id = spec.instance_spec.env.get("RUNPOD_ENDPOINT_ID")
        if not endpoint_id:
            raise RuntimeError(
                "runpod-serverless requires instance_spec.env['RUNPOD_ENDPOINT_ID'] — "
                "deploy a serverless endpoint pointing at the remote-worker image first."
            )
        client = self._build_client()
        payload = {
            "input": {
                "studio_broker_url": spec.broker_url,
                "studio_job_token": spec.job_token,
                "worker_class": spec.worker_class,
                "job_id": spec.job_id,
                **spec.env,
            }
        }
        resp = await client.post(f"/{endpoint_id}/run", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"runpod-serverless run failed ({resp.status_code}): {resp.text}"
            )
        data = resp.json()
        request_id = data.get("id")
        if not request_id:
            raise RuntimeError(f"serverless response missing id: {data!r}")
        return InstanceHandle(
            provider_id=self.id,
            instance_id=request_id,
            region=None,
            gpu_class=spec.instance_spec.gpu_class,
            launched_at=time.time(),
            metadata={"endpoint_id": endpoint_id},
        )

    async def status(self, handle: InstanceHandle) -> InstanceStatus:
        endpoint_id = handle.metadata.get("endpoint_id")
        if not endpoint_id:
            return "error"
        client = self._build_client()
        resp = await client.get(f"/{endpoint_id}/status/{handle.instance_id}")
        if resp.status_code == 404:
            return "terminated"
        if resp.status_code >= 400:
            return "error"
        raw = (resp.json().get("status") or "").upper()
        if raw in {"IN_QUEUE", "IN_PROGRESS"}:
            return "pending" if raw == "IN_QUEUE" else "running"
        if raw == "COMPLETED":
            return "terminated"
        if raw in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            return "error"
        return "pending"

    async def terminate(self, handle: InstanceHandle) -> None:
        endpoint_id = handle.metadata.get("endpoint_id")
        if not endpoint_id:
            return
        client = self._build_client()
        try:
            await client.post(f"/{endpoint_id}/cancel/{handle.instance_id}")
        except httpx.HTTPError as exc:
            log.warning(
                "runpod-serverless cancel %s failed: %s", handle.instance_id, exc
            )

    async def logs(self, handle: InstanceHandle, tail: int = 200) -> str:
        # Serverless exposes an `/output` endpoint rather than a logs
        # stream. We surface the `stdout` field from the final response
        # for completed runs; in-flight runs return nothing useful.
        endpoint_id = handle.metadata.get("endpoint_id")
        if not endpoint_id:
            return ""
        client = self._build_client()
        try:
            resp = await client.get(
                f"/{endpoint_id}/status/{handle.instance_id}"
            )
        except httpx.HTTPError:
            return ""
        if resp.status_code >= 400:
            return ""
        data = resp.json()
        stdout = data.get("output", {}).get("stdout", "")
        lines = str(stdout).splitlines()
        return "\n".join(lines[-tail:])

    async def estimate_cost(
        self, spec: InstanceSpec, expected_duration_s: float
    ) -> int:
        # Serverless is billed per-second against the endpoint's gpu
        # class — pricing lives on the endpoint definition, not per-call.
        # The hourly-cents table above is a reasonable approximation.
        rate = _DEFAULT_HOURLY_CENTS.get(spec.gpu_class, 100)
        # +25% for on-demand premium; serverless costs more per-hour but
        # saves on idle minutes, so it shows up cheaper on short runs.
        hours = max(expected_duration_s, 30) / 3600.0
        cents = int(round(rate * hours * 1.25 * spec.gpu_count))
        return cents


# Self-register only when credentials are present. Empty key means the
# adapter is compiled-in but not exposed — the dispatcher returns a
# clear error if someone tries `execution_target="runpod"` without
# keys configured.
if settings.runpod_api_key:
    register(RunPodProvider())
    register(RunPodServerlessProvider())


__all__ = ["RunPodProvider", "RunPodServerlessProvider"]
