"""Vast.ai adapter.

Vast is an auction-driven marketplace: you don't pick a machine, you
pick a *query*, and the scheduler matches it to an offer. The adapter
models the launch as a two-step dance:

1. `search_offers` — translates `InstanceSpec` into Vast's JSON offer
   query (GPU name, disk, region, spot) and picks the cheapest
   matching offer.
2. `create_instance` — leases that offer with our remote-worker image
   and env (STUDIO_BROKER_URL + STUDIO_JOB_TOKEN + WORKER_MODE/CLASS).

Orphan handling: Vast interruptible (spot) offers get yanked
unpredictably. The orphan sweeper (already wired into the studio) has
to cross-reference `provider_instance_id` against Vast's
`/instances/{id}` every sweep; this adapter's `status()` returns
`error` for offers Vast has marked unusable so the sweeper sees them.

Credentials: `settings.vast_api_key` (env `VAST_API_KEY`).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, ClassVar, Optional

import httpx

from app.cloud.providers import register
from app.cloud.providers._http_base import ProviderHttp
from app.cloud.providers.base import (
    CloudProvider,
    InstanceHandle,
    InstanceStatus,
    LaunchSpec,
)
from app.config import settings
from app.jobs.schema import InstanceSpec

log = logging.getLogger(__name__)


# Vast's `gpu_name` filter is pretty literal — e.g. "RTX 4090". Map our
# canonical gpu_class strings onto whatever Vast expects.
_GPU_NAME: dict[str, str] = {
    "rtx3090": "RTX 3090",
    "rtx4090": "RTX 4090",
    "rtxa5000": "RTX A5000",
    "rtxa6000": "RTX A6000",
    "a40": "A40",
    "a100-40g": "A100",
    "a100-80g": "A100",
    "h100-80g": "H100",
    "h100-pcie": "H100",
    "l40s": "L40S",
    "l4": "L4",
}

# Vast publishes per-GPU market rates in $/hr; these fall-backs are used
# for `estimate_cost` when we don't want to pay for a live offer search.
_FALLBACK_HOURLY_CENTS: dict[str, int] = {
    "rtx3090": 25,
    "rtx4090": 40,
    "a100-40g": 100,
    "a100-80g": 140,
    "h100-80g": 250,
    "l40s": 80,
}


class VastProvider(CloudProvider):
    id = "vast"
    display_name = "Vast.ai (auction GPUs)"

    transport_override: ClassVar[Optional[httpx.BaseTransport]] = None

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self._http = ProviderHttp(
            base_url=api_base or settings.vast_api_base,
            headers={
                "Authorization": f"Bearer {api_key if api_key is not None else settings.vast_api_key}",
                "Content-Type": "application/json",
            },
            transport=VastProvider.transport_override,
        )

    # --- CloudProvider -------------------------------------------------

    async def launch(self, spec: LaunchSpec) -> InstanceHandle:
        # Honour test-time transport overrides set after __init__.
        if VastProvider.transport_override is not None and self._http._transport is None:
            self._http.set_transport(VastProvider.transport_override)
        client = self._http.client()

        query = self._offer_query(spec.instance_spec)
        resp = await client.put("/bundles/", json={"q": query, "order": [["dph_total", "asc"]]})
        if resp.status_code >= 400:
            raise RuntimeError(f"vast search failed ({resp.status_code}): {resp.text}")
        offers = resp.json().get("offers") or []
        if not offers:
            raise RuntimeError(
                f"vast: no offers matching {spec.instance_spec.gpu_class!r}"
                + (f" in region {spec.instance_spec.region!r}" if spec.instance_spec.region else "")
            )
        offer = offers[0]
        offer_id = offer.get("id") or offer.get("ask_contract_id")
        if offer_id is None:
            raise RuntimeError(f"vast: offer missing id: {offer!r}")

        env_map = {
            "STUDIO_BROKER_URL": spec.broker_url,
            "STUDIO_JOB_TOKEN": spec.job_token,
            "WORKER_MODE": "remote",
            "WORKER_CLASS": spec.worker_class,
            **spec.env,
        }
        create_body = {
            "client_id": "me",
            "image": spec.image,
            "disk": spec.instance_spec.disk_gb,
            "runtype": "ssh",
            "env": env_map,
            "onstart": "python -m app.worker_main",
        }
        resp = await client.put(f"/asks/{offer_id}/", json=create_body)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"vast create_instance failed ({resp.status_code}): {resp.text}"
            )
        data = resp.json()
        new_contract = data.get("new_contract")
        if not new_contract:
            raise RuntimeError(f"vast: no new_contract in response: {data!r}")
        return InstanceHandle(
            provider_id=self.id,
            instance_id=str(new_contract),
            region=offer.get("geolocation") or spec.instance_spec.region,
            gpu_class=spec.instance_spec.gpu_class,
            launched_at=time.time(),
            metadata={
                "offer_id": str(offer_id),
                "dph": str(offer.get("dph_total", "")),
            },
        )

    async def status(self, handle: InstanceHandle) -> InstanceStatus:
        client = self._http.client()
        resp = await client.get(f"/instances/{handle.instance_id}/")
        if resp.status_code == 404:
            return "terminated"
        if resp.status_code >= 400:
            return "error"
        payload = resp.json().get("instances") or resp.json()
        # Vast shape varies between calls; pick the actual_status if present.
        status = str(payload.get("actual_status") or payload.get("cur_state") or "").lower()
        if status in {"running"}:
            return "running"
        if status in {"loading", "scheduling", "created"}:
            return "pending"
        if status in {"exited", "stopped"}:
            return "terminated"
        return "error"

    async def terminate(self, handle: InstanceHandle) -> None:
        client = self._http.client()
        try:
            resp = await client.delete(f"/instances/{handle.instance_id}/")
            if resp.status_code not in (200, 204, 404):
                log.warning(
                    "vast terminate %s returned %d: %s",
                    handle.instance_id,
                    resp.status_code,
                    resp.text,
                )
        except httpx.HTTPError as exc:
            log.warning("vast terminate %s failed: %s", handle.instance_id, exc)

    async def logs(self, handle: InstanceHandle, tail: int = 200) -> str:
        # Vast exposes an SSH-only log stream for most images; best we
        # can do through the REST API is the last reported onstart log.
        client = self._http.client()
        try:
            resp = await client.get(f"/instances/{handle.instance_id}/logs/")
        except httpx.HTTPError:
            return ""
        if resp.status_code >= 400:
            return ""
        body = resp.text
        lines = body.splitlines()
        return "\n".join(lines[-tail:])

    async def estimate_cost(
        self, spec: InstanceSpec, expected_duration_s: float
    ) -> int:
        rate = _FALLBACK_HOURLY_CENTS.get(spec.gpu_class, 50)
        hours = max(expected_duration_s, 60) / 3600.0
        return int(round(rate * hours * spec.gpu_count))

    # --- internals ------------------------------------------------------

    def _offer_query(self, spec: InstanceSpec) -> dict[str, Any]:
        """Translate an InstanceSpec into Vast's offer-search filter."""
        gpu_name = _GPU_NAME.get(spec.gpu_class, spec.gpu_class.upper())
        q: dict[str, Any] = {
            "gpu_name": {"eq": gpu_name},
            "num_gpus": {"eq": spec.gpu_count},
            "disk_space": {"gte": spec.disk_gb},
            "rentable": {"eq": True},
        }
        if spec.region:
            q["geolocation"] = {"eq": spec.region}
        if spec.spot:
            q["type"] = {"eq": "interruptible"}
        else:
            q["type"] = {"eq": "on-demand"}
        return q


if settings.vast_api_key:
    register(VastProvider())


__all__ = ["VastProvider"]
