"""Pin the Vast.ai adapter's HTTP contract.

Vast's auction model makes launch a two-step dance (search offers,
then lease), and both halves matter: we pin the offer query shape and
the lease body independently because a wrong filter produces an
expensive silent mismatch (we pay for A100 when we asked for RTX 4090).

Uses `httpx.MockTransport` so tests run without a VAST_API_KEY. Run
with `pytest worker/tests/test_vast_provider.py -q`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.cloud.providers.base import InstanceHandle, LaunchSpec
from app.cloud.providers.vast import VastProvider
from app.jobs.schema import InstanceSpec


def _make_spec(**overrides: Any) -> LaunchSpec:
    base = {
        "job_id": "jobvast1",
        "worker_class": "slam",
        "instance_spec": InstanceSpec(gpu_class="rtx4090", disk_gb=100),
        "broker_url": "https://studio.example",
        "job_token": "TOK.SIG",
        "image": "lingbot-studio/worker-remote:latest",
        "env": {},
    }
    base.update(overrides)
    return LaunchSpec(**base)


def _install_mock(provider: VastProvider, responder) -> None:
    transport = httpx.MockTransport(responder)
    VastProvider.transport_override = transport
    provider._http._transport = transport
    provider._http._client = None


@pytest.fixture(autouse=True)
def _clear_transport():
    yield
    VastProvider.transport_override = None


async def test_launch_searches_then_leases():
    provider = VastProvider(api_key="k", api_base="https://vast.test")
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def _responder(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/bundles/":
            return httpx.Response(
                200,
                json={
                    "offers": [
                        {
                            "id": 42,
                            "dph_total": 0.41,
                            "geolocation": "US-CA",
                        }
                    ]
                },
            )
        # lease ask
        return httpx.Response(200, json={"success": True, "new_contract": 7777})

    _install_mock(provider, _responder)
    handle = await provider.launch(_make_spec())

    assert handle.instance_id == "7777"
    assert handle.provider_id == "vast"
    assert handle.region == "US-CA"
    assert handle.metadata["offer_id"] == "42"

    methods = [(m, p) for m, p, _ in calls]
    assert methods[0] == ("PUT", "/bundles/")
    assert methods[1] == ("PUT", "/asks/42/")

    # Offer filter has the canonical GPU name + disk floor + rentable flag.
    search_body = calls[0][2]
    q = search_body["q"]
    assert q["gpu_name"] == {"eq": "RTX 4090"}
    assert q["disk_space"] == {"gte": 100}
    assert q["rentable"] == {"eq": True}
    assert q["type"] == {"eq": "on-demand"}

    # Lease body carries the dispatcher env verbatim.
    lease_body = calls[1][2]
    assert lease_body["image"] == "lingbot-studio/worker-remote:latest"
    assert lease_body["disk"] == 100
    env = lease_body["env"]
    assert env["STUDIO_BROKER_URL"] == "https://studio.example"
    assert env["STUDIO_JOB_TOKEN"] == "TOK.SIG"
    assert env["WORKER_CLASS"] == "slam"
    assert env["WORKER_MODE"] == "remote"
    await provider._http.aclose()


async def test_launch_uses_interruptible_when_spot():
    provider = VastProvider(api_key="k", api_base="https://vast.test")
    captured: dict[str, Any] = {}

    def _responder(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/bundles/":
            captured["q"] = json.loads(request.content)["q"]
            return httpx.Response(
                200, json={"offers": [{"id": 1, "dph_total": 0.1}]}
            )
        return httpx.Response(200, json={"new_contract": 1})

    _install_mock(provider, _responder)
    await provider.launch(
        _make_spec(instance_spec=InstanceSpec(gpu_class="rtx4090", spot=True))
    )
    assert captured["q"]["type"] == {"eq": "interruptible"}
    await provider._http.aclose()


async def test_launch_raises_when_no_offers():
    provider = VastProvider(api_key="k", api_base="https://vast.test")

    def _responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"offers": []})

    _install_mock(provider, _responder)
    with pytest.raises(RuntimeError) as excinfo:
        await provider.launch(_make_spec())
    assert "no offers" in str(excinfo.value)
    await provider._http.aclose()


async def test_launch_raises_on_search_http_error():
    provider = VastProvider(api_key="k", api_base="https://vast.test")

    def _responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad api key"})

    _install_mock(provider, _responder)
    with pytest.raises(RuntimeError) as excinfo:
        await provider.launch(_make_spec())
    assert "401" in str(excinfo.value)
    await provider._http.aclose()


async def test_status_maps_vocabulary():
    provider = VastProvider(api_key="k", api_base="https://vast.test")
    vocab = [
        ("running", "running"),
        ("loading", "pending"),
        ("scheduling", "pending"),
        ("exited", "terminated"),
        ("stopped", "terminated"),
        ("offline", "error"),
    ]
    for raw, expected in vocab:
        _install_mock(
            provider,
            lambda _req, raw=raw: httpx.Response(
                200, json={"actual_status": raw}
            ),
        )
        result = await provider.status(
            InstanceHandle(provider_id="vast", instance_id="c1")
        )
        assert result == expected, f"{raw} → {expected}, got {result}"
    await provider._http.aclose()


async def test_status_404_is_terminated():
    provider = VastProvider(api_key="k", api_base="https://vast.test")
    _install_mock(provider, lambda _req: httpx.Response(404))
    result = await provider.status(
        InstanceHandle(provider_id="vast", instance_id="gone")
    )
    assert result == "terminated"
    await provider._http.aclose()


async def test_terminate_is_idempotent_on_404():
    provider = VastProvider(api_key="k", api_base="https://vast.test")
    calls = {"n": 0}

    def _responder(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    _install_mock(provider, _responder)
    await provider.terminate(InstanceHandle(provider_id="vast", instance_id="x"))
    await provider.terminate(InstanceHandle(provider_id="vast", instance_id="x"))
    assert calls["n"] == 2
    await provider._http.aclose()


async def test_estimate_cost_scales_with_duration_and_count():
    provider = VastProvider(api_key="k", api_base="https://vast.test")
    one_hour = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-80g"), expected_duration_s=3600
    )
    two_hours = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-80g"), expected_duration_s=7200
    )
    # With integer-cent rounding two_hours may be exactly 2x one_hour.
    assert two_hours == 2 * one_hour

    quad = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-80g", gpu_count=4), expected_duration_s=3600
    )
    assert quad == 4 * one_hour
    await provider._http.aclose()
