"""Pin the RunPod adapter's HTTP contract.

Real RunPod launches cost money and need credentials, so these tests
use `httpx.MockTransport` to stand in for the RunPod REST API. The
point is to verify the adapter makes the calls the plan says it makes:

- `launch()` POSTs to `/pods` with the dispatcher's broker URL + token
  baked into the env array, parses the returned pod id, and stashes it
  on the handle.
- `status()` maps RunPod's vocabulary (RUNNING / EXITED / FAILED) onto
  our compact `InstanceStatus` literal.
- `terminate()` is idempotent — a 404 mid-teardown must not raise.
- `estimate_cost()` is deterministic and respects the `spot` discount.

Run: `pytest worker/tests/test_runpod_provider.py -q`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.cloud.providers.base import InstanceHandle, LaunchSpec
from app.cloud.providers.runpod import RunPodProvider, RunPodServerlessProvider
from app.jobs.schema import InstanceSpec


def _make_spec(**overrides: Any) -> LaunchSpec:
    """Build a LaunchSpec matching what dispatcher.launch would hand the
    provider. Tests override one field at a time to isolate whatever
    they're pinning."""
    base = {
        "job_id": "jobabc123",
        "worker_class": "slam",
        "instance_spec": InstanceSpec(gpu_class="rtx4090", disk_gb=80),
        "broker_url": "https://studio.example",
        "job_token": "TOKEN.SIG",
        "image": "lingbot-studio/worker-remote:latest",
        "env": {},
    }
    base.update(overrides)
    return LaunchSpec(**base)


def _install_mock(provider, responder):
    """Point `provider`'s httpx client at a MockTransport built from
    `responder(request) -> httpx.Response`. Resets the cached client so
    the next call actually sees the new transport."""
    RunPodProvider.transport_override = httpx.MockTransport(responder)
    provider._client = None  # force rebuild on next call


@pytest.fixture(autouse=True)
def _clear_transport():
    yield
    RunPodProvider.transport_override = None


async def test_launch_sends_expected_body_and_parses_pod_id():
    provider = RunPodProvider(api_key="fake-key", api_base="https://rp.test")
    captured: dict[str, Any] = {}

    def _responder(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/pods"
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"id": "pod-xyz", "dataCenterId": "US-OR-1"})

    _install_mock(provider, _responder)
    handle = await provider.launch(_make_spec())

    assert handle.instance_id == "pod-xyz"
    assert handle.region == "US-OR-1"
    assert handle.provider_id == "runpod"
    assert captured["auth"] == "Bearer fake-key"

    body = captured["body"]
    # Standard fields we require the adapter to set on every call.
    assert body["gpuTypeId"] == "NVIDIA GeForce RTX 4090"
    assert body["cloudType"] == "SECURE"
    assert body["gpuCount"] == 1
    assert body["volumeInGb"] == 80
    env_map = {e["key"]: e["value"] for e in body["env"]}
    # Dispatcher env must survive into the pod env verbatim.
    assert env_map["STUDIO_BROKER_URL"] == "https://studio.example"
    assert env_map["STUDIO_JOB_TOKEN"] == "TOKEN.SIG"
    assert env_map["WORKER_CLASS"] == "slam"
    assert env_map["WORKER_MODE"] == "remote"
    await provider.aclose()


async def test_launch_uses_community_cloud_for_spot():
    provider = RunPodProvider(api_key="k", api_base="https://rp.test")
    captured: dict[str, Any] = {}

    def _responder(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "pod-1"})

    _install_mock(provider, _responder)
    spec = _make_spec(instance_spec=InstanceSpec(gpu_class="rtx4090", spot=True))
    await provider.launch(spec)
    assert captured["body"]["cloudType"] == "COMMUNITY"
    await provider.aclose()


async def test_launch_respects_gpu_type_override_env():
    provider = RunPodProvider(api_key="k", api_base="https://rp.test")
    captured: dict[str, Any] = {}

    def _responder(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "pod-override"})

    _install_mock(provider, _responder)
    spec = _make_spec(
        instance_spec=InstanceSpec(
            gpu_class="h100-80g",
            env={"RUNPOD_GPU_TYPE_ID": "NVIDIA H100 PCIe 80GB"},
        )
    )
    await provider.launch(spec)
    assert captured["body"]["gpuTypeId"] == "NVIDIA H100 PCIe 80GB"
    await provider.aclose()


async def test_launch_raises_on_http_error():
    provider = RunPodProvider(api_key="k", api_base="https://rp.test")

    def _responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": "out of credits"})

    _install_mock(provider, _responder)
    with pytest.raises(RuntimeError) as excinfo:
        await provider.launch(_make_spec())
    assert "402" in str(excinfo.value)
    await provider.aclose()


async def test_status_maps_runpod_vocabulary():
    provider = RunPodProvider(api_key="k", api_base="https://rp.test")
    calls = {"n": 0}
    vocab = [
        ("CREATED", "pending"),
        ("RUNNING", "running"),
        ("EXITED", "terminated"),
        ("FAILED", "error"),
    ]
    expectations = iter(vocab)

    def _responder(request: httpx.Request) -> httpx.Response:
        _, expected = next_vocab[0]
        return httpx.Response(200, json={"desiredStatus": next_vocab[0][0]})

    for raw, expected in vocab:
        next_vocab = [(raw, expected)]
        _install_mock(provider, _responder)
        result = await provider.status(
            InstanceHandle(provider_id="runpod", instance_id="pod-x")
        )
        assert result == expected, f"{raw} should map to {expected}, got {result}"
    await provider.aclose()


async def test_status_404_is_terminated():
    provider = RunPodProvider(api_key="k", api_base="https://rp.test")
    _install_mock(provider, lambda _req: httpx.Response(404))
    assert (
        await provider.status(InstanceHandle(provider_id="runpod", instance_id="gone"))
        == "terminated"
    )
    await provider.aclose()


async def test_terminate_is_idempotent_on_404():
    provider = RunPodProvider(api_key="k", api_base="https://rp.test")
    calls = {"n": 0}

    def _responder(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    _install_mock(provider, _responder)
    # Should not raise.
    await provider.terminate(InstanceHandle(provider_id="runpod", instance_id="gone"))
    await provider.terminate(InstanceHandle(provider_id="runpod", instance_id="gone"))
    assert calls["n"] == 2
    await provider.aclose()


async def test_estimate_cost_respects_spot_discount():
    provider = RunPodProvider(api_key="k", api_base="https://rp.test")
    on_demand = await provider.estimate_cost(
        InstanceSpec(gpu_class="rtx4090", spot=False), expected_duration_s=3600
    )
    community = await provider.estimate_cost(
        InstanceSpec(gpu_class="rtx4090", spot=True), expected_duration_s=3600
    )
    assert community < on_demand
    # Sanity: 60 cents/hr * 0.6 spot factor ≈ 30¢. Allow ±10% slop.
    assert community <= on_demand * 0.7
    await provider.aclose()


async def test_estimate_cost_scales_with_gpu_count():
    provider = RunPodProvider(api_key="k", api_base="https://rp.test")
    single = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-80g", gpu_count=1), expected_duration_s=3600
    )
    quad = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-80g", gpu_count=4), expected_duration_s=3600
    )
    assert quad == single * 4
    await provider.aclose()


# --- Serverless ------------------------------------------------------


async def test_serverless_launch_requires_endpoint_id():
    provider = RunPodServerlessProvider(api_key="k", api_base="https://rp.test")
    with pytest.raises(RuntimeError) as excinfo:
        await provider.launch(_make_spec())
    assert "RUNPOD_ENDPOINT_ID" in str(excinfo.value)


async def test_serverless_launch_posts_to_endpoint_run():
    provider = RunPodServerlessProvider(api_key="k", api_base="https://rp.test")
    captured: dict[str, Any] = {}

    def _responder(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "sv-req-1"})

    _install_mock(provider, _responder)
    spec = _make_spec(
        instance_spec=InstanceSpec(
            gpu_class="rtx4090",
            env={"RUNPOD_ENDPOINT_ID": "ep-123"},
        )
    )
    handle = await provider.launch(spec)
    assert handle.instance_id == "sv-req-1"
    assert handle.metadata["endpoint_id"] == "ep-123"
    assert captured["path"] == "/ep-123/run"
    assert captured["body"]["input"]["studio_job_token"] == "TOKEN.SIG"
