"""Pin both Paperspace adapters: Gradient (managed) + Core (raw VM).

Paperspace is the only provider that exposes two distinct launch
surfaces under one credential. We test them as separate providers so
the dispatcher's registry stays honest — each sub-target maps to a
real `CloudProvider.id`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.cloud.providers.base import InstanceHandle, LaunchSpec
from app.cloud.providers.paperspace import (
    PaperspaceCoreProvider,
    PaperspaceGradientProvider,
)
from app.jobs.schema import InstanceSpec


def _make_spec(**overrides: Any) -> LaunchSpec:
    base = {
        "job_id": "jobps001",
        "worker_class": "slam",
        "instance_spec": InstanceSpec(gpu_class="a100-40g"),
        "broker_url": "https://studio.example",
        "job_token": "TOK.SIG",
        "image": "lingbot-studio/worker-remote:latest",
        "env": {},
    }
    base.update(overrides)
    return LaunchSpec(**base)


def _install_gradient(provider: PaperspaceGradientProvider, responder) -> None:
    transport = httpx.MockTransport(responder)
    PaperspaceGradientProvider.transport_override = transport
    provider._http._transport = transport
    provider._http._client = None


def _install_core(provider: PaperspaceCoreProvider, responder) -> None:
    transport = httpx.MockTransport(responder)
    PaperspaceCoreProvider.transport_override = transport
    provider._http._transport = transport
    provider._http._client = None


@pytest.fixture(autouse=True)
def _clear_transport():
    yield
    PaperspaceGradientProvider.transport_override = None
    PaperspaceCoreProvider.transport_override = None


# --- Gradient -------------------------------------------------------------


async def test_gradient_launch_posts_deployment_spec():
    provider = PaperspaceGradientProvider(api_key="pk", api_base="https://ps.test")
    captured: dict[str, Any] = {}

    def _responder(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/deployments"
        captured["auth"] = request.headers.get("x-api-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "dep-123"})

    _install_gradient(provider, _responder)
    handle = await provider.launch(_make_spec())

    assert handle.provider_id == "paperspace-gradient"
    assert handle.instance_id == "dep-123"
    assert handle.metadata["machine_type"] == "A100"
    assert captured["auth"] == "pk"

    body = captured["body"]
    assert body["name"] == "lingbot-jobps001"
    assert body["spec"]["image"] == "lingbot-studio/worker-remote:latest"
    assert body["spec"]["resources"]["instanceType"] == "A100"
    env = {e["name"]: e["value"] for e in body["spec"]["env"]}
    assert env["STUDIO_BROKER_URL"] == "https://studio.example"
    assert env["STUDIO_JOB_TOKEN"] == "TOK.SIG"
    assert env["WORKER_CLASS"] == "slam"
    assert env["WORKER_MODE"] == "remote"
    await provider._http.aclose()


async def test_gradient_status_vocabulary():
    provider = PaperspaceGradientProvider(api_key="k", api_base="https://ps.test")
    vocab = [
        ("Ready", "running"),
        ("Running", "running"),
        ("Provisioning", "pending"),
        ("Scaling", "pending"),
        ("Stopped", "terminated"),
        ("Error", "error"),
    ]
    for raw, expected in vocab:
        _install_gradient(
            provider,
            lambda _req, raw=raw: httpx.Response(200, json={"state": raw}),
        )
        got = await provider.status(
            InstanceHandle(provider_id="paperspace-gradient", instance_id="d1")
        )
        assert got == expected, f"{raw} → {expected}, got {got}"
    await provider._http.aclose()


async def test_gradient_terminate_is_idempotent_on_404():
    provider = PaperspaceGradientProvider(api_key="k", api_base="https://ps.test")
    _install_gradient(provider, lambda _req: httpx.Response(404))
    await provider.terminate(
        InstanceHandle(provider_id="paperspace-gradient", instance_id="x")
    )
    await provider._http.aclose()


async def test_gradient_estimate_cost_scales():
    provider = PaperspaceGradientProvider(api_key="k", api_base="https://ps.test")
    single = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-40g"), expected_duration_s=3600
    )
    quad = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-40g", gpu_count=4), expected_duration_s=3600
    )
    assert quad == 4 * single
    await provider._http.aclose()


# --- Core -----------------------------------------------------------------


async def test_core_launch_posts_user_data_bootstrap():
    provider = PaperspaceCoreProvider(api_key="pk", api_base="https://ps.test")
    captured: dict[str, Any] = {}

    def _responder(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/machines/createSingleMachinePublic"
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "mach-9"})

    _install_core(provider, _responder)
    handle = await provider.launch(_make_spec())

    assert handle.provider_id == "paperspace-core"
    assert handle.instance_id == "mach-9"
    body = captured["body"]
    assert body["machineType"] == "A100"
    assert body["startOnCreate"] is True
    user_data = body["userData"]
    assert "STUDIO_BROKER_URL=https://studio.example" in user_data
    assert "STUDIO_JOB_TOKEN=TOK.SIG" in user_data
    assert "--gpus all" in user_data
    await provider._http.aclose()


async def test_core_status_vocabulary():
    provider = PaperspaceCoreProvider(api_key="k", api_base="https://ps.test")
    vocab = [
        ("ready", "running"),
        ("serviceready", "running"),
        ("provisioning", "pending"),
        ("starting", "pending"),
        ("off", "terminated"),
        ("stopped", "terminated"),
    ]
    for raw, expected in vocab:
        _install_core(
            provider,
            lambda _req, raw=raw: httpx.Response(200, json={"state": raw}),
        )
        got = await provider.status(
            InstanceHandle(provider_id="paperspace-core", instance_id="m1")
        )
        assert got == expected, f"{raw} → {expected}, got {got}"
    await provider._http.aclose()


async def test_core_terminate_is_idempotent_on_404():
    provider = PaperspaceCoreProvider(api_key="k", api_base="https://ps.test")
    calls = {"n": 0}

    def _responder(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    _install_core(provider, _responder)
    await provider.terminate(
        InstanceHandle(provider_id="paperspace-core", instance_id="x")
    )
    await provider.terminate(
        InstanceHandle(provider_id="paperspace-core", instance_id="x")
    )
    assert calls["n"] == 2
    await provider._http.aclose()
