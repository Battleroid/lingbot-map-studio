"""Pin the Lambda Labs adapter's HTTP contract.

Lambda runs VMs, not containers, so we verify two things beyond the
usual `CloudProvider` surface:

- The launch body targets the right `instance_type_name` for our
  `gpu_class` mapping.
- The `user_data` field carries a docker-bootstrap script that runs
  our remote-worker image with the dispatcher env injected. That's the
  whole point of `_ssh_bootstrap.py`; if this test passes we know
  `build_bootstrap_script` stays wired in.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.cloud.providers.base import InstanceHandle, LaunchSpec
from app.cloud.providers.lambda_labs import LambdaLabsProvider
from app.jobs.schema import InstanceSpec


def _make_spec(**overrides: Any) -> LaunchSpec:
    base = {
        "job_id": "joblambda",
        "worker_class": "gs",
        "instance_spec": InstanceSpec(gpu_class="a100-40g"),
        "broker_url": "https://studio.example",
        "job_token": "TOK.SIG",
        "image": "lingbot-studio/worker-remote:latest",
        "env": {},
    }
    base.update(overrides)
    return LaunchSpec(**base)


def _install_mock(provider: LambdaLabsProvider, responder) -> None:
    transport = httpx.MockTransport(responder)
    LambdaLabsProvider.transport_override = transport
    provider._http._transport = transport
    provider._http._client = None


@pytest.fixture(autouse=True)
def _clear_transport():
    yield
    LambdaLabsProvider.transport_override = None


async def test_launch_sends_mapped_instance_type_and_user_data():
    provider = LambdaLabsProvider(api_key="lk", api_base="https://lambda.test")
    captured: dict[str, Any] = {}

    def _responder(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/instance-operations/launch"
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"instance_ids": ["i-abc"]}})

    _install_mock(provider, _responder)
    handle = await provider.launch(_make_spec())

    assert handle.instance_id == "i-abc"
    assert handle.provider_id == "lambda_labs"
    assert handle.metadata["instance_type"] == "gpu_1x_a100"
    assert captured["auth"] == "Bearer lk"

    body = captured["body"]
    assert body["instance_type_name"] == "gpu_1x_a100"
    assert body["quantity"] == 1
    assert body["name"] == "lingbot-joblambda"
    user_data = body["user_data"]
    # Bootstrap script carries the studio env and the image name.
    assert "STUDIO_BROKER_URL=https://studio.example" in user_data
    assert "STUDIO_JOB_TOKEN=TOK.SIG" in user_data
    assert "WORKER_CLASS=gs" in user_data
    assert "lingbot-studio/worker-remote:latest" in user_data
    # GPU flags are injected by the adapter.
    assert "--gpus all" in user_data
    await provider._http.aclose()


async def test_launch_respects_explicit_instance_type_override():
    provider = LambdaLabsProvider(api_key="k", api_base="https://lambda.test")
    captured: dict[str, Any] = {}

    def _responder(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"instance_ids": ["i-h100"]}})

    _install_mock(provider, _responder)
    spec = _make_spec(
        instance_spec=InstanceSpec(
            gpu_class="h100-80g",
            env={"LAMBDA_INSTANCE_TYPE": "gpu_8x_h100_sxm5"},
        )
    )
    await provider.launch(spec)
    assert captured["body"]["instance_type_name"] == "gpu_8x_h100_sxm5"
    await provider._http.aclose()


async def test_launch_raises_when_gpu_class_unmapped():
    provider = LambdaLabsProvider(api_key="k", api_base="https://lambda.test")
    _install_mock(provider, lambda _req: httpx.Response(500))
    with pytest.raises(RuntimeError) as excinfo:
        await provider.launch(
            _make_spec(instance_spec=InstanceSpec(gpu_class="made-up-gpu"))
        )
    assert "no instance_type mapping" in str(excinfo.value)
    await provider._http.aclose()


async def test_launch_raises_on_http_error():
    provider = LambdaLabsProvider(api_key="k", api_base="https://lambda.test")

    def _responder(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "capacity"})

    _install_mock(provider, _responder)
    with pytest.raises(RuntimeError) as excinfo:
        await provider.launch(_make_spec())
    assert "403" in str(excinfo.value)
    await provider._http.aclose()


async def test_status_maps_vocabulary():
    provider = LambdaLabsProvider(api_key="k", api_base="https://lambda.test")
    vocab = [
        ("active", "running"),
        ("booting", "pending"),
        ("unhealthy", "error"),
        ("terminated", "terminated"),
    ]
    for raw, expected in vocab:
        _install_mock(
            provider,
            lambda _req, raw=raw: httpx.Response(
                200, json={"data": {"status": raw}}
            ),
        )
        result = await provider.status(
            InstanceHandle(provider_id="lambda_labs", instance_id="i1")
        )
        assert result == expected, f"{raw} → {expected}, got {result}"
    await provider._http.aclose()


async def test_status_404_is_terminated():
    provider = LambdaLabsProvider(api_key="k", api_base="https://lambda.test")
    _install_mock(provider, lambda _req: httpx.Response(404))
    result = await provider.status(
        InstanceHandle(provider_id="lambda_labs", instance_id="gone")
    )
    assert result == "terminated"
    await provider._http.aclose()


async def test_terminate_is_idempotent_on_404():
    provider = LambdaLabsProvider(api_key="k", api_base="https://lambda.test")
    calls = {"n": 0}

    def _responder(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    _install_mock(provider, _responder)
    await provider.terminate(
        InstanceHandle(provider_id="lambda_labs", instance_id="x")
    )
    await provider.terminate(
        InstanceHandle(provider_id="lambda_labs", instance_id="x")
    )
    assert calls["n"] == 2
    await provider._http.aclose()


async def test_estimate_cost_matches_hourly_rate():
    provider = LambdaLabsProvider(api_key="k", api_base="https://lambda.test")
    one_hour = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-40g"), expected_duration_s=3600
    )
    # Hardcoded fallback is 129 ¢/hr for a100-40g.
    assert one_hour == 129

    quad = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-40g", gpu_count=4), expected_duration_s=3600
    )
    assert quad == 4 * one_hour
    await provider._http.aclose()
