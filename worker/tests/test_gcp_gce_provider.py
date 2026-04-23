"""Pin the GCP Compute Engine adapter without google-cloud-compute installed.

Same pattern as `test_aws_ec2_provider.py`: a `_FakeGceClient`
records calls and returns canned responses. Focus on launch-body
shape, state-name mapping, and preemptible toggling.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.cloud.providers.base import InstanceHandle, LaunchSpec
from app.cloud.providers.gcp_gce import GcpGceProvider
from app.jobs.schema import InstanceSpec


class _FakeGceInstance:
    def __init__(self, status: str) -> None:
        self.status = status


class _FakeGceClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses: dict[str, Any] = {}
        self.raises: dict[str, Exception] = {}

    def _dispatch(self, name: str, **kwargs: Any) -> Any:
        self.calls.append((name, kwargs))
        if name in self.raises:
            raise self.raises[name]
        return self.responses.get(name, object())

    def insert(self, **kwargs: Any) -> Any:
        return self._dispatch("insert", **kwargs)

    def get(self, **kwargs: Any) -> Any:
        return self._dispatch("get", **kwargs)

    def delete(self, **kwargs: Any) -> Any:
        return self._dispatch("delete", **kwargs)


@pytest.fixture
def fake_client():
    client = _FakeGceClient()
    GcpGceProvider.client_factory = lambda _c=client: _c
    yield client
    GcpGceProvider.client_factory = None


def _make_spec(**overrides: Any) -> LaunchSpec:
    base = {
        "job_id": "jobgcp01",
        "worker_class": "gs",
        "instance_spec": InstanceSpec(
            gpu_class="a100-40g",
            disk_gb=128,
            region="us-central1",
            env={
                "GCP_PROJECT_ID": "lingbot-stg",
                "GCP_IMAGE_URI": "projects/lingbot-stg/global/images/worker-remote-1",
            },
        ),
        "broker_url": "https://studio.example",
        "job_token": "TOK.SIG",
        "image": "lingbot-studio/worker-remote:latest",
        "env": {},
    }
    base.update(overrides)
    return LaunchSpec(**base)


async def test_launch_builds_insert_body(fake_client):
    provider = GcpGceProvider()
    handle = await provider.launch(_make_spec())

    assert handle.provider_id == "gcp-gce"
    assert handle.instance_id == "lingbot-jobgcp01"
    assert handle.metadata["project"] == "lingbot-stg"
    assert handle.metadata["zone"] == "us-central1-a"

    name, kwargs = fake_client.calls[0]
    assert name == "insert"
    assert kwargs["project"] == "lingbot-stg"
    assert kwargs["zone"] == "us-central1-a"
    body = kwargs["instance_resource"]
    assert body["name"] == "lingbot-jobgcp01"
    assert "a2-highgpu-1g" in body["machine_type"]
    assert body["guest_accelerators"][0]["accelerator_count"] == 1
    assert "nvidia-tesla-a100" in body["guest_accelerators"][0]["accelerator_type"]
    # startup-script metadata carries the bootstrap script.
    metadata_items = body["metadata"]["items"]
    startup = next(i for i in metadata_items if i["key"] == "startup-script")
    assert "STUDIO_BROKER_URL=https://studio.example" in startup["value"]
    assert "--gpus all" in startup["value"]


async def test_launch_spot_flips_preemptible(fake_client):
    provider = GcpGceProvider()
    spec = _make_spec(
        instance_spec=InstanceSpec(
            gpu_class="a100-40g",
            spot=True,
            region="us-central1",
            env={
                "GCP_PROJECT_ID": "p",
                "GCP_IMAGE_URI": "image-uri",
            },
        )
    )
    await provider.launch(spec)
    body = fake_client.calls[0][1]["instance_resource"]
    assert body["scheduling"]["preemptible"] is True
    assert body["scheduling"]["provisioning_model"] == "SPOT"


async def test_launch_requires_project_id(fake_client):
    provider = GcpGceProvider()
    spec = _make_spec(
        instance_spec=InstanceSpec(
            gpu_class="a100-40g",
            env={"GCP_IMAGE_URI": "image-uri"},
        )
    )
    with pytest.raises(RuntimeError) as excinfo:
        await provider.launch(spec)
    assert "project id" in str(excinfo.value)


async def test_launch_requires_image_uri(fake_client):
    provider = GcpGceProvider()
    spec = _make_spec(
        instance_spec=InstanceSpec(
            gpu_class="a100-40g",
            env={"GCP_PROJECT_ID": "p"},
        )
    )
    with pytest.raises(RuntimeError) as excinfo:
        await provider.launch(spec)
    assert "source image" in str(excinfo.value)


async def test_status_maps_gce_states(fake_client):
    provider = GcpGceProvider()
    handle = InstanceHandle(
        provider_id="gcp-gce",
        instance_id="lingbot-x",
        metadata={"project": "p", "zone": "us-central1-a"},
    )
    vocab = [
        ("RUNNING", "running"),
        ("PROVISIONING", "pending"),
        ("STAGING", "pending"),
        ("STOPPED", "terminated"),
        ("TERMINATED", "terminated"),
        ("SUSPENDED", "terminated"),
    ]
    for raw, expected in vocab:
        fake_client.responses["get"] = _FakeGceInstance(status=raw)
        got = await provider.status(handle)
        assert got == expected, f"{raw} → {expected}, got {got}"


async def test_status_not_found_is_terminated(fake_client):
    provider = GcpGceProvider()
    fake_client.raises["get"] = Exception("404 Not Found")
    handle = InstanceHandle(
        provider_id="gcp-gce",
        instance_id="gone",
        metadata={"project": "p", "zone": "us-central1-a"},
    )
    assert await provider.status(handle) == "terminated"


async def test_terminate_is_idempotent_on_not_found(fake_client):
    provider = GcpGceProvider()
    fake_client.raises["delete"] = Exception("404 Not Found")
    handle = InstanceHandle(
        provider_id="gcp-gce",
        instance_id="gone",
        metadata={"project": "p", "zone": "us-central1-a"},
    )
    await provider.terminate(handle)


async def test_estimate_cost_spot_is_cheaper():
    provider = GcpGceProvider()
    on_demand = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-40g"), expected_duration_s=3600
    )
    spot = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-40g", spot=True), expected_duration_s=3600
    )
    assert spot < on_demand
