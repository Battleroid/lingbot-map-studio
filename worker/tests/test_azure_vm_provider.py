"""Pin the Azure VM adapter without azure-mgmt-compute installed.

Same pattern as the AWS/GCP adapters: a `_FakeAzureClient` records
calls and returns canned responses. Covers launch-body shape, spot
toggling, state-name mapping, and custom_data base64 round-trip.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from app.cloud.providers.azure_vm import AzureVmProvider
from app.cloud.providers.base import InstanceHandle, LaunchSpec
from app.jobs.schema import InstanceSpec


class _FakeStatus:
    def __init__(self, code: str) -> None:
        self.code = code


class _FakeInstanceView:
    def __init__(self, statuses: list[str]) -> None:
        self.statuses = [_FakeStatus(c) for c in statuses]


class _FakeVmsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses: dict[str, Any] = {}
        self.raises: dict[str, Exception] = {}

    def _dispatch(self, name: str, **kwargs: Any) -> Any:
        self.calls.append((name, kwargs))
        if name in self.raises:
            raise self.raises[name]
        return self.responses.get(name, object())

    def begin_create_or_update(self, **kwargs: Any) -> Any:
        return self._dispatch("begin_create_or_update", **kwargs)

    def instance_view(self, **kwargs: Any) -> Any:
        return self._dispatch("instance_view", **kwargs)

    def begin_delete(self, **kwargs: Any) -> Any:
        return self._dispatch("begin_delete", **kwargs)


class _FakeAzureClient:
    def __init__(self, subscription_id: str) -> None:
        self.subscription_id = subscription_id
        self.virtual_machines = _FakeVmsClient()


@pytest.fixture
def fake_client():
    client = _FakeAzureClient(subscription_id="sub-1")
    AzureVmProvider.client_factory = lambda _sub, _c=client: _c
    yield client
    AzureVmProvider.client_factory = None


def _make_spec(**overrides: Any) -> LaunchSpec:
    base = {
        "job_id": "jobazure1",
        "worker_class": "slam",
        "instance_spec": InstanceSpec(
            gpu_class="a100-40g",
            disk_gb=128,
            region="eastus",
            env={
                "AZURE_SUBSCRIPTION_ID": "sub-1",
                "AZURE_RESOURCE_GROUP": "lingbot-rg",
                "AZURE_IMAGE_ID": "/subscriptions/sub-1/.../images/worker-remote",
                "AZURE_NIC_ID": "/subscriptions/sub-1/.../nics/lingbot-nic-1",
            },
        ),
        "broker_url": "https://studio.example",
        "job_token": "TOK.SIG",
        "image": "lingbot-studio/worker-remote:latest",
        "env": {},
    }
    base.update(overrides)
    return LaunchSpec(**base)


async def test_launch_builds_create_vm_body(fake_client):
    provider = AzureVmProvider()
    handle = await provider.launch(_make_spec())

    assert handle.provider_id == "azure-vm"
    assert handle.instance_id == "lingbot-jobazure1"
    assert handle.metadata["resource_group"] == "lingbot-rg"
    assert handle.metadata["subscription_id"] == "sub-1"
    assert handle.metadata["vm_size"] == "Standard_NC24ads_A100_v4"

    name, kwargs = fake_client.virtual_machines.calls[0]
    assert name == "begin_create_or_update"
    assert kwargs["resource_group_name"] == "lingbot-rg"
    assert kwargs["vm_name"] == "lingbot-jobazure1"
    params = kwargs["parameters"]
    assert params["location"] == "eastus"
    assert params["hardware_profile"]["vm_size"] == "Standard_NC24ads_A100_v4"
    assert params["storage_profile"]["os_disk"]["disk_size_gb"] == 128

    decoded = base64.b64decode(params["os_profile"]["custom_data"]).decode()
    assert "STUDIO_BROKER_URL=https://studio.example" in decoded
    assert "STUDIO_JOB_TOKEN=TOK.SIG" in decoded
    assert "--gpus all" in decoded


async def test_launch_spot_flips_priority(fake_client):
    provider = AzureVmProvider()
    spec = _make_spec(
        instance_spec=InstanceSpec(
            gpu_class="a100-40g",
            spot=True,
            region="eastus",
            env={
                "AZURE_SUBSCRIPTION_ID": "sub-1",
                "AZURE_RESOURCE_GROUP": "lingbot-rg",
                "AZURE_IMAGE_ID": "img",
                "AZURE_NIC_ID": "nic",
            },
        )
    )
    await provider.launch(spec)
    params = fake_client.virtual_machines.calls[0][1]["parameters"]
    assert params["priority"] == "Spot"
    assert params["eviction_policy"] == "Delete"


async def test_launch_requires_resource_group(fake_client):
    provider = AzureVmProvider()
    spec = _make_spec(
        instance_spec=InstanceSpec(
            gpu_class="a100-40g",
            env={
                "AZURE_SUBSCRIPTION_ID": "sub-1",
                "AZURE_IMAGE_ID": "img",
                "AZURE_NIC_ID": "nic",
            },
        )
    )
    with pytest.raises(RuntimeError) as excinfo:
        await provider.launch(spec)
    assert "resource group" in str(excinfo.value)


async def test_launch_requires_nic_id(fake_client):
    provider = AzureVmProvider()
    spec = _make_spec(
        instance_spec=InstanceSpec(
            gpu_class="a100-40g",
            env={
                "AZURE_SUBSCRIPTION_ID": "sub-1",
                "AZURE_RESOURCE_GROUP": "rg",
                "AZURE_IMAGE_ID": "img",
            },
        )
    )
    with pytest.raises(RuntimeError) as excinfo:
        await provider.launch(spec)
    assert "network interface" in str(excinfo.value)


async def test_status_maps_powerstate(fake_client):
    provider = AzureVmProvider()
    handle = InstanceHandle(
        provider_id="azure-vm",
        instance_id="vm-1",
        metadata={"resource_group": "rg", "subscription_id": "sub-1"},
    )
    vocab = [
        (["PowerState/running"], "running"),
        (["PowerState/starting"], "pending"),
        (["PowerState/creating"], "pending"),
        (["PowerState/stopped"], "terminated"),
        (["PowerState/deallocated"], "terminated"),
    ]
    for raw, expected in vocab:
        fake_client.virtual_machines.responses["instance_view"] = _FakeInstanceView(raw)
        got = await provider.status(handle)
        assert got == expected, f"{raw} → {expected}, got {got}"


async def test_status_not_found_is_terminated(fake_client):
    provider = AzureVmProvider()
    fake_client.virtual_machines.raises["instance_view"] = Exception("404 Not Found")
    handle = InstanceHandle(
        provider_id="azure-vm",
        instance_id="gone",
        metadata={"resource_group": "rg", "subscription_id": "sub-1"},
    )
    assert await provider.status(handle) == "terminated"


async def test_terminate_is_idempotent_on_not_found(fake_client):
    provider = AzureVmProvider()
    fake_client.virtual_machines.raises["begin_delete"] = Exception("404 Not Found")
    handle = InstanceHandle(
        provider_id="azure-vm",
        instance_id="gone",
        metadata={"resource_group": "rg", "subscription_id": "sub-1"},
    )
    await provider.terminate(handle)


async def test_estimate_cost_spot_is_cheaper():
    provider = AzureVmProvider()
    on_demand = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-40g"), expected_duration_s=3600
    )
    spot = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-40g", spot=True), expected_duration_s=3600
    )
    assert spot < on_demand
