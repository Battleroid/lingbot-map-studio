"""Pin the AWS EC2 adapter without needing boto3 installed.

The adapter lazy-imports boto3 inside a client factory; tests inject a
`FakeEc2Client` via `AwsEc2Provider.client_factory` that records calls
and returns canned responses in the exact shape boto3 would produce.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from app.cloud.providers.aws_ec2 import AwsEc2Provider
from app.cloud.providers.base import InstanceHandle, LaunchSpec
from app.jobs.schema import InstanceSpec


class _FakeEc2Client:
    """Stand-in for `boto3.client('ec2')`. Records every call and lets
    each test override a single response via the `responses` dict."""

    def __init__(self, region: str) -> None:
        self.region = region
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses: dict[str, Any] = {}
        self.raises: dict[str, Exception] = {}

    def _dispatch(self, name: str, **kwargs: Any) -> Any:
        self.calls.append((name, kwargs))
        if name in self.raises:
            raise self.raises[name]
        return self.responses.get(name, {})

    def run_instances(self, **kwargs: Any) -> Any:
        return self._dispatch("run_instances", **kwargs)

    def describe_instances(self, **kwargs: Any) -> Any:
        return self._dispatch("describe_instances", **kwargs)

    def terminate_instances(self, **kwargs: Any) -> Any:
        return self._dispatch("terminate_instances", **kwargs)


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeEc2Client(region="us-east-1")
    AwsEc2Provider.client_factory = lambda _region, _c=client: _c
    yield client
    AwsEc2Provider.client_factory = None


def _make_spec(**overrides: Any) -> LaunchSpec:
    base = {
        "job_id": "jobaws01",
        "worker_class": "slam",
        "instance_spec": InstanceSpec(
            gpu_class="a100-40g",
            disk_gb=128,
            env={"AWS_AMI_ID": "ami-0123456789"},
        ),
        "broker_url": "https://studio.example",
        "job_token": "TOK.SIG",
        "image": "lingbot-studio/worker-remote:latest",
        "env": {},
    }
    base.update(overrides)
    return LaunchSpec(**base)


async def test_launch_builds_run_instances_request(fake_client):
    fake_client.responses["run_instances"] = {
        "Instances": [{"InstanceId": "i-111"}]
    }
    provider = AwsEc2Provider()
    handle = await provider.launch(_make_spec())

    assert handle.instance_id == "i-111"
    assert handle.provider_id == "aws-ec2"
    assert handle.metadata["instance_type"] == "p4d.24xlarge"
    assert handle.metadata["ami_id"] == "ami-0123456789"

    name, kwargs = fake_client.calls[0]
    assert name == "run_instances"
    assert kwargs["ImageId"] == "ami-0123456789"
    assert kwargs["InstanceType"] == "p4d.24xlarge"
    assert kwargs["MinCount"] == 1 and kwargs["MaxCount"] == 1
    assert kwargs["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == 128

    # user_data survives the base64 round-trip and carries our env.
    decoded = base64.b64decode(kwargs["UserData"]).decode()
    assert "STUDIO_BROKER_URL=https://studio.example" in decoded
    assert "STUDIO_JOB_TOKEN=TOK.SIG" in decoded
    assert "--gpus all" in decoded

    tags = {t["Key"]: t["Value"] for t in kwargs["TagSpecifications"][0]["Tags"]}
    assert tags["lingbot:job_id"] == "jobaws01"


async def test_launch_spot_flips_market_options(fake_client):
    fake_client.responses["run_instances"] = {
        "Instances": [{"InstanceId": "i-spot"}]
    }
    provider = AwsEc2Provider()
    spec = _make_spec(
        instance_spec=InstanceSpec(
            gpu_class="a100-40g",
            spot=True,
            env={"AWS_AMI_ID": "ami-99"},
        )
    )
    await provider.launch(spec)
    kwargs = fake_client.calls[0][1]
    assert kwargs["InstanceMarketOptions"]["MarketType"] == "spot"
    assert kwargs["InstanceMarketOptions"]["SpotOptions"]["SpotInstanceType"] == "one-time"


async def test_launch_requires_ami_id():
    AwsEc2Provider.client_factory = lambda _r: _FakeEc2Client(_r)
    try:
        provider = AwsEc2Provider()
        spec = _make_spec(
            instance_spec=InstanceSpec(gpu_class="a100-40g")  # no AWS_AMI_ID
        )
        with pytest.raises(RuntimeError) as excinfo:
            await provider.launch(spec)
        assert "AMI id" in str(excinfo.value)
    finally:
        AwsEc2Provider.client_factory = None


async def test_launch_requires_known_gpu_class():
    AwsEc2Provider.client_factory = lambda _r: _FakeEc2Client(_r)
    try:
        provider = AwsEc2Provider()
        spec = _make_spec(
            instance_spec=InstanceSpec(
                gpu_class="unknown-gpu",
                env={"AWS_AMI_ID": "ami-1"},
            )
        )
        with pytest.raises(RuntimeError) as excinfo:
            await provider.launch(spec)
        assert "instance-type mapping" in str(excinfo.value)
    finally:
        AwsEc2Provider.client_factory = None


async def test_status_maps_ec2_states(fake_client):
    provider = AwsEc2Provider()
    vocab = [
        ("running", "running"),
        ("pending", "pending"),
        ("rebooting", "pending"),
        ("stopped", "terminated"),
        ("shutting-down", "terminated"),
        ("terminated", "terminated"),
    ]
    for raw, expected in vocab:
        fake_client.responses["describe_instances"] = {
            "Reservations": [{"Instances": [{"State": {"Name": raw}}]}]
        }
        got = await provider.status(
            InstanceHandle(provider_id="aws-ec2", instance_id="i-1", region="us-east-1")
        )
        assert got == expected, f"{raw} → {expected}, got {got}"


async def test_status_not_found_is_terminated(fake_client):
    provider = AwsEc2Provider()
    fake_client.raises["describe_instances"] = Exception(
        "An error occurred (InvalidInstanceID.NotFound) when calling ..."
    )
    got = await provider.status(
        InstanceHandle(provider_id="aws-ec2", instance_id="i-gone", region="us-east-1")
    )
    assert got == "terminated"


async def test_terminate_is_idempotent_on_not_found(fake_client):
    provider = AwsEc2Provider()
    fake_client.raises["terminate_instances"] = Exception(
        "InvalidInstanceID.NotFound"
    )
    # Should swallow the NotFound.
    await provider.terminate(
        InstanceHandle(provider_id="aws-ec2", instance_id="i-gone", region="us-east-1")
    )


async def test_estimate_cost_discount_for_spot():
    provider = AwsEc2Provider()
    on_demand = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-40g", spot=False), expected_duration_s=3600
    )
    spot = await provider.estimate_cost(
        InstanceSpec(gpu_class="a100-40g", spot=True), expected_duration_s=3600
    )
    assert spot < on_demand
    assert spot <= on_demand * 0.5


async def test_preflight_dry_run_operation_is_success(fake_client):
    provider = AwsEc2Provider()
    fake_client.raises["run_instances"] = Exception(
        "An error occurred (DryRunOperation) when calling the RunInstances operation"
    )
    # DryRunOperation means creds work → no exception re-raised.
    await provider.preflight(
        InstanceSpec(gpu_class="a100-40g", env={"AWS_AMI_ID": "ami-9"})
    )


async def test_preflight_unauthorized_raises(fake_client):
    provider = AwsEc2Provider()
    fake_client.raises["run_instances"] = Exception(
        "UnauthorizedOperation: ec2:RunInstances"
    )
    with pytest.raises(RuntimeError) as excinfo:
        await provider.preflight(
            InstanceSpec(gpu_class="a100-40g", env={"AWS_AMI_ID": "ami-9"})
        )
    assert "preflight failed" in str(excinfo.value)
