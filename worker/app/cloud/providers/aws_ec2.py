"""AWS EC2 adapter.

Raw-VM target that leans on `boto3`. The adapter builds a spot-capable
`RunInstances` request from the `InstanceSpec`, boots an AMI that has
docker + nvidia-container-runtime pre-baked, and hands it our standard
user-data script that runs the remote-worker image.

Design notes:

- **Lazy SDK import.** `boto3` is heavy (~50MB with deps) and the
  studio shouldn't require it to start. The client is built on first
  call via `client_factory`; the module itself imports cleanly even
  without boto3 on path. Registration still gates on `boto3` being
  importable so a missing SDK leaves the target invisible rather than
  producing a runtime error at launch.
- **Credentials.** We don't read keys from our own settings; instead we
  use `boto3`'s default credential chain (env, shared config, IAM
  role). That matches how AWS users expect it to behave and keeps us
  out of the secret-storage business.
- **IAM preflight.** `preflight()` does a dry-run `RunInstances` call;
  missing `ec2:RunInstances` shows up as a clear exception on submit
  rather than a silent launch failure later.
- **Spot.** `instance_spec.spot=True` flips on `InstanceMarketOptions`;
  the orphan sweeper's normal path catches preemption via `status()`
  returning `terminated`.
- **Security group.** We don't touch SGs — the user must ensure the
  subnet lets the instance reach `settings.broker_public_url` outbound
  on 443. (Most default VPCs do.)

Credentials: standard boto3 chain.
Ship image: an Amazon Linux 2023 GPU AMI published by us with docker +
`nvidia-container-runtime` pre-installed. Override via
`InstanceSpec.env["AWS_AMI_ID"]`.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Callable, ClassVar, Optional

from app.cloud.providers import register
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


ClientFactory = Callable[[str], Any]


# EC2 instance-type mapping from our canonical gpu_class onto the
# cheapest AWS family with that GPU class.
_INSTANCE_TYPE: dict[str, str] = {
    "t4": "g4dn.xlarge",
    "a10": "g5.xlarge",
    "a10g": "g5.xlarge",
    "a100-40g": "p4d.24xlarge",  # 8x A100 — AWS doesn't offer single-A100
    "a100-80g": "p4de.24xlarge",
    "h100-80g": "p5.48xlarge",  # 8x H100 minimum on AWS
    "v100": "p3.2xlarge",
}

# On-demand hourly rates in integer cents; coarse and region-dependent.
# For multi-GPU types the rate is for the whole instance.
_HOURLY_CENTS: dict[str, int] = {
    "t4": 53,
    "a10": 101,
    "a10g": 101,
    "a100-40g": 3290,
    "a100-80g": 4090,
    "h100-80g": 9820,
    "v100": 306,
}


def _default_client_factory(region: str) -> Any:
    """Production factory: construct a real `boto3.client('ec2', ...)`.

    Raises ImportError if boto3 isn't installed — the adapter checks
    importability at registration time so this path shouldn't trigger
    in production, but we surface a clear message if someone removed
    boto3 post-launch.
    """
    import boto3  # noqa: WPS433 -- deliberate lazy import

    kwargs: dict[str, Any] = {"region_name": region}
    if settings.aws_endpoint_url:
        kwargs["endpoint_url"] = settings.aws_endpoint_url
    return boto3.client("ec2", **kwargs)


class AwsEc2Provider(CloudProvider):
    id = "aws-ec2"
    display_name = "AWS EC2 (GPU)"

    #: Tests override to inject a stub that records calls + returns
    #: canned responses. Production leaves this None (falls back to
    #: `_default_client_factory`).
    client_factory: ClassVar[Optional[ClientFactory]] = None

    def _client(self, region: Optional[str] = None) -> Any:
        # Access via the class, not `self`, so a callable assigned at
        # class level doesn't get descriptor-bound into a method (which
        # would prepend `self` as the first arg).
        factory = type(self).client_factory or _default_client_factory
        return factory(region or settings.aws_region_default)

    async def launch(self, spec: LaunchSpec) -> InstanceHandle:
        instance_type = (
            spec.instance_spec.env.get("AWS_INSTANCE_TYPE")
            or _INSTANCE_TYPE.get(spec.instance_spec.gpu_class)
        )
        if not instance_type:
            raise RuntimeError(
                f"aws-ec2: no instance-type mapping for "
                f"gpu_class={spec.instance_spec.gpu_class!r}. Set "
                "instance_spec.env['AWS_INSTANCE_TYPE'] explicitly."
            )
        ami_id = spec.instance_spec.env.get("AWS_AMI_ID") or ""
        if not ami_id:
            raise RuntimeError(
                "aws-ec2: no AMI id provided. Set "
                "instance_spec.env['AWS_AMI_ID'] to an AMI with docker + "
                "nvidia-container-runtime pre-installed."
            )
        region = spec.instance_spec.region or settings.aws_region_default
        user_data = build_bootstrap_script(
            spec,
            extra_docker_args=("--gpus all", "--shm-size=8g"),
        )
        user_data_b64 = base64.b64encode(user_data.encode("utf-8")).decode("ascii")

        request: dict[str, Any] = {
            "ImageId": ami_id,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "UserData": user_data_b64,
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/xvda",
                    "Ebs": {
                        "VolumeSize": spec.instance_spec.disk_gb,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"lingbot-{spec.job_id}"},
                        {"Key": "lingbot:job_id", "Value": spec.job_id},
                        {"Key": "lingbot:worker_class", "Value": spec.worker_class},
                    ],
                }
            ],
        }
        if spec.instance_spec.spot:
            # `one-time` so AWS doesn't retry the spot request after our
            # one-shot job exits — we never want a zombie spot replacement.
            request["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {"SpotInstanceType": "one-time"},
            }

        client = self._client(region)
        resp = await _maybe_await(client.run_instances(**request))
        instances = resp.get("Instances") or []
        if not instances:
            raise RuntimeError(f"aws-ec2: run_instances returned no instances: {resp!r}")
        inst = instances[0]
        return InstanceHandle(
            provider_id=self.id,
            instance_id=str(inst["InstanceId"]),
            region=region,
            gpu_class=spec.instance_spec.gpu_class,
            launched_at=time.time(),
            metadata={
                "instance_type": instance_type,
                "ami_id": ami_id,
                "spot": "1" if spec.instance_spec.spot else "0",
            },
        )

    async def status(self, handle: InstanceHandle) -> InstanceStatus:
        client = self._client(handle.region)
        try:
            resp = await _maybe_await(
                client.describe_instances(InstanceIds=[handle.instance_id])
            )
        except Exception as exc:  # noqa: BLE001
            # InvalidInstanceID.NotFound → treated as terminated; anything
            # else surfaces as `error` so the sweeper retries or fails.
            msg = str(exc).lower()
            if "notfound" in msg or "does not exist" in msg:
                return "terminated"
            log.warning("aws-ec2 describe_instances failed: %s", exc)
            return "error"
        reservations = resp.get("Reservations") or []
        instances = reservations[0].get("Instances") if reservations else []
        if not instances:
            return "terminated"
        state = str(instances[0].get("State", {}).get("Name", "")).lower()
        if state == "running":
            return "running"
        if state in {"pending", "rebooting"}:
            return "pending"
        if state in {"stopping", "stopped", "shutting-down", "terminated"}:
            return "terminated"
        return "error"

    async def terminate(self, handle: InstanceHandle) -> None:
        client = self._client(handle.region)
        try:
            await _maybe_await(
                client.terminate_instances(InstanceIds=[handle.instance_id])
            )
        except Exception as exc:  # noqa: BLE001
            # `InvalidInstanceID.NotFound` means it's already gone — fine.
            msg = str(exc).lower()
            if "notfound" in msg or "does not exist" in msg:
                return
            log.warning("aws-ec2 terminate %s failed: %s", handle.instance_id, exc)

    async def logs(self, handle: InstanceHandle, tail: int = 200) -> str:
        # EC2 exposes `get_console_output` but it's noisy (full kernel
        # boot log) and rate-limited. The broker-published events are
        # the authoritative source; fall back to empty.
        return ""

    async def estimate_cost(
        self, spec: InstanceSpec, expected_duration_s: float
    ) -> int:
        rate = _HOURLY_CENTS.get(spec.gpu_class, 500)
        if spec.spot:
            # Spot typically ~30% of on-demand; be conservative at 40%.
            rate = int(round(rate * 0.4))
        hours = max(expected_duration_s, 60) / 3600.0
        return int(round(rate * hours))

    async def preflight(self, spec: InstanceSpec) -> None:
        """Dry-run the launch to surface IAM / quota errors on submit.

        Boto3's EC2 client accepts `DryRun=True` on RunInstances; a
        successful dry-run raises with `DryRunOperation` (expected). Any
        other exception — particularly `UnauthorizedOperation` — is the
        real problem and we re-raise with a cleaner message.
        """
        instance_type = (
            spec.env.get("AWS_INSTANCE_TYPE")
            or _INSTANCE_TYPE.get(spec.gpu_class)
        )
        ami_id = spec.env.get("AWS_AMI_ID") or ""
        if not instance_type or not ami_id:
            return  # launch() will raise with the right message later.
        client = self._client(spec.region or settings.aws_region_default)
        try:
            await _maybe_await(
                client.run_instances(
                    DryRun=True,
                    ImageId=ami_id,
                    InstanceType=instance_type,
                    MinCount=1,
                    MaxCount=1,
                )
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "DryRunOperation" in msg:
                return  # expected: credentials work
            raise RuntimeError(f"aws-ec2 preflight failed: {msg}") from exc


async def _maybe_await(result: Any) -> Any:
    """Allow both real boto3 (sync) and test stubs (sync OR async). Keeps
    the adapter an async-compatible `CloudProvider` without forcing the
    test double to pretend to be async."""
    if hasattr(result, "__await__"):
        return await result
    return result


# boto3 is an optional dep; only register when it's importable so a
# fresh dev install without AWS creds leaves the target invisible.
try:
    import boto3  # noqa: F401 -- import-time check

    register(AwsEc2Provider())
except ImportError:
    pass


__all__ = ["AwsEc2Provider"]
