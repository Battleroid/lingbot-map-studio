"""`CloudProvider` ABC — the uniform surface every remote-execution adapter
implements. The dispatcher only ever talks to this interface; adapters
internally translate to/from their native API.

Design constraints:

- `launch()` returns quickly with an `InstanceHandle` once the instance is
  scheduled, even if boot is still in flight. Actually-running-and-claiming
  is an observable condition (the broker sees the pod claim a job), not a
  precondition of this call. That keeps the API responsive: a slow provider
  doesn't block the `POST /api/jobs` request for 90 seconds.
- `terminate()` must be idempotent. We call it both on normal completion
  (the finally block of `dispatcher.launch`'s watcher) and from the orphan
  sweeper if a pod goes missing — calling twice must be safe.
- `estimate_cost()` is side-effect-free. The UI calls it on a debounce from
  `CostPreview`; it must not count against provider rate limits.
- `status()` maps the provider's native state string to our compact
  `InstanceStatus` literal so the sweeper + UI don't branch on
  provider-specific strings.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal, Optional

from app.jobs.schema import InstanceSpec


# Compact lifecycle state for a provider-side instance, independent of
# provider-specific vocabulary (RunPod's "RUNNING" vs AWS's "running" vs
# Vast's "loaded"). The sweeper only needs the 4 values below.
InstanceStatus = Literal["pending", "running", "terminated", "error"]


@dataclass
class InstanceHandle:
    """Opaque reference to a launched instance.

    The dispatcher persists `instance_id` onto the job row so the sweeper
    can reconcile "job running but provider says terminated" without
    remembering provider-specific shapes. `metadata` is free-form and goes
    straight into the job row's JSON blob — each provider uses it to
    stash whatever it needs to find the instance again (pod id, region,
    AMI, ssh host, …).
    """

    provider_id: str
    instance_id: str
    # Region / availability zone / cluster name — useful for the UI badge
    # and for the terminate path when the API is region-scoped.
    region: Optional[str] = None
    # GPU class actually assigned (may differ from the requested one if
    # the provider fell back to a compatible class on availability).
    gpu_class: Optional[str] = None
    # Monotonic launch timestamp (unix seconds) for elapsed-time cost
    # rollups. Filled in by `CloudProvider.launch`.
    launched_at: float = 0.0
    # Provider-specific extras the adapter needs to carry between calls.
    # Kept flat (strings only) so it can round-trip through JSON.
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class LaunchSpec:
    """Everything a provider needs to stand up a remote worker.

    Built by `dispatcher.launch` — providers never construct this
    themselves. Keeping the dispatcher as the sole builder means the
    broker URL, HMAC token, and worker-class derivation live in exactly
    one place.
    """

    # The job this instance is provisioned for. Only one job per remote
    # instance in v1 — we don't reuse pods between jobs (keeps billing
    # and lifecycle simple, avoids worker contamination across runs).
    job_id: str
    # Routed to WORKER_CLASS env inside the remote container so it only
    # claims jobs it can run.
    worker_class: str
    # Desired hardware / region / image. Provider adapters map this into
    # their own shape (RunPod gpuTypeId, Vast offer query, EC2 InstanceType).
    instance_spec: InstanceSpec
    # URL the remote worker uses to reach the studio's broker endpoints.
    # Must be reachable from whichever provider network the pod sits in
    # (public HTTPS or a tunnel).
    broker_url: str
    # Short-TTL HMAC token minted by `cloud.tokens.mint` — scoped to this
    # job id + this execution target. The remote worker reads it from its
    # STUDIO_JOB_TOKEN env.
    job_token: str
    # Docker image to boot. Dispatcher defaults to the provider adapter's
    # standard remote-worker image if the spec doesn't override.
    image: str
    # Env merged on top of the standard {STUDIO_BROKER_URL, STUDIO_JOB_TOKEN,
    # WORKER_MODE=remote, WORKER_CLASS=<worker_class>} set. Used by tests
    # (the fake provider reads FAKE_SCRATCH_DIR) and by per-provider
    # advanced knobs.
    env: dict[str, str] = field(default_factory=dict)


class CloudProvider(abc.ABC):
    """Abstract provider adapter.

    Every method must be safe to call concurrently for different jobs on
    the same provider — the dispatcher uses an asyncio task per job and
    doesn't serialize provider calls.
    """

    #: Stable id matching one of the `ExecutionTarget` literals. The
    #: registry keys providers by this value.
    id: str = ""

    #: Human-readable name for logs + UI. Doesn't appear in any stored
    #: state; pure convenience.
    display_name: str = ""

    @abc.abstractmethod
    async def launch(self, spec: LaunchSpec) -> InstanceHandle:
        """Provision and boot an instance for this job. Returns as soon as
        the instance is scheduled with a stable id; boot can still be in
        flight. The caller watches `status()` or the broker's claim feed
        for readiness."""

    @abc.abstractmethod
    async def status(self, handle: InstanceHandle) -> InstanceStatus:
        """Current lifecycle state of the instance."""

    @abc.abstractmethod
    async def terminate(self, handle: InstanceHandle) -> None:
        """Stop + release the instance. Must be idempotent: calling after
        a natural termination (spot preemption, provider-side kill) must
        succeed quietly."""

    @abc.abstractmethod
    async def logs(self, handle: InstanceHandle, tail: int = 200) -> str:
        """Fetch the last `tail` lines of the container's stdout/stderr.

        Best-effort — if the provider doesn't expose logs post-termination
        return an empty string rather than raising. The UI uses this for
        the "something went wrong before the broker even saw a claim"
        diagnostic path.
        """

    @abc.abstractmethod
    async def estimate_cost(
        self, spec: InstanceSpec, expected_duration_s: float
    ) -> int:
        """Return the predicted cost in integer cents for running `spec`
        for `expected_duration_s`. Side-effect-free — safe to call on a
        UI debounce. Providers without a public pricing API return a
        coarse constant (annotated in the adapter module)."""

    async def preflight(self, spec: InstanceSpec) -> None:
        """Optional: check that our credentials + permissions are sufficient
        to actually launch `spec`. Raise with a human-readable message on
        failure. Default no-op — adapters with fiddly IAM (AWS, GCP) override.
        """
        return None


__all__ = [
    "CloudProvider",
    "InstanceHandle",
    "InstanceStatus",
    "LaunchSpec",
]
