"""Dispatcher — picks a provider, mints a token, launches an instance.

This is the one place that turns "the user submitted a job with
`execution_target=runpod`" into "a pod is booting and will claim this
job when it's ready". Kept deliberately thin: no per-provider logic
leaks in here; everything provider-specific lives behind
`CloudProvider`.

Responsibilities:

1. Translate the config's `execution_target` into a registered
   `CloudProvider`, or refuse clearly if nothing is registered for it.
2. Gate the launch on cost caps (per-job and studio-wide).
3. Mint a per-job HMAC token, record its hash + the provider instance
   id onto the job row so the orphan sweeper + the UI's billing panel
   can look both up later.
4. Hand the `LaunchSpec` to the provider and return the `InstanceHandle`.

Not the dispatcher's job:

- Running the claim loop — the provider's pod does that.
- Streaming events — the broker + `HttpJobSource` own that.
- Terminating the pod on success — the pod exits itself after one
  `runner.run_job` call; a sweeper watches for pods that fail to exit
  (cost runaway / hung SLAM / spot preemption).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

from app.cloud import tokens
from app.cloud.providers import CloudProvider, InstanceHandle, get
from app.cloud.providers.base import LaunchSpec
from app.config import settings
from app.jobs import store
from app.jobs.schema import (
    AnyJobConfig,
    ExecutionTarget,
    InstanceSpec,
    processor_kind,
)

log = logging.getLogger(__name__)


# Default image id slotted into the launch spec when the job config
# doesn't override `instance_spec.image`. Providers can override this
# per-adapter (the RunPod adapter, for example, has its own pre-baked
# image). The studio publishes this image for the generic path.
DEFAULT_REMOTE_IMAGE = "lingbot-map-studio/worker-remote:latest"


class DispatchError(RuntimeError):
    """Raised when the dispatcher refuses to launch.

    Kept as a single class so the API can translate every dispatch
    failure into a 400 with the message attached. Granular distinctions
    (cost exceeded vs missing provider vs preflight failed) show up in
    the message text, not the type hierarchy.
    """


@dataclass(frozen=True)
class DispatchResult:
    """What `launch()` returns. The instance handle is authoritative;
    the token + token_hash are returned so the API can emit an audit
    event without re-deriving them.
    """

    provider_id: str
    instance_handle: InstanceHandle
    job_token: str
    token_hash: str
    cost_estimate_cents: int


def _resolve_worker_class(config: AnyJobConfig) -> str:
    """Map the processor family onto one of the three worker classes.

    Lingbot goes to `lingbot`, every SLAM backend to `slam`, gsplat to
    `gs`. The remote pod reads `WORKER_CLASS` from its env and passes
    it to `claim_next`, so a mis-mapped job would pick up the wrong
    type of work — worth keeping the mapping explicit and close to the
    dispatcher.
    """
    kind = processor_kind(config)
    if kind == "reconstruction":
        return "lingbot"
    if kind == "slam":
        return "slam"
    if kind == "gsplat":
        return "gs"
    raise DispatchError(f"no worker_class mapping for processor kind {kind!r}")


def _effective_instance_spec(config: AnyJobConfig) -> InstanceSpec:
    """Config might not carry an `instance_spec` (legacy rows, lazy
    users); fall back to the mixin default so the provider always gets
    a concrete spec. Pydantic's default factory handles this already
    for new configs, but a migration row might carry `None`.
    """
    spec = getattr(config, "instance_spec", None)
    if spec is None:
        return InstanceSpec()
    return spec


def _hash_token(token: str) -> str:
    """Blake2b-64 of the raw token. Short enough to live in a varchar
    column; strong enough that a collision is a cryptographic event.
    """
    return hashlib.blake2b(token.encode("utf-8"), digest_size=32).hexdigest()


def _enforce_cost_caps(
    config: AnyJobConfig, estimate_cents: int
) -> None:
    """Abort the dispatch if the provider's estimate exceeds either cap.

    Two caps apply:

    - Per-job `cost_cap_cents` (from `ExecutionFields` on the config).
      Defaults to None; None means "use the studio-wide default".
    - Studio-wide `settings.cloud_cost_cap_cents_default`. Hard upper
      bound regardless of what the user asked for — a user with a
      misconfigured per-job cap shouldn't be able to exceed this.
    """
    studio_cap = settings.cloud_cost_cap_cents_default
    job_cap = getattr(config, "cost_cap_cents", None) or studio_cap
    effective_cap = min(job_cap, studio_cap)
    if estimate_cents > effective_cap:
        raise DispatchError(
            f"estimated cost {estimate_cents}¢ exceeds cap {effective_cap}¢ "
            f"(job cap: {job_cap}¢, studio default: {studio_cap}¢). "
            "raise cost_cap_cents or pick a cheaper instance_spec."
        )


async def launch(
    job_id: str,
    config: AnyJobConfig,
    *,
    expected_duration_s: float = 15 * 60,
    provider_override: Optional[CloudProvider] = None,
) -> DispatchResult:
    """Provision a rented pod for a submitted job.

    `provider_override` is a test seam — production always resolves
    the provider from the registry by `config.execution_target`.
    """
    target: ExecutionTarget = getattr(config, "execution_target", "local")
    if target == "local":
        raise DispatchError(
            "dispatch.launch called for a local job — local jobs run in the "
            "in-process worker, never through the dispatcher."
        )

    provider = provider_override if provider_override is not None else get(target)
    spec = _effective_instance_spec(config)

    # Preflight is best-effort: adapters with strict creds (AWS, GCP)
    # raise here so the user sees the problem at submit time, not after
    # the UI has rendered a "job queued" state for 30 seconds.
    await provider.preflight(spec)

    cost_estimate = await provider.estimate_cost(spec, expected_duration_s)
    _enforce_cost_caps(config, cost_estimate)

    # Mint the token with the full scope set — the pod runs every
    # broker endpoint, so a reduced scope would just force us to
    # widen it the first time we added a new endpoint.
    token = tokens.mint(
        job_id=job_id,
        execution_target=target,
        scopes=list(tokens.SCOPES),
        ttl_s=settings.cloud_broker_token_ttl_s,
        key=settings.cloud_broker_hmac_key,
    )
    token_hash = _hash_token(token)

    image = spec.image or DEFAULT_REMOTE_IMAGE
    worker_class = _resolve_worker_class(config)

    launch_spec = LaunchSpec(
        job_id=job_id,
        worker_class=worker_class,
        instance_spec=spec,
        broker_url=settings.cloud_broker_public_url,
        job_token=token,
        image=image,
        env={**spec.env},  # provider adds STUDIO_BROKER_URL / STUDIO_JOB_TOKEN / WORKER_MODE / WORKER_CLASS
    )

    try:
        handle = await provider.launch(launch_spec)
    except Exception as exc:
        log.exception("provider %s failed to launch job %s", provider.id, job_id)
        raise DispatchError(
            f"provider {provider.id} failed to launch: {type(exc).__name__}: {exc}"
        ) from exc

    # Persist the bookkeeping fields on the job row so the sweeper +
    # billing UI can find the instance later. Token itself never
    # touches the DB; only its hash.
    await store.set_provider_bookkeeping(
        job_id,
        provider_instance_id=handle.instance_id,
        cost_estimate_cents=cost_estimate,
        token_hash=token_hash,
    )

    log.info(
        "dispatched %s → %s instance=%s region=%s gpu=%s est=%d¢",
        job_id,
        provider.id,
        handle.instance_id,
        handle.region,
        handle.gpu_class,
        cost_estimate,
    )
    return DispatchResult(
        provider_id=provider.id,
        instance_handle=handle,
        job_token=token,
        token_hash=token_hash,
        cost_estimate_cents=cost_estimate,
    )


async def terminate(
    provider_id: str, instance_id: str, *, metadata: Optional[dict[str, str]] = None
) -> None:
    """Tear down a provider instance by id. Idempotent — used by the
    orphan sweeper and by the API's cancel-with-teardown path.

    Rebuilds an `InstanceHandle` from the stored id + metadata because
    the dispatcher doesn't persist the full handle (the fields it needs
    are provider-specific and we didn't want to add per-provider
    columns). Passing `metadata=None` works for providers that only
    need the id (RunPod, the fake provider); AWS + GCP that need region
    look it up from `metadata`.
    """
    provider = get(provider_id)
    handle = InstanceHandle(
        provider_id=provider_id,
        instance_id=instance_id,
        metadata=dict(metadata or {}),
    )
    await provider.terminate(handle)


__all__ = [
    "DEFAULT_REMOTE_IMAGE",
    "DispatchError",
    "DispatchResult",
    "launch",
    "terminate",
]
