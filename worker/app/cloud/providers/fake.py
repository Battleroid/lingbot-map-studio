"""In-process fake provider for CI + developer smoke runs.

`FakeProvider` doesn't touch any cloud API. When the dispatcher calls
`launch()`, it spawns an asyncio task in the same event loop that plays
the role of a rented pod: build an `HttpJobSource` pointed at the
broker URL the dispatcher handed over, run one claim loop iteration,
execute the job via the shared `runner.run_job`, and exit.

That means `execution_target="fake"` exercises the *entire* remote
codepath — `HttpJobSource`, the broker's HMAC auth, chunked artifact
uploads, the cancel long-poll, the terminal finalise — without needing
docker-in-docker or a real tunnel. Exactly the same `Processor.run(ctx)`
runs as in a local claim; the only thing that changes is the transport.

Tests inject an `httpx.AsyncClient` backed by `ASGITransport` via the
`client_factory` class attribute so the "remote" worker and the "studio"
FastAPI app share a process without needing uvicorn on a real port.
Production never sets `client_factory` and a plain TCP client is built.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx

from app.cloud.providers.base import (
    CloudProvider,
    InstanceHandle,
    InstanceStatus,
    LaunchSpec,
)
from app.jobs.schema import InstanceSpec

log = logging.getLogger(__name__)


# Where each "pod" stages its scratch dir. Parallel fake instances get
# separate subdirs keyed by instance id so partial snapshots from one
# job never land on another's path.
_FAKE_SCRATCH_ROOT = Path(os.environ.get("FAKE_PROVIDER_SCRATCH", "/tmp/fake-provider-scratch"))

# Coarse constant used by `estimate_cost`. The fake provider is free —
# this matches the "no spend" assumption of CI and the UI's badge still
# renders `$0.00`.
_FAKE_COST_CENTS_PER_HOUR = 0


# Test-visible hook. Set by conftest to a factory that returns an httpx
# `AsyncClient` bound to an `ASGITransport` against the in-process
# FastAPI app. When None (production), a plain `httpx.AsyncClient` over
# real TCP is constructed instead.
ClientFactory = Callable[[str, str], httpx.AsyncClient]


class FakeProvider(CloudProvider):
    id = "fake"
    display_name = "Fake (in-process)"

    # Class-level so tests can assign once per module and every FakeProvider
    # instance picks it up.  Pattern: conftest sets it on fixture setup,
    # clears on teardown.
    client_factory: Optional[ClientFactory] = None

    # Hook for tests that want to watch one specific claim complete.
    # The dispatcher's launch path returns before work finishes, so CI
    # tests await `FakeProvider.wait_for(instance_id)` to block on the
    # worker task finishing before asserting on artifacts.
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._handles: dict[str, InstanceHandle] = {}
        self._statuses: dict[str, InstanceStatus] = {}
        self._logs: dict[str, list[str]] = {}

    # --- CloudProvider --------------------------------------------------

    async def launch(self, spec: LaunchSpec) -> InstanceHandle:
        instance_id = f"fake-{spec.job_id}-{int(time.time() * 1000)}"
        handle = InstanceHandle(
            provider_id=self.id,
            instance_id=instance_id,
            region="in-process",
            gpu_class=spec.instance_spec.gpu_class,
            launched_at=time.time(),
            metadata={
                "job_id": spec.job_id,
                "worker_class": spec.worker_class,
            },
        )
        self._handles[instance_id] = handle
        self._statuses[instance_id] = "pending"
        self._logs[instance_id] = []

        task = asyncio.create_task(
            self._run_pod(spec, instance_id),
            name=f"fake-pod:{instance_id}",
        )
        self._tasks[instance_id] = task

        # Register a done-callback so a crashed pod flips to "error"
        # without the caller having to await the task. The sweeper /
        # `status()` call will observe it.
        def _on_done(t: asyncio.Task[None], _instance_id: str = instance_id) -> None:
            if t.cancelled():
                self._statuses[_instance_id] = "terminated"
                return
            exc = t.exception()
            if exc is not None:
                self._statuses[_instance_id] = "error"
                self._logs.setdefault(_instance_id, []).append(
                    f"pod crashed: {type(exc).__name__}: {exc}"
                )
            else:
                self._statuses[_instance_id] = "terminated"

        task.add_done_callback(_on_done)
        return handle

    async def status(self, handle: InstanceHandle) -> InstanceStatus:
        return self._statuses.get(handle.instance_id, "terminated")

    async def terminate(self, handle: InstanceHandle) -> None:
        task = self._tasks.get(handle.instance_id)
        if task is None or task.done():
            return
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

    async def logs(self, handle: InstanceHandle, tail: int = 200) -> str:
        lines = self._logs.get(handle.instance_id, [])
        return "\n".join(lines[-tail:])

    async def estimate_cost(
        self, spec: InstanceSpec, expected_duration_s: float
    ) -> int:
        return _FAKE_COST_CENTS_PER_HOUR * int(expected_duration_s) // 3600

    # --- Test helpers ---------------------------------------------------

    async def wait_for(self, instance_id: str, timeout: float = 60.0) -> None:
        """Block until the sibling worker task finishes (success or fail).

        Tests use this to synchronise on "the job has been claimed, run,
        and finalised" before asserting on studio-side state. Production
        never calls it — the dispatcher's own watcher does the same work
        via the provider's `status()` + job row polling.
        """
        task = self._tasks.get(instance_id)
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"fake instance {instance_id} did not finish within {timeout}s"
            ) from exc
        except Exception:
            # The done-callback already recorded the error; tests read it
            # via `logs()` or the studio-side error column.
            pass

    def clear(self) -> None:
        """Reset per-process state. Test fixtures call this between runs so
        the handle/task dicts don't leak between independent tests."""
        self._tasks.clear()
        self._handles.clear()
        self._statuses.clear()
        self._logs.clear()

    # --- Internals ------------------------------------------------------

    async def _run_pod(self, spec: LaunchSpec, instance_id: str) -> None:
        """Embodies the 'rented pod' lifetime: build the remote source,
        claim the one job the dispatcher minted this token for, run it,
        exit.

        The sequence mirrors `worker_main._run_forever` but scoped to a
        single job (no `while True`): the dispatcher provisions one pod
        per job, and the HMAC token is scoped to that job id anyway.
        """
        from app.cloud.http_source import HttpJobSource
        from app.jobs import runner, store

        self._statuses[instance_id] = "running"
        pod_scratch = _FAKE_SCRATCH_ROOT / instance_id
        pod_scratch.mkdir(parents=True, exist_ok=True)

        # Build the "remote" HTTP client. Tests inject an ASGI transport;
        # prod uses a plain TCP client.
        if FakeProvider.client_factory is not None:
            client = FakeProvider.client_factory(spec.broker_url, spec.job_token)
        else:
            client = httpx.AsyncClient(
                base_url=spec.broker_url,
                headers={"Authorization": f"Bearer {spec.job_token}"},
                timeout=60.0,
            )

        source = HttpJobSource(
            base_url=spec.broker_url,
            token=spec.job_token,
            scratch_root=pod_scratch,
            client=client,
        )

        worker_id = f"fake:{instance_id}"
        self._logs[instance_id].append(
            f"pod {instance_id} online — claiming {spec.job_id} as {worker_id}"
        )

        try:
            claim = await source.claim_next(spec.worker_class, worker_id=worker_id)
            if claim is None:
                # The dispatcher only launches us *because* there's a
                # queued job for this token; claiming `None` means the
                # job got stolen or cancelled between dispatch and claim.
                self._logs[instance_id].append(
                    f"nothing to claim for worker_class={spec.worker_class} — exiting"
                )
                return
            self._logs[instance_id].append(
                f"claimed {claim.job_id} processor={claim.config.processor}"
            )
            await runner.run_job(
                claim.job_id,
                claim.uploads,
                claim.config,
                worker_id=worker_id,
                source=source,
            )
            self._logs[instance_id].append(f"job {claim.job_id} finalised")
        finally:
            try:
                await source.aclose()
            except Exception:  # noqa: BLE001
                pass
            # Best-effort cleanup of the pod's scratch dir. On a real pod
            # the whole disk goes away; here we clean up to keep /tmp tidy.
            try:
                shutil.rmtree(pod_scratch, ignore_errors=True)
            except Exception:  # noqa: BLE001
                pass


__all__ = ["FakeProvider"]
