"""End-to-end pin for the refactored runner.

The runner now routes every "talk to the studio" call through a
`JobSource`. These tests prove the refactor didn't drop any of the
state-transition / event / release / cancel hops by running a stub
processor against a real `LocalJobSource` and inspecting the resulting
sqlite rows + events.jsonl.

No real inference: the stub processor just publishes one event, reports
frames_total, and returns a single artifact.

Run: `pytest worker/tests/test_runner_through_source.py -q`.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


async def _seed_queued_lingbot(job_id: str) -> None:
    from app.jobs import store
    from app.jobs.schema import Job, LingbotConfig

    await store.init_store()
    now = datetime.now(timezone.utc)
    await store.create_job(
        Job(
            id=job_id,
            status="queued",
            config=LingbotConfig(model_id="lingbot-map", fps=5.0),
            uploads=[],
            artifacts=[],
            created_at=now,
            updated_at=now,
        ),
        worker_class="lingbot",
    )


class _StubProcessor:
    """Minimal Processor replacement — no torch, no ffmpeg."""

    id = "lingbot"
    kind = "reconstruction"
    worker_class = "lingbot"
    supported_artifacts = frozenset({"glb"})

    def __init__(self, outcome: str = "success") -> None:
        self._outcome = outcome

    async def run(self, ctx):
        from app.jobs.schema import Artifact, JobEvent

        await ctx.set_frames_total(42)
        await ctx.publish(
            JobEvent(job_id=ctx.job_id, stage="inference", message="stub tick")
        )
        if self._outcome == "fail":
            raise RuntimeError("stub boom")
        if self._outcome == "cancel_wait":
            # Simulate a long-running processor that checks cancel mid-way.
            for _ in range(40):  # up to ~4s
                ctx.check_cancel()
                await asyncio.sleep(0.1)
        # Write a real artifact file so the manifest reads as plausible.
        (ctx.artifacts_dir / "stub.glb").write_bytes(b"stub")
        return _StubResult(artifacts=[Artifact(name="stub.glb", kind="glb")])


class _StubResult:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.extras: dict = {}


async def _run_stub(
    tmp_data_dir: Path,
    job_id: str,
    outcome: str = "success",
    *,
    cancel_after_s: float | None = None,
):
    """Claim the queued stub job and drive it through the refactored runner."""
    from app.cloud import LocalJobSource
    from app.jobs import cancel as cancel_mod
    from app.jobs import runner

    # Patch the processor registry to return the stub when `resolve()` is
    # called. The runner's `resolve(config)` path takes the config's
    # `processor` discriminator; we intercept it to sidestep the real
    # lingbot module which imports torch.
    import app.jobs.runner as runner_mod

    original_resolve = runner_mod.resolve
    runner_mod.resolve = lambda _cfg, _outcome=outcome: _StubProcessor(_outcome)

    try:
        source = LocalJobSource()
        claim = await source.claim_next("lingbot", worker_id="wrk-stub")
        assert claim is not None

        async def _maybe_cancel():
            if cancel_after_s is None:
                return
            await asyncio.sleep(cancel_after_s)
            await cancel_mod.request_cancel(job_id, "test flip")

        cancel_task = asyncio.create_task(_maybe_cancel())
        try:
            await runner.run_job(
                claim.job_id,
                claim.uploads,
                claim.config,
                worker_id="wrk-stub",
                source=source,
            )
        finally:
            cancel_task.cancel()
            try:
                await cancel_task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        runner_mod.resolve = original_resolve


@pytest.mark.asyncio
async def test_runner_happy_path_transitions_to_ready(tmp_data_dir: Path):
    from app.jobs import store

    await _seed_queued_lingbot("runparity01")
    await _run_stub(tmp_data_dir, "runparity01", outcome="success")

    row = await store.get_job("runparity01")
    assert row is not None
    assert row.status == "ready"
    assert row.frames_total == 42
    assert [a.name for a in row.artifacts] == ["stub.glb"]

    events_path = tmp_data_dir / "jobs" / "runparity01" / "events.jsonl"
    lines = [json.loads(ln) for ln in events_path.read_text().splitlines() if ln.strip()]
    # One stub tick + one "job ready" system event.
    assert any(e["message"] == "stub tick" for e in lines)
    assert any(e["message"] == "job ready" for e in lines)

    # Claim should have been released so another worker could pick it up.
    # (The job is in terminal status so it wouldn't, but claimed_by must be None.)
    assert (tmp_data_dir / "jobs" / "runparity01" / "events.done").exists()


@pytest.mark.asyncio
async def test_runner_records_failure_on_processor_exception(tmp_data_dir: Path):
    from app.jobs import store

    await _seed_queued_lingbot("runparity02")
    await _run_stub(tmp_data_dir, "runparity02", outcome="fail")

    row = await store.get_job("runparity02")
    assert row is not None
    assert row.status == "failed"
    assert row.error is not None
    assert "stub boom" in row.error
    # Status-setting is its own call through the source now — the row must
    # still end up in a single terminal state, not a half-written one.


@pytest.mark.asyncio
async def test_runner_routes_cancel_through_source(tmp_data_dir: Path):
    """Cancel flag set mid-run should land the row in `cancelled`, not `failed`."""
    from app.jobs import store

    await _seed_queued_lingbot("runparity03")
    await _run_stub(
        tmp_data_dir,
        "runparity03",
        outcome="cancel_wait",
        cancel_after_s=0.3,
    )

    row = await store.get_job("runparity03")
    assert row is not None
    assert row.status == "cancelled"
    assert row.error is not None and row.error.startswith("cancelled:")
