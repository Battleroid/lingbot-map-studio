"""End-to-end pin for the cloud dispatch path via the `fake` provider.

This is the highest-leverage test in the cloud stack: it exercises the
whole round trip — dispatcher → provider → pod → `HttpJobSource` →
broker → `runner.run_job` → artifacts back to the studio — entirely
in-process, so CI can run it without docker or cloud credentials.

The `fake` provider runs the "pod" as an asyncio task in the same
event loop, using an httpx `AsyncClient` backed by `ASGITransport`
against the in-process FastAPI app. That means every byte that would
flow over HTTPS in a real RunPod run actually flows through this
transport, including chunked artifact PUTs and cancel long-polls.

Run: `pytest worker/tests/test_dispatcher_fake_e2e.py -q`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient


async def _seed_queued(job_id: str, *, execution_target: str = "fake") -> None:
    """Seed a queued lingbot job with a cloud execution target.

    The dispatcher reads `execution_target` off the config, so the
    round-trip only kicks in when the config carries a non-local value.
    """
    from app.config import settings
    from app.jobs import store
    from app.jobs.schema import Job, LingbotConfig

    await store.init_store()
    settings.job_uploads(job_id).mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    await store.create_job(
        Job(
            id=job_id,
            status="queued",
            config=LingbotConfig(
                model_id="lingbot-map",
                fps=5.0,
                execution_target=execution_target,
            ),
            uploads=[],
            artifacts=[],
            created_at=now,
            updated_at=now,
        ),
        worker_class="lingbot",
    )


class _StubProcessor:
    """Same shape as the stub in `test_runner_through_source` — no torch,
    no ffmpeg, just writes a single artifact and publishes one event.

    The point of the E2E test is the *transport*, not the processor, so
    keeping this trivial is correct. A heavier processor would drown out
    whether the broker round-tripped the state changes properly.
    """

    id = "lingbot"
    kind = "reconstruction"
    worker_class = "lingbot"
    supported_artifacts = frozenset({"glb"})

    async def run(self, ctx):
        from app.jobs.schema import Artifact, JobEvent

        await ctx.set_frames_total(7)
        await ctx.publish(
            JobEvent(job_id=ctx.job_id, stage="inference", message="stub via fake")
        )
        (ctx.artifacts_dir / "stub.glb").write_bytes(b"REMOTE-BYTES")
        return _StubResult(artifacts=[Artifact(name="stub.glb", kind="glb")])


class _StubResult:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.extras: dict = {}


@pytest.fixture
def asgi_app(tmp_data_dir: Path):
    """Bring up the in-process FastAPI app so the fake pod can hit its
    broker endpoints over httpx `ASGITransport`."""
    from app.main import app

    with TestClient(app):
        yield app


@pytest.fixture
def patch_resolve():
    """Swap `resolve()` to return the stub so we don't need torch or the
    real lingbot processor module to be importable in CI."""
    import app.jobs.runner as runner_mod

    original = runner_mod.resolve
    runner_mod.resolve = lambda _cfg: _StubProcessor()
    try:
        yield
    finally:
        runner_mod.resolve = original


@pytest.fixture
def fake_provider_with_asgi(asgi_app):
    """Wire the fake provider's client factory to build httpx clients
    that speak to the in-process app via ASGITransport. Clears the hook
    on teardown so a later test that builds its own transport isn't
    affected."""
    from app.cloud.providers import get
    from app.cloud.providers.fake import FakeProvider

    def _factory(base_url: str, token: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=asgi_app),
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )

    FakeProvider.client_factory = _factory
    provider = get("fake")
    # Ensure the registered instance is clean between tests (no leftover
    # handles from a previous parametrisation).
    provider.clear()
    try:
        yield provider
    finally:
        FakeProvider.client_factory = None
        provider.clear()


@pytest.fixture
def patch_broker_url(monkeypatch: pytest.MonkeyPatch):
    """Point the dispatcher at a URL the ASGI transport recognises.

    `httpx.ASGITransport` dispatches based on hostname, so the broker URL
    only needs to *parse* as a URL — the actual bytes route through the
    transport regardless.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "cloud_broker_public_url", "http://studio.test")


async def test_dispatcher_fake_e2e_round_trip(
    asgi_app,
    patch_resolve,
    fake_provider_with_asgi,
    patch_broker_url,
    tmp_data_dir: Path,
):
    """The golden path: queue a job with `execution_target=fake`, dispatch
    it, and confirm the studio-side DB + filesystem look like a local run
    once the fake pod finishes."""
    from app.cloud import dispatcher as cloud_dispatcher
    from app.config import settings
    from app.jobs import store
    from app.jobs.schema import LingbotConfig

    job_id = "fakee2e01"
    await _seed_queued(job_id)

    cfg = LingbotConfig(
        model_id="lingbot-map", fps=5.0, execution_target="fake"
    )
    result = await cloud_dispatcher.launch(job_id, cfg)

    # The dispatcher should have written bookkeeping columns before
    # returning. `provider_instance_id` is the key one — the sweeper +
    # cost watchdog key off it.
    async with store.session() as s:
        row = await s.get(store.JobRow, job_id)
        assert row is not None
        assert row.provider_instance_id == result.instance_handle.instance_id
        assert row.remote_worker_token_hash is not None
        assert row.execution_target == "fake"

    # Wait for the fake pod task to finish claiming + running + finalising.
    await fake_provider_with_asgi.wait_for(
        result.instance_handle.instance_id, timeout=30.0
    )

    # Studio DB should reflect the completed run exactly as if a local
    # worker had done it.
    final = await store.get_job(job_id)
    assert final is not None
    assert final.status == "ready"
    assert final.frames_total == 7
    assert [a.name for a in final.artifacts] == ["stub.glb"]

    # The bytes that the fake pod wrote into its own scratch dir must
    # have been synced up to the studio's shared artifacts dir.
    studio_artifact = settings.job_artifacts(job_id) / "stub.glb"
    assert studio_artifact.read_bytes() == b"REMOTE-BYTES"


async def test_dispatcher_refuses_local_target(tmp_data_dir: Path):
    """The dispatcher is remote-only — calling it for a `local` job is
    a programming error, not a runtime failure, so it should raise
    loudly rather than silently no-op.
    """
    from app.cloud import dispatcher as cloud_dispatcher
    from app.jobs.schema import LingbotConfig

    cfg = LingbotConfig(model_id="lingbot-map", execution_target="local")
    with pytest.raises(cloud_dispatcher.DispatchError):
        await cloud_dispatcher.launch("dispatchlocal", cfg)


async def test_dispatcher_unknown_target_is_clear(tmp_data_dir: Path):
    """An `execution_target` with no registered provider should error
    with the target name in the message, not a cryptic KeyError.
    """
    from app.cloud import dispatcher as cloud_dispatcher
    from app.jobs.schema import LingbotConfig

    # Bypass Literal validation: build via dict so the dispatcher sees
    # an unknown target from a misconfigured deploy rather than a
    # well-formed Pydantic error.
    cfg = LingbotConfig(model_id="lingbot-map", execution_target="fake")
    object.__setattr__(cfg, "execution_target", "nonexistent-provider")
    with pytest.raises(KeyError) as excinfo:
        await cloud_dispatcher.launch("dispatchunk", cfg)
    assert "nonexistent-provider" in str(excinfo.value)


async def test_dispatcher_cost_cap_blocks_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_data_dir: Path
):
    """The cost guard must refuse a launch whose estimate exceeds either
    the per-job cap or the studio-wide default.

    We seed the fake provider with a non-zero estimate by monkey-patching
    its `estimate_cost` for this one test.
    """
    from app.cloud import dispatcher as cloud_dispatcher
    from app.cloud.providers import get
    from app.jobs.schema import LingbotConfig

    provider = get("fake")

    async def _expensive(_spec, _duration):
        return 99_999  # $999.99 — blows past any sensible cap

    monkeypatch.setattr(provider, "estimate_cost", _expensive)

    cfg = LingbotConfig(
        model_id="lingbot-map",
        execution_target="fake",
        cost_cap_cents=500,  # $5 cap
    )
    with pytest.raises(cloud_dispatcher.DispatchError) as excinfo:
        await cloud_dispatcher.launch("costcap01", cfg)
    assert "exceeds cap" in str(excinfo.value)
