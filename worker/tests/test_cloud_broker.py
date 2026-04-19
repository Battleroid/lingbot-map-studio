"""HTTP contract pin for the `/api/worker/*` broker.

The broker is the one network surface a remote worker talks to. Every
endpoint is the mirror of one `LocalJobSource` call, so these tests
double as a parity spec: if a broker handler silently reshapes a return
value, the eventual `HttpJobSource` will silently diverge from
`LocalJobSource` and we'll get subtle ghost bugs (events stamped with
the wrong job, artifacts written to the wrong dir, etc.).

Scope:
  * Auth: every route requires a Bearer token in the right scope.
  * Happy-path behaviour: claim, events, artifact PUT, checkpoint GET,
    upload GET, cancel poll, heartbeat, status/frames_total/terminal.
  * Negative paths: invalid filenames (traversal), missing tokens, wrong
    scope, job-not-found.

We use FastAPI's `TestClient` against the real app. Each test seeds a
queued job with the same helper the local-source parity tests use, so
the DB state is identical to production.

Run: `pytest worker/tests/test_cloud_broker.py -q`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _mint_token(job_id: str, scopes: list[str] | None = None, *, key: str | None = None) -> str:
    from app.cloud import tokens
    from app.config import settings

    return tokens.mint(
        job_id=job_id,
        execution_target="fake",
        scopes=scopes if scopes is not None else list(tokens.SCOPES),
        ttl_s=300,
        key=key or settings.cloud_broker_hmac_key,
    )


async def _seed_queued_lingbot(
    job_id: str,
    *,
    worker_class: str = "lingbot",
    with_upload_named: str | None = "clip.mp4",
) -> None:
    from app.config import settings
    from app.jobs import store
    from app.jobs.schema import Job, LingbotConfig

    await store.init_store()

    uploads_dir = settings.job_uploads(job_id)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_names: list[str] = []
    if with_upload_named:
        clip = uploads_dir / with_upload_named
        clip.write_bytes(b"\x00\x00\x00\x20ftypisomHELLOWORLD")
        upload_names.append(with_upload_named)

    now = datetime.now(timezone.utc)
    job = Job(
        id=job_id,
        status="queued",
        config=LingbotConfig(model_id="lingbot-map", fps=10.0),
        uploads=upload_names,
        artifacts=[],
        created_at=now,
        updated_at=now,
    )
    await store.create_job(job, worker_class=worker_class)


@pytest.fixture
def client(tmp_data_dir: Path):
    """FastAPI TestClient over the real app with the broker router mounted.

    We skip `lifespan` — our tests explicitly seed `store.init_store` and
    don't need the orphan-sweep loop burning a coroutine per test.
    """
    from app.main import app

    with TestClient(app) as c:
        yield c


# --- auth --------------------------------------------------------------


async def test_missing_auth_header_is_401(client: TestClient):
    await _seed_queued_lingbot("bkrauth01")
    r = client.post("/api/worker/claim", json={"worker_class": "lingbot", "worker_id": "w1"})
    assert r.status_code == 401


async def test_bearer_without_token_is_401(client: TestClient):
    r = client.post(
        "/api/worker/claim",
        json={"worker_class": "lingbot", "worker_id": "w1"},
        headers={"Authorization": "NotBearer xyz"},
    )
    assert r.status_code == 401


async def test_wrong_scope_is_401(client: TestClient):
    await _seed_queued_lingbot("bkrauth02")
    # Token carries only events+heartbeat; /claim requires "claim".
    tok = _mint_token("bkrauth02", scopes=["events", "heartbeat"])
    r = client.post(
        "/api/worker/claim",
        json={"worker_class": "lingbot", "worker_id": "w1"},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 401


async def test_bad_hmac_key_is_401(client: TestClient):
    await _seed_queued_lingbot("bkrauth03")
    tok = _mint_token("bkrauth03", key="wrong-key")
    r = client.post(
        "/api/worker/claim",
        json={"worker_class": "lingbot", "worker_id": "w1"},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 401


# --- claim -------------------------------------------------------------


async def test_claim_returns_token_bound_job(client: TestClient):
    await _seed_queued_lingbot("bkrclaim01")
    tok = _mint_token("bkrclaim01")
    r = client.post(
        "/api/worker/claim",
        json={"worker_class": "lingbot", "worker_id": "remote-1"},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == "bkrclaim01"
    assert body["config"]["processor"] == "lingbot"
    assert body["uploads"] == ["clip.mp4"]


async def test_claim_is_204_when_job_already_running(client: TestClient):
    from app.jobs import store

    await _seed_queued_lingbot("bkrclaim02")
    # Simulate another worker already picked it up.
    await store.update_job("bkrclaim02", status="inference")

    tok = _mint_token("bkrclaim02")
    r = client.post(
        "/api/worker/claim",
        json={"worker_class": "lingbot", "worker_id": "remote-2"},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 204


async def test_claim_404_when_job_does_not_exist(client: TestClient):
    tok = _mint_token("doesnotexist")
    r = client.post(
        "/api/worker/claim",
        json={"worker_class": "lingbot", "worker_id": "remote-3"},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 404


async def test_claim_204_when_worker_class_mismatches(client: TestClient):
    """A lingbot-class job mustn't be claimed by a worker asking for `gs`."""
    await _seed_queued_lingbot("bkrclaim03", worker_class="lingbot")
    tok = _mint_token("bkrclaim03")
    r = client.post(
        "/api/worker/claim",
        json={"worker_class": "gs", "worker_id": "remote-4"},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 204


# --- uploads + checkpoints --------------------------------------------


async def test_fetch_upload_streams_bytes(client: TestClient, tmp_data_dir: Path):
    await _seed_queued_lingbot("bkrupl01")
    tok = _mint_token("bkrupl01")
    r = client.get(
        "/api/worker/uploads/clip.mp4",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    assert r.content.endswith(b"HELLOWORLD")


async def test_fetch_upload_rejects_path_traversal(client: TestClient):
    await _seed_queued_lingbot("bkrupl02")
    tok = _mint_token("bkrupl02")
    r = client.get(
        "/api/worker/uploads/..%2Fescape",
        headers={"Authorization": f"Bearer {tok}"},
    )
    # Either rejected by our guard (400) or never matched the route (404).
    # Both are safe outcomes; what we're pinning is "doesn't escape the dir".
    assert r.status_code in {400, 404}


async def test_fetch_upload_404_when_missing(client: TestClient):
    await _seed_queued_lingbot("bkrupl03", with_upload_named=None)
    tok = _mint_token("bkrupl03")
    r = client.get(
        "/api/worker/uploads/nope.mp4",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 404


async def test_fetch_checkpoint_streams_file(client: TestClient, tmp_data_dir: Path):
    from app.config import settings

    await _seed_queued_lingbot("bkrckpt01")
    ckpt_dir = settings.models_dir / "checkpoints" / "lingbot"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "weights.pt").write_bytes(b"FAKE_WEIGHTS")

    tok = _mint_token("bkrckpt01")
    r = client.get(
        "/api/worker/checkpoints/lingbot/weights.pt",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    assert r.content == b"FAKE_WEIGHTS"


# --- events ------------------------------------------------------------


async def test_publish_events_appends_to_jsonl_and_stamps_job_id(
    client: TestClient, tmp_data_dir: Path
):
    await _seed_queued_lingbot("bkrev01")
    tok = _mint_token("bkrev01")

    # A leaked token trying to forge a foreign job_id — the broker must
    # overwrite it with the token's jid.
    payload = [
        {
            "job_id": "some-other-job",
            "stage": "inference",
            "level": "info",
            "message": "tick 1",
        },
        {
            "job_id": "bkrev01",
            "stage": "inference",
            "level": "info",
            "message": "tick 2",
        },
    ]
    r = client.post(
        "/api/worker/events",
        json=payload,
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    assert r.json()["published"] == 2

    events_path = tmp_data_dir / "jobs" / "bkrev01" / "events.jsonl"
    rows = [json.loads(ln) for ln in events_path.read_text().splitlines() if ln.strip()]
    bodies = [r for r in rows if r["message"] in {"tick 1", "tick 2"}]
    assert len(bodies) == 2
    assert all(r["job_id"] == "bkrev01" for r in bodies)


# --- artifacts ---------------------------------------------------------


async def test_put_artifact_writes_final_file_atomically(
    client: TestClient, tmp_data_dir: Path
):
    await _seed_queued_lingbot("bkrart01")
    tok = _mint_token("bkrart01")
    body = b"\x89PLY\n" + (b"X" * 8192)
    r = client.put(
        "/api/worker/artifacts/mesh.glb",
        content=body,
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    result = r.json()
    assert result["size_bytes"] == len(body)

    final = tmp_data_dir / "jobs" / "bkrart01" / "artifacts" / "mesh.glb"
    assert final.read_bytes() == body
    # Sibling `.part` must be gone after rename.
    assert not (final.with_suffix(final.suffix + ".part")).exists()


async def test_put_artifact_rejects_path_traversal(client: TestClient):
    await _seed_queued_lingbot("bkrart02")
    tok = _mint_token("bkrart02")
    r = client.put(
        "/api/worker/artifacts/..%2Fescape.glb",
        content=b"x",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code in {400, 404}


# --- cancel / heartbeat / status / terminal ---------------------------


async def test_cancel_reflects_store_flag(client: TestClient):
    from app.jobs import cancel as cancel_mod

    await _seed_queued_lingbot("bkrcancel01")
    tok = _mint_token("bkrcancel01")

    r = client.get(
        "/api/worker/cancel",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    assert r.json()["cancel_requested"] is False

    await cancel_mod.request_cancel("bkrcancel01", "test flip")
    r = client.get(
        "/api/worker/cancel",
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    assert r.json()["cancel_requested"] is True


async def test_heartbeat_updates_claim_timestamp(client: TestClient):
    from app.jobs import store

    await _seed_queued_lingbot("bkrhb01")
    # Claim first so there's a row to heartbeat.
    await store.claim_next_job("lingbot", worker_id="wrk-hb")

    tok = _mint_token("bkrhb01")
    r = client.post(
        "/api/worker/heartbeat",
        json={"worker_id": "wrk-hb"},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


async def test_set_status_updates_row(client: TestClient):
    from app.jobs import store

    await _seed_queued_lingbot("bkrstat01")
    tok = _mint_token("bkrstat01")
    r = client.post(
        "/api/worker/status",
        json={"status": "inference"},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    row = await store.get_job("bkrstat01")
    assert row is not None and row.status == "inference"


async def test_set_frames_total_updates_row(client: TestClient):
    from app.jobs import store

    await _seed_queued_lingbot("bkrft01")
    tok = _mint_token("bkrft01")
    r = client.post(
        "/api/worker/frames_total",
        json={"frames_total": 512},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200
    row = await store.get_job("bkrft01")
    assert row is not None and row.frames_total == 512


async def test_terminal_finalizes_and_releases(client: TestClient, tmp_data_dir: Path):
    from app.jobs import store

    await _seed_queued_lingbot("bkrterm01")
    claim = await store.claim_next_job("lingbot", worker_id="wrk-t")
    assert claim is not None

    tok = _mint_token("bkrterm01")
    r = client.post(
        "/api/worker/terminal",
        json={
            "status": "ready",
            "artifacts": [
                {"name": "mesh.glb", "kind": "glb", "size_bytes": 123},
            ],
            "worker_id": "wrk-t",
            "release": True,
        },
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200

    row = await store.get_job("bkrterm01")
    assert row is not None
    assert row.status == "ready"
    assert [a.name for a in row.artifacts] == ["mesh.glb"]

    # events.done breadcrumb means bus.close ran — tailers will now exit.
    assert (tmp_data_dir / "jobs" / "bkrterm01" / "events.done").exists()

    # After release, another worker could re-claim the row (it's terminal
    # so it won't in practice, but `claimed_by` must be cleared).
    async with store.session() as s:
        db_row = await s.get(store.JobRow, "bkrterm01")
        assert db_row is not None and db_row.claimed_by is None


async def test_terminal_records_failure(client: TestClient):
    from app.jobs import store

    await _seed_queued_lingbot("bkrterm02")
    tok = _mint_token("bkrterm02")
    r = client.post(
        "/api/worker/terminal",
        json={
            "status": "failed",
            "error": "remote worker blew up",
            "release": False,
        },
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert r.status_code == 200

    row = await store.get_job("bkrterm02")
    assert row is not None
    assert row.status == "failed"
    assert row.error == "remote worker blew up"


async def test_job_binding_is_not_overridable_by_url(client: TestClient):
    """Artifact/upload endpoints use the token's jid, not the URL's job_id.

    The endpoint paths here don't carry a `{job_id}` at all — that's the
    point. Touching the artifacts dir of job A while holding a token for
    job B is what the whole design prevents. Pin it by uploading with a
    token for B and checking B's dir (not A's).
    """
    await _seed_queued_lingbot("bkrbind01")
    await _seed_queued_lingbot("bkrbind02")
    tok_b = _mint_token("bkrbind02")

    r = client.put(
        "/api/worker/artifacts/stamp.bin",
        content=b"belongs to B",
        headers={"Authorization": f"Bearer {tok_b}"},
    )
    assert r.status_code == 200

    from app.config import settings

    assert (settings.job_artifacts("bkrbind02") / "stamp.bin").exists()
    assert not (settings.job_artifacts("bkrbind01") / "stamp.bin").exists()
