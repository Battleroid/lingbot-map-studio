"""R6 cloud-API endpoints: providers, session credentials, cost readout.

These back the frontend ExecutionPanel / CloudCredentialsDialog /
CostPreview trio, plus the JobStatusStrip's cloud badge. The session-
creds store is in-memory and the cost cells read straight off the job
row — neither path goes anywhere near a provider's real API, so this
file is pure FastAPI / SQLite and safe to run in CI.

We use `TestClient` against the real app and lean on the existing
`tmp_data_dir` fixture so SQLite and the artifacts dir are both swept
between tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_data_dir: Path):
    from app.main import app

    with TestClient(app) as c:
        yield c


async def _seed_local_lingbot(job_id: str) -> None:
    """Create a queued local lingbot job so the cost endpoint has
    something concrete to read back."""
    from app.jobs import store
    from app.jobs.schema import Job, LingbotConfig

    await store.init_store()
    now = datetime.now(timezone.utc)
    await store.create_job(
        Job(
            id=job_id,
            status="queued",
            config=LingbotConfig(model_id="lingbot-map", fps=10.0),
            uploads=[],
            artifacts=[],
            created_at=now,
            updated_at=now,
        ),
        worker_class="lingbot",
    )


# --- /api/cloud/providers ----------------------------------------------


def test_providers_always_includes_local(client: TestClient):
    res = client.get("/api/cloud/providers")
    assert res.status_code == 200
    body = res.json()
    assert "local" in body["targets"]
    # Session targets default empty without a header.
    assert body["session_targets"] == []
    assert isinstance(body["cost_cap_cents_default"], int)


def test_providers_reports_session_targets_when_header_present(client: TestClient):
    # First paste creds → server mints a session id in the response.
    res = client.post(
        "/api/cloud/credentials/session",
        json={"provider": "runpod", "values": {"api_key": "k"}},
    )
    assert res.status_code == 201
    sid = res.headers["x-cloud-session"]
    assert sid  # non-empty

    # Second call with that header: runpod shows up under session_targets.
    res2 = client.get("/api/cloud/providers", headers={"x-cloud-session": sid})
    assert res2.status_code == 200
    body = res2.json()
    assert "runpod" in body["session_targets"]


# --- /api/cloud/credentials/session ------------------------------------


def test_session_credentials_require_provider_and_values(client: TestClient):
    # Missing `provider` → 422.
    res = client.post(
        "/api/cloud/credentials/session", json={"values": {"api_key": "k"}}
    )
    assert res.status_code == 422

    # Empty `values` → 422 (prevents accidentally clearing a bag).
    res = client.post(
        "/api/cloud/credentials/session",
        json={"provider": "runpod", "values": {}},
    )
    assert res.status_code == 422


def test_session_credentials_round_trip(client: TestClient):
    res = client.post(
        "/api/cloud/credentials/session",
        json={"provider": "runpod", "values": {"api_key": "secret-1"}},
    )
    assert res.status_code == 201
    sid = res.json()["session_id"]

    # Dispatcher would call this path server-side. We reach into the
    # module directly because there's no user-facing readback of stored
    # secrets (by design).
    from app.cloud import session_creds

    got = session_creds.get_credentials(sid, "runpod")
    assert got == {"api_key": "secret-1"}


def test_session_credentials_never_leak_across_sessions(client: TestClient):
    r1 = client.post(
        "/api/cloud/credentials/session",
        json={"provider": "vast", "values": {"api_key": "a"}},
    )
    sid1 = r1.headers["x-cloud-session"]

    r2 = client.post(
        "/api/cloud/credentials/session",
        json={"provider": "vast", "values": {"api_key": "b"}},
    )
    sid2 = r2.headers["x-cloud-session"]
    assert sid1 != sid2

    from app.cloud import session_creds

    assert session_creds.get_credentials(sid1, "vast") == {"api_key": "a"}
    assert session_creds.get_credentials(sid2, "vast") == {"api_key": "b"}


def test_clear_session_credentials_is_idempotent(client: TestClient):
    # No header → 200 with cleared=false, no error.
    res = client.delete("/api/cloud/credentials/session")
    assert res.status_code == 200
    assert res.json()["cleared"] is False

    # Seed + clear → cleared=true, then subsequent get returns empty.
    r1 = client.post(
        "/api/cloud/credentials/session",
        json={"provider": "runpod", "values": {"api_key": "k"}},
    )
    sid = r1.headers["x-cloud-session"]
    res = client.delete(
        "/api/cloud/credentials/session", headers={"x-cloud-session": sid}
    )
    assert res.status_code == 200
    assert res.json()["cleared"] is True

    from app.cloud import session_creds

    assert session_creds.get_credentials(sid, "runpod") is None


# --- /api/cloud/estimate ------------------------------------------------


def test_estimate_for_local_is_zero(client: TestClient):
    res = client.post(
        "/api/cloud/estimate",
        json={"execution_target": "local"},
    )
    assert res.status_code == 200
    assert res.json()["cents"] == 0


def test_estimate_for_unknown_target_is_400(client: TestClient):
    res = client.post(
        "/api/cloud/estimate",
        json={
            "execution_target": "does-not-exist",
            "instance_spec": {"gpu_class": "rtx4090"},
        },
    )
    assert res.status_code == 400


def test_estimate_for_fake_provider_returns_cents(client: TestClient):
    # `fake` is always registered (no creds needed).
    res = client.post(
        "/api/cloud/estimate",
        json={
            "execution_target": "fake",
            "instance_spec": {"gpu_class": "rtx4090"},
            "expected_duration_s": 600,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["target"] == "fake"
    assert body["expected_duration_s"] == 600
    assert isinstance(body["cents"], int)
    assert body["cents"] >= 0


# --- /api/jobs/{id}/cost ------------------------------------------------


async def test_cost_local_job_zero_fields(client: TestClient):
    await _seed_local_lingbot("cost-local")
    res = client.get("/api/jobs/cost-local/cost")
    assert res.status_code == 200
    body = res.json()
    assert body["job_id"] == "cost-local"
    assert body["execution_target"] == "local"
    assert body["cost_estimate_cents"] == 0
    assert body["cost_actual_cents"] == 0
    # No claim yet on a queued job.
    assert body["elapsed_s"] is None


async def test_cost_reflects_dispatcher_bookkeeping(client: TestClient):
    from app.jobs import store

    await _seed_local_lingbot("cost-remote")
    await store.set_provider_bookkeeping(
        "cost-remote",
        provider_instance_id="pod-abc",
        cost_estimate_cents=137,
        token_hash="hash",
    )
    res = client.get("/api/jobs/cost-remote/cost")
    assert res.status_code == 200
    body = res.json()
    assert body["provider_instance_id"] == "pod-abc"
    assert body["cost_estimate_cents"] == 137


def test_cost_for_missing_job_is_404(client: TestClient):
    res = client.get("/api/jobs/not-a-real-job/cost")
    assert res.status_code == 404
