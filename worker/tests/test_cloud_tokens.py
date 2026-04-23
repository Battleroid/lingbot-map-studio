"""HMAC token invariants for the broker.

Cover the failure modes the broker relies on (bad sig, expired, wrong
scope, wrong job, malformed). These are the only things between a
leaked token and the studio running a stranger's code; the suite pins
every branch.

Run: `pytest worker/tests/test_cloud_tokens.py -q`.
"""

from __future__ import annotations

import pytest


def _mint(**overrides):
    from app.cloud import tokens

    defaults = dict(
        job_id="job-abc",
        execution_target="runpod",
        scopes=["claim", "events", "artifacts"],
        ttl_s=60,
        key="test-key",
        now=1_700_000_000,
    )
    defaults.update(overrides)
    return tokens.mint(**defaults)


def test_mint_and_verify_roundtrip():
    from app.cloud import tokens

    tok = _mint()
    payload = tokens.verify(
        tok, key="test-key", required_scope="events", now=1_700_000_010
    )
    assert payload.job_id == "job-abc"
    assert payload.execution_target == "runpod"
    assert payload.scopes == frozenset({"claim", "events", "artifacts"})


def test_verify_rejects_bad_signature():
    from app.cloud import tokens

    tok = _mint()
    with pytest.raises(tokens.TokenError):
        tokens.verify(tok, key="other-key", required_scope="events", now=1_700_000_010)


def test_verify_rejects_expired_token():
    from app.cloud import tokens

    tok = _mint(ttl_s=30)
    # Past the expiry.
    with pytest.raises(tokens.TokenError):
        tokens.verify(
            tok, key="test-key", required_scope="events", now=1_700_000_100
        )


def test_verify_enforces_required_scope():
    from app.cloud import tokens

    # Mint with just "events" + "heartbeat"; ask for "artifacts".
    tok = _mint(scopes=["events", "heartbeat"])
    with pytest.raises(tokens.TokenError):
        tokens.verify(
            tok, key="test-key", required_scope="artifacts", now=1_700_000_010
        )


def test_verify_enforces_job_binding_when_supplied():
    from app.cloud import tokens

    tok = _mint(job_id="job-abc")
    with pytest.raises(tokens.TokenError):
        tokens.verify(
            tok,
            key="test-key",
            required_scope="events",
            expected_job_id="job-xyz",
            now=1_700_000_010,
        )


def test_verify_rejects_malformed_token():
    from app.cloud import tokens

    with pytest.raises(tokens.TokenError):
        tokens.verify("not-a-token", key="test-key", required_scope="events")


def test_mint_rejects_unknown_scope():
    from app.cloud import tokens

    with pytest.raises(ValueError):
        tokens.mint(
            job_id="j",
            execution_target="fake",
            scopes=["events", "doesnotexist"],
            ttl_s=10,
            key="k",
        )


def test_verify_rejects_unknown_required_scope_at_call_site():
    """Catch typos in broker handler code, not at a deploy-time failure."""
    from app.cloud import tokens

    tok = _mint()
    with pytest.raises(ValueError):
        tokens.verify(tok, key="test-key", required_scope="ooops")


def test_nonce_differs_across_mints():
    tok1 = _mint()
    tok2 = _mint()
    # Same payload except the nonce — so the final base64 must differ.
    assert tok1 != tok2
