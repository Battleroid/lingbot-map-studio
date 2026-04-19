"""Short-TTL HMAC tokens for the remote-worker broker.

The studio issues one token per dispatched job. The remote worker
attaches it as `Authorization: Bearer <token>` on every broker call;
the broker middleware verifies the signature, the expiry, and the
scope, and derives the job_id from the payload so the endpoint can
authorise the operation without trusting URL parameters.

Wire format (compact, self-contained — no DB lookup on verify):

    base64url(payload_json) "." base64url(hmac_sha256(payload_json))

Payload shape:

    {
      "jid": <job_id>,
      "et":  <execution_target>,        # "runpod", "vast", "fake", …
      "sc":  <sorted list of scopes>,   # ["claim","events","artifacts",…]
      "exp": <unix_ts>,                 # issued_at + TTL
      "iat": <unix_ts>,                 # issued_at
      "n":   <hex nonce>                # guards against reuse across jobs
    }

We don't try to revoke tokens individually — the TTL is short and the
cost cap watchdog bounds damage from a leaked token to at most one
cheap cloud job. If we ever need revocation, add a `tok_revoked` table
and a bloom-filter fast-path on verify.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Iterable


class TokenError(RuntimeError):
    """Raised by `verify` on any validation failure.

    Deliberately one class with a `.reason` attribute rather than a
    hierarchy — every failure mode (bad signature, expired, wrong
    scope, wrong job) lands in the same 401 on the broker side, and a
    hierarchy would just tempt callers to distinguish cases that
    shouldn't be distinguishable to the remote worker.
    """


@dataclass(frozen=True)
class TokenPayload:
    job_id: str
    execution_target: str
    scopes: frozenset[str]
    expires_at: int
    issued_at: int
    nonce: str


# Canonical scope strings. A broker endpoint requires one of these; a
# token may carry many. Kept tight — adding a scope is a code change,
# not a config change.
SCOPES = frozenset(
    {
        "claim",
        "events",
        "artifacts",
        "uploads",
        "checkpoints",
        "cancel",
        "heartbeat",
        "terminal",
    }
)


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _hmac(key: str, payload: bytes) -> bytes:
    return hmac.new(key.encode("utf-8"), payload, hashlib.sha256).digest()


def mint(
    *,
    job_id: str,
    execution_target: str,
    scopes: Iterable[str],
    ttl_s: int,
    key: str,
    now: int | None = None,
) -> str:
    """Sign a token that authorises a remote worker to work on one job.

    `scopes` must be a subset of `SCOPES`; unknown scopes raise so a
    typo doesn't silently yield a broader-than-intended token.
    """
    scope_set = frozenset(scopes)
    unknown = scope_set - SCOPES
    if unknown:
        raise ValueError(f"unknown scopes: {sorted(unknown)}")

    issued_at = int(now if now is not None else time.time())
    expires_at = issued_at + ttl_s
    payload = {
        "jid": job_id,
        "et": execution_target,
        "sc": sorted(scope_set),
        "exp": expires_at,
        "iat": issued_at,
        "n": secrets.token_hex(8),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    sig = _hmac(key, payload_bytes)
    return f"{_b64url_encode(payload_bytes)}.{_b64url_encode(sig)}"


def verify(
    token: str,
    *,
    key: str,
    required_scope: str,
    expected_job_id: str | None = None,
    now: int | None = None,
) -> TokenPayload:
    """Parse + validate a token. Raises `TokenError` on any failure.

    On success returns the decoded payload so the caller can read the
    job_id and execution_target without re-parsing. Always requires a
    `required_scope` — an "any scope is fine" call would be a bug, so
    we refuse to compile one.
    """
    if required_scope not in SCOPES:
        raise ValueError(f"required_scope={required_scope!r} is not a known scope")

    parts = token.split(".")
    if len(parts) != 2:
        raise TokenError("malformed token")
    payload_b64, sig_b64 = parts
    try:
        payload_bytes = _b64url_decode(payload_b64)
        sig = _b64url_decode(sig_b64)
    except Exception as exc:  # noqa: BLE001
        raise TokenError(f"not base64: {exc}") from exc

    expected_sig = _hmac(key, payload_bytes)
    if not hmac.compare_digest(sig, expected_sig):
        raise TokenError("bad signature")

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise TokenError(f"bad payload json: {exc}") from exc

    # Strict field presence — silent fallbacks here would let an attacker
    # who tampered with the key discover gaps rather than fail closed.
    for field in ("jid", "et", "sc", "exp", "iat", "n"):
        if field not in payload:
            raise TokenError(f"missing field: {field}")

    current = int(now if now is not None else time.time())
    if current >= int(payload["exp"]):
        raise TokenError("token expired")

    token_scopes = frozenset(payload["sc"])
    if required_scope not in token_scopes:
        raise TokenError(f"token missing required scope: {required_scope}")

    if expected_job_id is not None and payload["jid"] != expected_job_id:
        raise TokenError("token bound to a different job")

    return TokenPayload(
        job_id=payload["jid"],
        execution_target=payload["et"],
        scopes=token_scopes,
        expires_at=int(payload["exp"]),
        issued_at=int(payload["iat"]),
        nonce=payload["n"],
    )
