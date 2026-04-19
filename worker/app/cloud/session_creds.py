"""In-memory per-session cloud-provider credential stash.

R6 lets a user paste provider API keys in the browser when the studio's
env doesn't carry them. Those keys must never hit SQLite — if the user
closes the tab they're gone, and a compromised DB snapshot can't leak
them.

Layout: a plain dict keyed by (session_token, provider_id). The session
token is a random UUID the browser stores in sessionStorage and sends on
every cloud-related request. Nothing here survives an API process
restart, which is exactly the point.

This module is deliberately tiny and sync — the dispatcher and estimate
endpoints check it on-request to decide whether a provider is usable
for *this* session even if registration from settings alone failed.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from time import time
from typing import Dict, Optional


@dataclass(frozen=True)
class SessionCreds:
    # Free-form mapping of provider-specific fields (api_key, region,
    # project_id, …). Adapters reach in with known keys; we don't
    # validate schema here because each provider has its own shape.
    values: Dict[str, str]
    created_at: float


# Session ids have one entry per provider the user configured. A short
# TTL keeps a rarely-used tab from keeping keys forever in memory.
_TTL_S = 8 * 60 * 60
_store: Dict[str, Dict[str, SessionCreds]] = {}
_lock = threading.Lock()


def new_session() -> str:
    """Mint a fresh session token. The browser persists this in
    sessionStorage; every `/api/cloud/*` request is expected to send it
    via a header so the server can scope credential lookups.
    """
    return secrets.token_urlsafe(24)


def _sweep_locked(now: float) -> None:
    """Drop any sessions whose oldest entry is past TTL."""
    dead = []
    for sid, entries in _store.items():
        if all(now - c.created_at > _TTL_S for c in entries.values()):
            dead.append(sid)
    for sid in dead:
        _store.pop(sid, None)


def set_credentials(
    session_id: str, provider_id: str, values: Dict[str, str]
) -> None:
    """Install credentials for `provider_id` scoped to `session_id`.

    Replaces whatever was there (no merge) — the caller is expected to
    post the complete credential bag for a provider.
    """
    now = time()
    with _lock:
        _sweep_locked(now)
        _store.setdefault(session_id, {})[provider_id] = SessionCreds(
            values=dict(values), created_at=now
        )


def get_credentials(
    session_id: Optional[str], provider_id: str
) -> Optional[Dict[str, str]]:
    """Return the credential bag for `(session_id, provider_id)` or None.

    A `None` session_id is accepted (and always returns None) so the
    dispatcher can call this unconditionally without branching on header
    presence.
    """
    if not session_id:
        return None
    with _lock:
        _sweep_locked(time())
        entry = _store.get(session_id, {}).get(provider_id)
        return dict(entry.values) if entry is not None else None


def clear_session(session_id: str) -> None:
    """Drop every credential for `session_id`. Used by the `/logout`
    path on the browser side so closing the tab or switching accounts
    wipes state explicitly instead of waiting for the TTL sweep."""
    with _lock:
        _store.pop(session_id, None)


def known_providers_for_session(session_id: Optional[str]) -> list[str]:
    """List provider ids the session has credentials for. Used by the
    `/api/cloud/providers` endpoint to mark providers as usable by
    paste-only flow, even when settings alone didn't register them."""
    if not session_id:
        return []
    with _lock:
        _sweep_locked(time())
        return sorted(_store.get(session_id, {}).keys())


__all__ = [
    "SessionCreds",
    "clear_session",
    "get_credentials",
    "known_providers_for_session",
    "new_session",
    "set_credentials",
]
