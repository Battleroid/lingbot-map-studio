"""Shared HTTP plumbing for cloud-provider adapters.

Every REST-backed provider (RunPod, Vast, Lambda Labs, Paperspace)
needs the same three things: build an httpx client with auth headers,
allow tests to inject a `MockTransport`, and clean up the client on
close. Pulling it here keeps the adapters focused on the per-provider
API shape instead of re-implementing boilerplate.

Deliberately not its own ABC — composition (each provider owns one of
these via `self._http`) keeps the adapter classes inheriting directly
from `CloudProvider` so they stay readable and type-checkable.
"""

from __future__ import annotations

from typing import Optional

import httpx


class ProviderHttp:
    """Lazy-constructed httpx client with a per-class transport hook.

    Construction deliberately picks up the `transport_override` hook
    from the *subclass* (passed in by each adapter) so tests can swap
    the transport per-provider without touching a global. Attach an
    `httpx.MockTransport` to the subclass and clear the cached client:

        RunPodProvider.transport_override = httpx.MockTransport(responder)
        provider._http.reset()
    """

    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str],
        transport: Optional[httpx.BaseTransport] = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = dict(headers)
        self._transport = transport
        self._timeout_s = timeout_s
        self._client: Optional[httpx.AsyncClient] = None

    def set_transport(self, transport: Optional[httpx.BaseTransport]) -> None:
        """Swap the transport. Also resets the cached client so the next
        call actually sees the new transport."""
        self._transport = transport
        self._client = None

    def client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        kwargs = {
            "base_url": self._base_url,
            "headers": self._headers,
            "timeout": self._timeout_s,
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def reset(self) -> None:
        """Drop the cached client without closing it (tests own lifetime)."""
        self._client = None


__all__ = ["ProviderHttp"]
