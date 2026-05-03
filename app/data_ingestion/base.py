"""Abstract base classes for ingestion plugins."""
from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Dict, List


class ExchangeWebSocketClient(abc.ABC):
    """Abstract async WS client. Yields normalised market events."""

    name: str = "base"

    @abc.abstractmethod
    def subscribe_symbols(self) -> List[str]:
        ...

    @abc.abstractmethod
    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Yield events. Each event MUST have a ``type`` key in
        {"trade", "orderbook", "ticker", "funding_rate", "open_interest"}.
        """
        if False:  # pragma: no cover - abstract async generator hint
            yield {}


class ExchangeRestClient(abc.ABC):
    """Abstract REST poller for periodic data."""

    name: str = "base"

    @abc.abstractmethod
    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    async def fetch_open_interest(self, symbol: str) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        ...


class OnchainProvider(abc.ABC):
    """Abstract on-chain metrics provider."""

    name: str = "base"

    @abc.abstractmethod
    async def fetch_metrics(self) -> Dict[str, Any]:
        """Return latest snapshot of on-chain metrics."""
        ...
