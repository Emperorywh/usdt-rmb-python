"""Async ingestion orchestrator.

Runs three concurrent workers per process:

1. WebSocket consumers - one task per ``ExchangeWebSocketClient``; persists
   trades / orderbook / funding / OI events into PostgreSQL.
2. REST poller - periodically pulls funding rate + OI as a recovery path.
3. On-chain poller - periodically writes mock on-chain metrics.

Trades and orderbook writes are buffered briefly to amortise DB round-trips.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Sequence

from app.config import Settings
from app.data_ingestion.base import (
    ExchangeRestClient,
    ExchangeWebSocketClient,
    OnchainProvider,
)
from app.data_storage.repositories import Repositories
from app.logging_config import get_logger

logger = get_logger(__name__)


class IngestionRunner:
    """Owns the lifecycle of all ingestion tasks."""

    def __init__(
        self,
        settings: Settings,
        repos: Repositories,
        ws_clients: Sequence[ExchangeWebSocketClient],
        rest_client: ExchangeRestClient,
        onchain: OnchainProvider,
        trade_flush_size: int = 50,
        trade_flush_interval: float = 1.0,
    ):
        self.settings = settings
        self.repos = repos
        self.ws_clients = list(ws_clients)
        self.rest_client = rest_client
        self.onchain = onchain
        self.trade_flush_size = trade_flush_size
        self.trade_flush_interval = trade_flush_interval

        self._tasks: List[asyncio.Task[Any]] = []
        self._trade_buffer: List[Dict[str, Any]] = []
        self._trade_lock = asyncio.Lock()
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._tasks:
            return
        logger.info(
            "Starting ingestion: %d ws-clients / rest poller / onchain poller",
            len(self.ws_clients),
        )
        for client in self.ws_clients:
            self._tasks.append(asyncio.create_task(self._run_ws(client), name=f"ws-{client.name}"))
        self._tasks.append(asyncio.create_task(self._run_trade_flusher(), name="trade-flusher"))
        self._tasks.append(asyncio.create_task(self._run_rest_poller(), name="rest-poller"))
        self._tasks.append(asyncio.create_task(self._run_onchain_poller(), name="onchain-poller"))

    async def stop(self) -> None:
        if not self._tasks:
            return
        logger.info("Stopping ingestion runner")
        self._stopping.set()
        for client in self.ws_clients:
            stop = getattr(client, "stop", None)
            if callable(stop):
                try:
                    await stop()
                except Exception:  # noqa: BLE001
                    pass
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()
        await self._flush_trades(force=True)
        logger.info("Ingestion runner stopped")

    # ------------------------------------------------------------------
    # workers
    # ------------------------------------------------------------------
    async def _run_ws(self, client: ExchangeWebSocketClient) -> None:
        try:
            async for event in client.stream():
                await self._dispatch(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("WS client %s crashed", client.name)

    async def _dispatch(self, event: Dict[str, Any]) -> None:
        etype = event.get("type")
        try:
            if etype == "trade":
                await self._buffer_trade(event)
            elif etype == "orderbook":
                await self.repos.insert_orderbook(
                    exchange=event["exchange"],
                    symbol=event["symbol"],
                    ts=event["ts"],
                    bids=event["bids"],
                    asks=event["asks"],
                )
            elif etype == "funding_rate":
                await self.repos.insert_funding_rate(
                    exchange=event["exchange"],
                    symbol=event["symbol"],
                    ts=event["ts"],
                    funding_rate=event["funding_rate"],
                    next_funding_ts=event.get("next_funding_ts"),
                )
            elif etype == "open_interest":
                await self.repos.insert_open_interest(
                    exchange=event["exchange"],
                    symbol=event["symbol"],
                    ts=event["ts"],
                    oi=event["oi"],
                    oi_ccy=event.get("oi_ccy"),
                )
            # tickers are informational only - we don't persist them
        except Exception:
            logger.exception("Failed to persist event type=%s", etype)

    async def _buffer_trade(self, event: Dict[str, Any]) -> None:
        async with self._trade_lock:
            self._trade_buffer.append(event)
            should_flush = len(self._trade_buffer) >= self.trade_flush_size
        if should_flush:
            await self._flush_trades()

    async def _run_trade_flusher(self) -> None:
        try:
            while not self._stopping.is_set():
                try:
                    await asyncio.wait_for(
                        self._stopping.wait(), timeout=self.trade_flush_interval
                    )
                except asyncio.TimeoutError:
                    pass
                await self._flush_trades()
        except asyncio.CancelledError:
            raise

    async def _flush_trades(self, force: bool = False) -> None:
        async with self._trade_lock:
            if not self._trade_buffer:
                return
            batch = self._trade_buffer
            self._trade_buffer = []
        try:
            await self.repos.insert_trades(batch)
            logger.debug("Flushed %d trades", len(batch))
        except Exception:
            logger.exception("Trade flush failed (lost %d rows)", len(batch))

    async def _run_rest_poller(self) -> None:
        interval = self.settings.rest_poll_interval_seconds
        try:
            while not self._stopping.is_set():
                for symbol in self.settings.symbols:
                    try:
                        fr = await self.rest_client.fetch_funding_rate(symbol)
                        await self.repos.insert_funding_rate(
                            exchange=fr["exchange"],
                            symbol=fr["symbol"],
                            ts=fr["ts"],
                            funding_rate=fr["funding_rate"],
                            next_funding_ts=fr.get("next_funding_ts"),
                        )
                    except Exception:
                        logger.exception("REST funding poll failed for %s", symbol)
                    try:
                        oi = await self.rest_client.fetch_open_interest(symbol)
                        await self.repos.insert_open_interest(
                            exchange=oi["exchange"],
                            symbol=oi["symbol"],
                            ts=oi["ts"],
                            oi=oi["oi"],
                            oi_ccy=oi.get("oi_ccy"),
                        )
                    except Exception:
                        logger.exception("REST OI poll failed for %s", symbol)
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    async def _run_onchain_poller(self) -> None:
        # Sample on-chain metrics every minute.
        try:
            while not self._stopping.is_set():
                try:
                    metrics = await self.onchain.fetch_metrics()
                    await self.repos.insert_onchain(metrics)
                except Exception:
                    logger.exception("Onchain poll failed")
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=60)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
