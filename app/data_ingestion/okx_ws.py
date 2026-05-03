"""OKX V5 public WebSocket client.

Channels subscribed: ``trades``, ``books5``, ``tickers``. Funding rate / OI are
also broadcast on the public WS via ``funding-rate`` and ``open-interest``
channels which we subscribe to as well, freeing the REST poller to serve as a
recovery path.

Auto-reconnects with exponential backoff; sends ``ping`` every 25 s to keep the
connection alive (OKX disconnects idle sockets after ~30 s).
"""
from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from app.data_ingestion.base import ExchangeWebSocketClient
from app.logging_config import get_logger

logger = get_logger(__name__)


def _ms_to_dt(ms: str | int) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


class OKXWebSocketClient(ExchangeWebSocketClient):
    name = "okx"

    def __init__(self, ws_url: str, symbols: List[str], depth: int = 5):
        self.ws_url = ws_url
        self.symbols = symbols
        self.depth = depth
        self._stop = asyncio.Event()

    def subscribe_symbols(self) -> List[str]:
        return list(self.symbols)

    def _build_subscribe_args(self) -> List[Dict[str, str]]:
        args: List[Dict[str, str]] = []
        books_channel = "books5" if self.depth <= 5 else "books"
        for sym in self.symbols:
            args.append({"channel": "trades", "instId": sym})
            args.append({"channel": books_channel, "instId": sym})
            args.append({"channel": "tickers", "instId": sym})
            args.append({"channel": "funding-rate", "instId": sym})
            args.append({"channel": "open-interest", "instId": sym})
        return args

    async def stop(self) -> None:
        self._stop.set()

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async for event in self._stream_once():
                    backoff = 1.0
                    yield event
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - top-level resilience
                logger.warning(
                    "OKX WS error: %s; reconnecting in %.1fs", exc, backoff
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2 + random.random(), 30.0)

    async def _stream_once(self) -> AsyncIterator[Dict[str, Any]]:
        logger.info("Connecting OKX WS %s", self.ws_url)
        async with websockets.connect(
            self.ws_url,
            ping_interval=None,
            close_timeout=5,
            max_size=2**22,
        ) as ws:
            sub_msg = {"op": "subscribe", "args": self._build_subscribe_args()}
            await ws.send(json.dumps(sub_msg))
            logger.info(
                "OKX WS subscribed: %d channels for symbols=%s",
                len(sub_msg["args"]),
                self.symbols,
            )

            ping_task = asyncio.create_task(self._keepalive(ws))
            try:
                while not self._stop.is_set():
                    raw = await ws.recv()
                    if raw == "pong":
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    async for ev in self._handle_message(msg):
                        yield ev
            except ConnectionClosed as exc:
                logger.warning("OKX WS connection closed: %s", exc)
                raise
            finally:
                ping_task.cancel()

    async def _keepalive(self, ws) -> None:
        try:
            while True:
                await asyncio.sleep(25)
                await ws.send("ping")
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("ping loop ended: %s", exc)

    async def _handle_message(self, msg: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        event_type = msg.get("event")
        if event_type:
            if event_type == "error":
                logger.error("OKX WS server error: %s", msg)
            else:
                logger.debug("OKX WS control: %s", msg)
            return

        arg = msg.get("arg") or {}
        channel = arg.get("channel")
        symbol = arg.get("instId")
        data = msg.get("data") or []
        if not channel or not data:
            return

        if channel == "trades":
            for item in data:
                yield {
                    "type": "trade",
                    "exchange": self.name,
                    "symbol": symbol,
                    "ts": _ms_to_dt(item["ts"]),
                    "price": float(item["px"]),
                    "size": float(item["sz"]),
                    "side": item["side"],
                    "trade_id": item.get("tradeId"),
                }
        elif channel in ("books5", "books"):
            for item in data:
                yield {
                    "type": "orderbook",
                    "exchange": self.name,
                    "symbol": symbol,
                    "ts": _ms_to_dt(item["ts"]),
                    # OKX format: [price, size, liquidated_orders, num_orders]
                    "bids": [[float(b[0]), float(b[1])] for b in item.get("bids", [])],
                    "asks": [[float(a[0]), float(a[1])] for a in item.get("asks", [])],
                }
        elif channel == "tickers":
            for item in data:
                yield {
                    "type": "ticker",
                    "exchange": self.name,
                    "symbol": symbol,
                    "ts": _ms_to_dt(item["ts"]),
                    "last": float(item["last"]),
                    "bid": float(item.get("bidPx") or 0) or None,
                    "ask": float(item.get("askPx") or 0) or None,
                }
        elif channel == "funding-rate":
            for item in data:
                yield {
                    "type": "funding_rate",
                    "exchange": self.name,
                    "symbol": symbol,
                    "ts": _ms_to_dt(item["ts"]),
                    "funding_rate": float(item["fundingRate"]),
                    "next_funding_ts": _ms_to_dt(item["nextFundingTime"])
                    if item.get("nextFundingTime")
                    else None,
                }
        elif channel == "open-interest":
            for item in data:
                yield {
                    "type": "open_interest",
                    "exchange": self.name,
                    "symbol": symbol,
                    "ts": _ms_to_dt(item["ts"]),
                    "oi": float(item.get("oi") or 0),
                    "oi_ccy": float(item.get("oiCcy") or 0) or None,
                }
