"""OKX V5 public WebSocket client.

Channels subscribed: ``trades``, ``books5``, ``tickers``.

Funding rate / open interest are intentionally NOT subscribed on the WS:
they update slowly (funding settles every 8 h, OI updates every few seconds
with mostly identical values), so polling the REST endpoint once per minute
is cheaper and avoids duplicate writes against the unique-keyed tables.

Trade ``size`` and orderbook ``size`` come back from OKX in *contracts*
(张) for SWAP / FUTURES. We multiply by ``ct_val`` (contract face value)
on emission so downstream factor / signal code consistently sees the
quantity in the base currency (e.g. ETH).

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
    """
    OKX V5 公共行情 WebSocket 客户端
    ---------------------------------------------------------------
    职责：
        - 订阅 trades / books5 / tickers 三个频道，输出标准化事件流。
        - 不订阅 funding-rate / open-interest，这两类数据由 REST 60s 轮询。
        - 把交易所推送中的 sz（合约张数）按 contract_values[symbol]
          换算成基础币种数量（如 ETH），统一全链路单位。
    """

    name = "okx"

    def __init__(
        self,
        ws_url: str,
        symbols: List[str],
        depth: int = 5,
        contract_values: Optional[Dict[str, float]] = None,
        default_contract_value: float = 1.0,
    ):
        """
        构造函数
        ---------------------------------------------------------------
        参数：
            ws_url:                 OKX 公共 WS 地址
            symbols:                需要订阅的 symbol 列表
            depth:                  订单簿深度（<=5 用 books5，否则 books）
            contract_values:        {symbol: ctVal} 合约面值映射，可空
            default_contract_value: 在 contract_values 中找不到时使用的兜底值
        """
        self.ws_url = ws_url
        self.symbols = symbols
        self.depth = depth
        self._contract_values: Dict[str, float] = dict(contract_values or {})
        self._default_ct_val = float(default_contract_value)
        self._stop = asyncio.Event()

    def _ct_val(self, symbol: str) -> float:
        """
        获取指定 symbol 的合约面值
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码
        返回：
            ctVal（每张合约对应多少基础币种），找不到时返回默认值。
        """
        return self._contract_values.get(symbol, self._default_ct_val)

    def subscribe_symbols(self) -> List[str]:
        return list(self.symbols)

    def _build_subscribe_args(self) -> List[Dict[str, str]]:
        """
        构造订阅参数列表
        ---------------------------------------------------------------
        说明：
            funding-rate / open-interest 不再订阅，由 REST 轮询负责。
        """
        args: List[Dict[str, str]] = []
        books_channel = "books5" if self.depth <= 5 else "books"
        for sym in self.symbols:
            args.append({"channel": "trades", "instId": sym})
            args.append({"channel": books_channel, "instId": sym})
            args.append({"channel": "tickers", "instId": sym})
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

        ct_val = self._ct_val(symbol) if symbol else self._default_ct_val

        if channel == "trades":
            for item in data:
                # OKX SWAP / FUTURES 推送中 sz 单位是张数，乘以 ctVal 转成基础币种
                yield {
                    "type": "trade",
                    "exchange": self.name,
                    "symbol": symbol,
                    "ts": _ms_to_dt(item["ts"]),
                    "price": float(item["px"]),
                    "size": float(item["sz"]) * ct_val,
                    "side": item["side"],
                    "trade_id": item.get("tradeId"),
                }
        elif channel in ("books5", "books"):
            for item in data:
                # 同上：盘口每档 size 也按 ctVal 换算成基础币种数量
                yield {
                    "type": "orderbook",
                    "exchange": self.name,
                    "symbol": symbol,
                    "ts": _ms_to_dt(item["ts"]),
                    # OKX format: [price, size, liquidated_orders, num_orders]
                    "bids": [
                        [float(b[0]), float(b[1]) * ct_val]
                        for b in item.get("bids", [])
                    ],
                    "asks": [
                        [float(a[0]), float(a[1]) * ct_val]
                        for a in item.get("asks", [])
                    ],
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
        # funding-rate / open-interest 故意不在 WS 消费，统一交给 REST 轮询。
