"""OKX V5 public WebSocket client.

Channels subscribed: ``trades``, ``books5``, ``tickers``,
``funding-rate``, ``open-interest``, ``liquidation-orders``（P0 新增）.

Architectural note: funding-rate / open-interest are now consumed via the
**WebSocket as the primary path** (push-based, no rate-limit cost). The REST
client in :mod:`app.data_ingestion.okx_rest` is kept around as a *fallback*
that the runner only invokes when WS data goes stale (cold start, long
silence). This avoids the situation where a flaky REST egress (e.g. CN ->
www.okx.com being throttled) leaves these two tables empty even though the
WS endpoint at ``ws.okx.com`` is fine.

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
        - 订阅 trades / books5 / tickers / funding-rate / open-interest
          五个频道，输出统一格式的事件流。
        - funding-rate / open-interest 走 WS 主路径（推送，无频控成本），
          REST 仅作为 watchdog 兜底（参见 runner._run_rest_watchdog）。
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

    def update_contract_value(self, symbol: str, ct_val: float) -> None:
        """
        动态更新指定 symbol 的合约面值
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码
            ct_val: 新的合约面值（每张合约对应多少基础币种）
        说明：
            - 容器在启动后会异步从 OKX instruments 接口拉取真实 ctVal，
              拿到后通过本方法写回；写入后下一条 trade/book 推送即按真值
              换算，无需重启 WS。
            - dict 写入是原子的，不需要加锁。
        """
        self._contract_values[symbol] = float(ct_val)

    def subscribe_symbols(self) -> List[str]:
        return list(self.symbols)

    def _build_subscribe_args(self) -> List[Dict[str, str]]:
        """
        构造订阅参数列表
        ---------------------------------------------------------------
        说明：
            - funding-rate / open-interest 也走 WS 主路径，REST 仅在
              数据陈旧时由 runner 的 watchdog 兜底拉一次。
            - liquidation-orders 是 OKX 的"全市场推送"频道，按 instType
              订阅而非 instId（订阅时给 instId 会被服务端忽略），客户端
              必须自行按 self.symbols 过滤，否则 BTC / SOL 等无关合约的
              强平也会被写到我们的 liquidations 表里污染数据。
        """
        args: List[Dict[str, str]] = []
        books_channel = "books5" if self.depth <= 5 else "books"
        for sym in self.symbols:
            args.append({"channel": "trades", "instId": sym})
            args.append({"channel": books_channel, "instId": sym})
            args.append({"channel": "tickers", "instId": sym})
            args.append({"channel": "funding-rate", "instId": sym})
            args.append({"channel": "open-interest", "instId": sym})
        # liquidation-orders 全市场频道：每种 instType 订阅一次即可。
        # 当前我们只关心 SWAP（永续合约），如果以后扩到 FUTURES 再加。
        args.append({"channel": "liquidation-orders", "instType": "SWAP"})
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
                    "OKX WS 异常：%s；将在 %.1fs 后重连", exc, backoff
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2 + random.random(), 30.0)

    async def _stream_once(self) -> AsyncIterator[Dict[str, Any]]:
        logger.info("正在连接 OKX WebSocket %s", self.ws_url)
        async with websockets.connect(
            self.ws_url,
            ping_interval=None,
            close_timeout=5,
            max_size=2**22,
        ) as ws:
            sub_msg = {"op": "subscribe", "args": self._build_subscribe_args()}
            await ws.send(json.dumps(sub_msg))
            logger.info(
                "OKX WS 已订阅：%d 个频道，symbols=%s",
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
                logger.warning("OKX WS 连接已断开：%s", exc)
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
            logger.debug("心跳循环已结束：%s", exc)

    async def _handle_message(self, msg: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        event_type = msg.get("event")
        if event_type:
            if event_type == "error":
                logger.error("OKX WS 服务端错误：%s", msg)
            else:
                logger.debug("OKX WS 控制消息：%s", msg)
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
        elif channel == "funding-rate":
            # OKX V5 funding-rate 频道：每 ~1 分钟推一次，是 funding 数据
            # 的主路径；nextFundingTime 用 fundingTime 兜底，老返回字段较杂。
            for item in data:
                ts = (
                    _ms_to_dt(item.get("ts"))
                    or _ms_to_dt(item.get("fundingTime"))
                    or datetime.now(timezone.utc)
                )
                next_funding_ts = _ms_to_dt(
                    item.get("nextFundingTime") or item.get("fundingTime")
                )
                yield {
                    "type": "funding_rate",
                    "exchange": self.name,
                    "symbol": symbol,
                    "ts": ts,
                    "funding_rate": float(item.get("fundingRate") or 0.0),
                    "next_funding_ts": next_funding_ts,
                }
        elif channel == "open-interest":
            # OKX V5 open-interest 频道：~3s 推一次，比 60s REST 信息密度高
            # 一个量级。oi 单位是张数，oiCcy 是按基础币种折算后的数量。
            for item in data:
                ts = _ms_to_dt(item.get("ts")) or datetime.now(timezone.utc)
                yield {
                    "type": "open_interest",
                    "exchange": self.name,
                    "symbol": symbol,
                    "ts": ts,
                    "oi": float(item.get("oi") or 0.0),
                    "oi_ccy": float(item.get("oiCcy") or 0) or None,
                }
        elif channel == "liquidation-orders":
            # OKX V5 liquidation-orders 是全市场推送：data 里每条都带各自的
            # instId，客户端必须按 self.symbols 过滤掉不感兴趣的合约。
            # 单条结构：
            #   {"instType":"SWAP","instId":"ETH-USDT-SWAP",
            #    "details":[{"side":"sell","posSide":"long","bkPx":"...","sz":"..."}, ...]}
            #
            # side 字段语义：
            #   side=sell → 平多操作 → 一个多头仓位被强平 → 我们记 'long'
            #   side=buy  → 平空操作 → 一个空头仓位被强平 → 我们记 'short'
            # 部分版本会同时给 posSide，优先用 posSide（语义直接），
            # 不可用时按上面的反向映射回退。
            async for ev in self._emit_liquidations(data):
                yield ev

    async def _emit_liquidations(
        self, data: List[Dict[str, Any]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        把一批 liquidation-orders 推送展开成扁平的强平事件流
        ---------------------------------------------------------------
        参数：
            data: msg["data"]，每个元素带 instId 与 details 列表。
        说明：
            - 必须按 self.symbols 客户端过滤，详见 _build_subscribe_args 注释。
            - size 同样按 ctVal 换算到基础币种数量。
            - notional = price * size（已换算）。
        产出：
            type=liquidation 的事件 dict，字段：
                exchange / symbol / ts / side(long|short) / price / size / notional
        """
        for entry in data:
            inst_id = entry.get("instId")
            if not inst_id or inst_id not in self.symbols:
                continue
            ct_val = self._ct_val(inst_id)
            details = entry.get("details") or []
            for det in details:
                side_raw = (det.get("side") or "").lower()
                pos_side = (det.get("posSide") or "").lower()
                if pos_side in ("long", "short"):
                    side_norm = pos_side
                elif side_raw == "sell":
                    side_norm = "long"
                elif side_raw == "buy":
                    side_norm = "short"
                else:
                    continue

                price_str = det.get("bkPx") or det.get("fillPx") or det.get("px")
                size_str = det.get("sz")
                ts_raw = det.get("ts") or det.get("cTime")
                if price_str is None or size_str is None or ts_raw is None:
                    continue

                try:
                    price = float(price_str)
                    size = float(size_str) * ct_val
                except (TypeError, ValueError):
                    continue

                yield {
                    "type": "liquidation",
                    "exchange": self.name,
                    "symbol": inst_id,
                    "ts": _ms_to_dt(ts_raw),
                    "side": side_norm,
                    "price": price,
                    "size": size,
                    "notional": price * size,
                }
