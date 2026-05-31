"""数据采集编排器（重构后）。

仅负责：
1. WS 连接管理与事件分发 → 各 Worker
2. Worker 生命周期管理（start / stop）
3. WS 通道健康度追踪
4. funding_rate / open_interest 直写（无需缓冲）

所有后台任务逻辑已拆分到 ``app.data_ingestion.workers/`` 下的独立 Worker。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import Settings
from app.data_ingestion.base import ExchangeRestClient, ExchangeWebSocketClient
from app.data_ingestion.workers.trade_buffer import TradeBufferWorker
from app.data_ingestion.workers.orderbook_writer import OrderbookWriter
from app.data_ingestion.workers.rest_watchdog import RestWatchdog
from app.data_ingestion.workers.retention import RetentionCleaner
from app.data_storage.repos.derivatives_repo import DerivativesRepo
from app.logging_config import get_logger

logger = get_logger(__name__)


class IngestionRunner:
    """数据采集编排器（重构后）。

    编排 5 个独立 Worker 的生命周期，并负责 WS 事件分发。
    """

    def __init__(
        self,
        settings: Settings,
        deriv_repo: DerivativesRepo,
        ws_clients: Sequence[ExchangeWebSocketClient],
        rest_client: ExchangeRestClient,
        trade_buffer: TradeBufferWorker,
        ob_writer: OrderbookWriter,
        watchdog: RestWatchdog,
        retention: RetentionCleaner,
    ):
        self.settings = settings
        self._deriv_repo = deriv_repo
        self.ws_clients = list(ws_clients)
        self.rest_client = rest_client

        # Workers
        self._trade_buffer = trade_buffer
        self._ob_writer = ob_writer
        self._watchdog = watchdog
        self._retention = retention
        # 有后台循环的 Worker 列表（统一启停）
        self._loop_workers = [trade_buffer, watchdog, retention]

        # WS 消费任务
        self._ws_tasks: List[asyncio.Task[Any]] = []
        # 共享停止信号
        self._stopping = asyncio.Event()
        # WS 通道健康度：(symbol, kind) → monotonic 时间戳
        self._last_ws_event_at: Dict[Tuple[str, str], float] = {}
        self._last_ws_event_iso: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """启动所有 WS 连接和 Worker。"""
        if self._ws_tasks:
            return
        logger.info("启动数据采集：%d 个 WS 客户端", len(self.ws_clients))

        # 启动 WS 消费
        for client in self.ws_clients:
            self._ws_tasks.append(
                asyncio.create_task(self._run_ws(client), name=f"ws-{client.name}")
            )

        # 启动所有有后台循环的 Worker
        for w in self._loop_workers:
            w.start()

    async def stop(self) -> None:
        """停止所有 WS 连接和 Worker。"""
        if not self._ws_tasks and not any(w.is_running for w in self._loop_workers):
            return
        logger.info("正在停止数据采集任务")
        self._stopping.set()

        # 停止 WS 连接
        for client in self.ws_clients:
            stop = getattr(client, "stop", None)
            if callable(stop):
                try:
                    await stop()
                except Exception:  # noqa: BLE001
                    pass

        # 取消 WS 消费任务
        for task in self._ws_tasks:
            task.cancel()
        for task in self._ws_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._ws_tasks.clear()

        # 停止所有 Worker（反序）
        for w in reversed(self._loop_workers):
            await w.stop()

        logger.info("数据采集任务已停止")

    # ------------------------------------------------------------------
    # WS 消费
    # ------------------------------------------------------------------
    async def _run_ws(self, client: ExchangeWebSocketClient) -> None:
        """WS 消费协程：读取事件流并分发。"""
        try:
            async for event in client.stream():
                await self._dispatch(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("WebSocket 客户端 %s 异常崩溃", client.name)

    async def _dispatch(self, event: Dict[str, Any]) -> None:
        """将 WS 事件分发到对应处理器。"""
        etype = event.get("type")
        symbol = event.get("symbol")
        try:
            if etype == "trade":
                await self._trade_buffer.enqueue_trade(event)
                self._mark_ws_event(symbol, "trade")
            elif etype == "orderbook":
                # 即使被节流丢弃也标记"通道是活的"
                self._mark_ws_event(symbol, "orderbook")
                await self._ob_writer.handle_orderbook(event)
            elif etype == "funding_rate":
                await self._deriv_repo.insert_funding_rate(
                    exchange=event["exchange"],
                    symbol=event["symbol"],
                    ts=event["ts"],
                    funding_rate=event["funding_rate"],
                    next_funding_ts=event.get("next_funding_ts"),
                )
                self._mark_ws_event(symbol, "funding_rate")
            elif etype == "open_interest":
                await self._deriv_repo.insert_open_interest(
                    exchange=event["exchange"],
                    symbol=event["symbol"],
                    ts=event["ts"],
                    oi=event["oi"],
                    oi_ccy=event.get("oi_ccy"),
                )
                self._mark_ws_event(symbol, "open_interest")
            elif etype == "ticker":
                self._mark_ws_event(symbol, "ticker")
            elif etype == "liquidation":
                if symbol:
                    await self._trade_buffer.enqueue_liquidation(event)
                    self._mark_ws_event(symbol, "liquidation")
        except Exception:
            logger.exception("事件持久化失败 type=%s", etype)

    # ------------------------------------------------------------------
    # WS 健康度
    # ------------------------------------------------------------------
    def _mark_ws_event(self, symbol: Optional[str], kind: str) -> None:
        """刷新指定 (symbol, kind) 的最近 WS 推送时间。"""
        if not symbol:
            return
        key = (symbol, kind)
        self._last_ws_event_at[key] = time.monotonic()
        self._last_ws_event_iso[key] = datetime.now(timezone.utc).isoformat()

    def ws_health_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """导出 WS 通道健康度，供 /healthz 路由读取。"""
        now = time.monotonic()
        out: Dict[str, Dict[str, Any]] = {}
        for (symbol, kind), ts in self._last_ws_event_at.items():
            out.setdefault(symbol, {})[kind] = {
                "age_seconds": round(now - ts, 1),
                "last_event_at": self._last_ws_event_iso.get((symbol, kind)),
            }
        return out
