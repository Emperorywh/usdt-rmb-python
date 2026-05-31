"""Orderbook 写入 + 指标计算 Worker。

处理 WS 推送的 orderbook 事件：
1. 按 symbol 节流写入 orderbook_snapshots（5s）
2. 按 symbol 节流写入 orderbook_metrics 时序指标（10s）

纯事件驱动，无后台循环。
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from app.config import Settings
from app.data_storage.orderbook_metrics import compute_orderbook_metric_row
from app.data_storage.repos.orderbook_repo import OrderbookRepo
from app.logging_config import get_logger

logger = get_logger(__name__)


class OrderbookWriter:
    """Orderbook 事件节流写入 + 指标计算。

    职责：
    - 按 symbol 节流写入 orderbook_snapshots
    - 按 symbol 节流计算并写入 orderbook_metrics
    - 无后台循环，由 Runner 的事件分发直接调用
    """

    def __init__(self, ob_repo: OrderbookRepo, settings: Settings):
        self._ob_repo = ob_repo
        self._settings = settings
        self._last_orderbook_write: Dict[str, float] = {}
        self._last_orderbook_metric_write: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 事件处理（由 Runner dispatch 调用）
    # ------------------------------------------------------------------
    async def handle_orderbook(self, event: Dict[str, Any]) -> None:
        """处理一个 orderbook 类型的 WS 事件。

        包含两层节流写入：
        1. orderbook_snapshots（5s 节流）—— 原始快照
        2. orderbook_metrics（10s 节流）—— 时序指标
        """
        symbol = event.get("symbol", "")

        # P0 路径：原始快照写入（5s 节流）
        if self._should_write_orderbook(symbol):
            await self._ob_repo.insert_orderbook(
                exchange=event["exchange"],
                symbol=symbol,
                ts=event["ts"],
                bids=event["bids"],
                asks=event["asks"],
            )

        # 独立时序指标（10s 节流，永远开启）
        if self._should_write_orderbook_metric(symbol):
            try:
                metric = compute_orderbook_metric_row(
                    snapshot=event,
                    wall_multiplier=float(
                        self._settings.factor.liquidity_wall_multiplier
                    ),
                    top_n=int(self._settings.ingestion.orderbook_depth),
                )
                await self._ob_repo.insert_orderbook_metric(
                    exchange=event["exchange"],
                    symbol=symbol,
                    ts=event["ts"],
                    metric=metric,
                )
            except Exception:
                logger.warning(
                    "orderbook_metrics 计算/落库失败 symbol=%s",
                    symbol,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # 节流判断
    # ------------------------------------------------------------------
    def _should_write_orderbook(self, symbol: str) -> bool:
        """判断 orderbook_snapshots 是否到达可写入间隔。"""
        min_interval = float(
            self._settings.ingestion.orderbook_min_interval_seconds or 0.0
        )
        if min_interval <= 0:
            return True
        now = time.monotonic()
        last = self._last_orderbook_write.get(symbol, 0.0)
        if now - last < min_interval:
            return False
        self._last_orderbook_write[symbol] = now
        return True

    def _should_write_orderbook_metric(self, symbol: str) -> bool:
        """判断 orderbook_metrics 是否到达可写入间隔。"""
        min_interval = float(
            self._settings.ingestion.orderbook_metrics_min_interval_seconds or 0.0
        )
        if min_interval <= 0:
            return True
        now = time.monotonic()
        last = self._last_orderbook_metric_write.get(symbol, 0.0)
        if now - last < min_interval:
            return False
        self._last_orderbook_metric_write[symbol] = now
        return True
