"""数据保留清理 Worker。

周期性清理各高频表的历史数据，防止磁盘被撑满。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

from app.config import Settings
from app.data_storage.repos.trade_repo import TradeRepo
from app.data_storage.repos.orderbook_repo import OrderbookRepo
from app.data_storage.repos.signal_repo import SignalRepo
from app.data_storage.repos.derivatives_repo import DerivativesRepo
from app.logging_config import get_logger

logger = get_logger(__name__)


class RetentionCleaner:
    """数据保留清理 Worker。

    职责：
    - 按 retention_*_seconds 配置清理各表过期数据
    - 支持 trades / orderbook_snapshots / orderbook_metrics / signals /
      funding_rates / open_interest
    - 单张表清理失败只 warn，不影响其他表
    """

    def __init__(
        self,
        trade_repo: TradeRepo,
        ob_repo: OrderbookRepo,
        signal_repo: SignalRepo,
        deriv_repo: DerivativesRepo,
        settings: Settings,
        stopping: Optional[asyncio.Event] = None,
    ):
        self._trade_repo = trade_repo
        self._ob_repo = ob_repo
        self._signal_repo = signal_repo
        self._deriv_repo = deriv_repo
        self._settings = settings
        self._stopping = stopping or asyncio.Event()
        self._task: Optional[asyncio.Task[Any]] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self.run(), name="worker-RetentionCleaner"
            )

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    async def run(self) -> None:
        interval = int(self._settings.retention.retention_run_interval_seconds)
        if interval <= 0:
            logger.warning(
                "数据保留清理任务已禁用（retention_run_interval_seconds<=0）；"
                "高频表将无限增长"
            )
            return

        logger.info("数据保留清理任务已启动（每 %ds 一次）", interval)
        try:
            # 启动后先等一个周期再开始清理
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

            while not self._stopping.is_set():
                await self._cleanup_once()
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    async def _cleanup_once(self) -> None:
        """执行一次全量清理。"""
        now = datetime.now(timezone.utc)
        targets: List[Tuple[str, int, Callable[..., Awaitable[int]]]] = [
            ("trades", int(self._settings.retention.retention_trades_seconds),
             self._trade_repo.delete_trades_older_than),
            ("orderbook_snapshots", int(self._settings.retention.retention_orderbook_seconds),
             self._ob_repo.delete_orderbook_older_than),
            ("signals", int(self._settings.retention.retention_signals_seconds),
             self._signal_repo.delete_signals_older_than),
            ("orderbook_metrics", int(self._settings.retention.retention_orderbook_seconds),
             self._ob_repo.delete_orderbook_metrics_older_than),
            ("funding_rates", int(self._settings.retention.retention_funding_seconds),
             self._deriv_repo.delete_funding_older_than),
            ("open_interest", int(self._settings.retention.retention_oi_seconds),
             self._deriv_repo.delete_oi_older_than),
        ]
        for table, retain_seconds, deleter in targets:
            if retain_seconds <= 0:
                continue
            cutoff = now - timedelta(seconds=retain_seconds)
            try:
                deleted = await deleter(cutoff)
                if deleted:
                    logger.info(
                        "数据保留：已从 %s 删除 %d 行（早于 %s）",
                        table,
                        deleted,
                        cutoff.isoformat(timespec="seconds"),
                    )
                else:
                    logger.debug(
                        "数据保留：%s 无需清理（截止时间=%s）",
                        table,
                        cutoff.isoformat(timespec="seconds"),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "数据保留清理失败 %s：%s：%s",
                    table,
                    exc.__class__.__name__,
                    exc,
                )
