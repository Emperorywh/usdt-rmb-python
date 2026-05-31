"""Trade 与 Liquidation 批量缓冲 Worker。

将内存中的 trade / liquidation 事件缓冲区按固定间隔批量落库，
避免高频 WS 推送逐条写库造成 IO 瓶颈。
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.data_storage.repos.trade_repo import TradeRepo
from app.data_storage.repos.derivatives_repo import DerivativesRepo
from app.logging_config import get_logger

logger = get_logger(__name__)


class TradeBufferWorker:
    """Trade + Liquidation 批量缓冲与 flush。

    职责：
    - 缓冲 WS 推送的 trade 和 liquidation 事件
    - 按固定间隔（默认 1s）批量写入 DB
    - 支持强制 flush（关闭时调用）
    """

    def __init__(
        self,
        trade_repo: TradeRepo,
        deriv_repo: DerivativesRepo,
        settings: Settings,
        flush_size: int = 50,
        stopping: Optional[asyncio.Event] = None,
    ):
        self._trade_repo = trade_repo
        self._deriv_repo = deriv_repo
        self._settings = settings
        self._flush_size = flush_size
        self._stopping = stopping or asyncio.Event()

        self._trade_buffer: List[Dict[str, Any]] = []
        self._trade_lock = asyncio.Lock()
        self._liquidation_buffer: List[Dict[str, Any]] = []
        self._liquidation_lock = asyncio.Lock()
        self._task: Optional[asyncio.Task[Any]] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动后台 flush 循环。"""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name="worker-TradeBufferWorker")

    async def stop(self) -> None:
        """停止后台任务并做最终 flush。"""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # 关闭时强制 flush 两个缓冲区
        await self.flush_trades(force=True)
        await self.flush_liquidations()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """交替 flush trade 和 liquidation 缓冲区。"""
        trade_interval = float(
            self._settings.ingestion.trade_flush_interval_seconds or 1.0
        )
        liq_interval = float(
            self._settings.ingestion.liquidation_flush_interval_seconds or 1.0
        )
        # 取较小间隔作为 tick，在一个 tick 里同时 flush 两者
        tick = min(trade_interval, liq_interval)
        try:
            while not self._stopping.is_set():
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=tick)
                except asyncio.TimeoutError:
                    pass
                await self.flush_trades()
                await self.flush_liquidations()
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    # Trade 缓冲
    # ------------------------------------------------------------------
    async def enqueue_trade(self, event: Dict[str, Any]) -> None:
        """缓冲一条 trade 事件；缓冲区满时立即 flush。"""
        async with self._trade_lock:
            self._trade_buffer.append(event)
            should_flush = len(self._trade_buffer) >= self._flush_size
        if should_flush:
            await self.flush_trades()

    async def flush_trades(self, force: bool = False) -> None:
        """将 trade 缓冲区批量写入 DB。"""
        async with self._trade_lock:
            if not self._trade_buffer:
                return
            batch = self._trade_buffer
            self._trade_buffer = []
        try:
            await self._trade_repo.insert_trades(batch)
            logger.debug("已批量落库 %d 条成交", len(batch))
        except Exception:
            logger.exception("成交批量落库失败（丢失 %d 行）", len(batch))

    # ------------------------------------------------------------------
    # Liquidation 缓冲
    # ------------------------------------------------------------------
    async def enqueue_liquidation(self, event: Dict[str, Any]) -> None:
        """缓冲一条 liquidation 事件。"""
        async with self._liquidation_lock:
            self._liquidation_buffer.append(event)

    async def flush_liquidations(self) -> None:
        """将 liquidation 缓冲区批量写入 DB。"""
        async with self._liquidation_lock:
            if not self._liquidation_buffer:
                return
            batch = self._liquidation_buffer
            self._liquidation_buffer = []
        try:
            await self._deriv_repo.insert_liquidations(batch)
            logger.debug("已批量落库 %d 条爆仓", len(batch))
        except Exception:
            logger.exception("爆仓批量落库失败（丢失 %d 行）", len(batch))
