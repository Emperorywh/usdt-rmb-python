"""异步采集编排器。

每个进程下并发运行 3 个 worker：

1. WebSocket 消费者：每个 ``ExchangeWebSocketClient`` 一个任务，把
   ``trade`` / ``orderbook`` 事件落库。
2. 成交批量 flusher：把内存里的成交缓冲区每秒清空一次。
3. REST 轮询器：funding_rates 与 open_interest 的唯一写入入口，
   默认 60 秒一次。

订单簿写入按 symbol 通过 ``settings.orderbook_min_interval_seconds``
节流，避免 books5 推送把数据库打爆。

注意：原本还有一个写入 mock 链上指标的 onchain poller，已经下线，
等接入真实链上数据源后再启用。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

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
    """采集任务生命周期管理器。"""

    def __init__(
        self,
        settings: Settings,
        repos: Repositories,
        ws_clients: Sequence[ExchangeWebSocketClient],
        rest_client: ExchangeRestClient,
        onchain: Optional[OnchainProvider] = None,
        trade_flush_size: int = 50,
        trade_flush_interval: float = 1.0,
    ):
        """
        构造采集编排器
        ---------------------------------------------------------------
        参数：
            settings:             全局配置
            repos:                数据仓储集合
            ws_clients:           交易所 WebSocket 客户端列表
            rest_client:          交易所 REST 客户端（funding/OI 兜底）
            onchain:              链上数据 provider，**当前未启用**，
                                  等接入真实数据源（Glassnode/Nansen 等）后再
                                  在 :meth:`start` 中拉起对应轮询任务
            trade_flush_size:     成交缓冲区批量写入阈值
            trade_flush_interval: 成交缓冲区强制 flush 的最大间隔（秒）
        """
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
        # 订单簿节流用：记录每个 symbol 最近一次成功落库的单调时钟时间戳
        self._last_orderbook_write: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """
        启动所有采集协程
        ---------------------------------------------------------------
        说明：
            - 当前启动 ws 消费者 + trade flusher + REST 轮询三类。
            - onchain 轮询任务暂时不再启动，等真实链上数据源接入后再
              在这里追加 ``self._run_onchain_poller`` 任务。
        """
        if self._tasks:
            return
        logger.info(
            "Starting ingestion: %d ws-clients / rest poller (onchain poller disabled)",
            len(self.ws_clients),
        )
        for client in self.ws_clients:
            self._tasks.append(asyncio.create_task(self._run_ws(client), name=f"ws-{client.name}"))
        self._tasks.append(asyncio.create_task(self._run_trade_flusher(), name="trade-flusher"))
        self._tasks.append(asyncio.create_task(self._run_rest_poller(), name="rest-poller"))
        # 数据保留清理任务：retention_run_interval_seconds <= 0 时彻底关闭。
        # 不启动该任务时高频表会无限增长，仅在外部已有清理脚本时才允许关闭。
        if int(getattr(self.settings, "retention_run_interval_seconds", 0) or 0) > 0:
            self._tasks.append(
                asyncio.create_task(self._run_retention_cleaner(), name="retention-cleaner")
            )
        else:
            logger.warning(
                "Retention cleaner disabled (retention_run_interval_seconds<=0); "
                "high-frequency tables will grow unbounded"
            )

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
        """
        分发并持久化一个 WS 事件
        ---------------------------------------------------------------
        说明：
            - trade 事件进入异步缓冲，由 _run_trade_flusher 批量落库。
            - orderbook 事件按 symbol 做节流：距离上次写入不足
              ``orderbook_min_interval_seconds`` 时直接丢弃，避免高频
              快照打爆 DB。
            - funding_rate / open_interest 不再从 WS 写入（已退订）。
            - tickers 仅用作行情参考，不入库。
        """
        etype = event.get("type")
        try:
            if etype == "trade":
                await self._buffer_trade(event)
            elif etype == "orderbook":
                if not self._should_write_orderbook(event["symbol"]):
                    return
                await self.repos.insert_orderbook(
                    exchange=event["exchange"],
                    symbol=event["symbol"],
                    ts=event["ts"],
                    bids=event["bids"],
                    asks=event["asks"],
                )
        except Exception:
            logger.exception("Failed to persist event type=%s", etype)

    def _should_write_orderbook(self, symbol: str) -> bool:
        """
        判断当前 symbol 的订单簿是否到达可以落库的最小间隔
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码
        返回：
            True  - 距离上次写入已经超过阈值，允许落库；
            False - 太密集，应当丢弃这次快照。
        说明：
            使用 time.monotonic() 单调时钟，避免系统时间跳变带来误判。
        """
        min_interval = float(
            getattr(self.settings, "orderbook_min_interval_seconds", 0.0) or 0.0
        )
        if min_interval <= 0:
            return True
        now = time.monotonic()
        last = self._last_orderbook_write.get(symbol, 0.0)
        if now - last < min_interval:
            return False
        self._last_orderbook_write[symbol] = now
        return True

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
        """
        REST 轮询任务
        ---------------------------------------------------------------
        说明：
            funding_rates 和 open_interest 现在唯一的写入路径。
            默认 60 秒一次，靠表上的唯一约束 + ON CONFLICT DO NOTHING
            兜底，即使重启后短时间内 ts 撞库也不会产生脏数据。
        """
        interval = self.settings.rest_poll_interval_seconds
        try:
            while not self._stopping.is_set():
                for symbol in self.settings.symbols:
                    # funding / OI 轮询允许单次失败：OKX REST 客户端内部已经做了
                    # 指数退避重试，这里只剩“彻底失败”的情况，打印精简 warning
                    # 即可，不再刷满屏 traceback。
                    try:
                        fr = await self.rest_client.fetch_funding_rate(symbol)
                        await self.repos.insert_funding_rate(
                            exchange=fr["exchange"],
                            symbol=fr["symbol"],
                            ts=fr["ts"],
                            funding_rate=fr["funding_rate"],
                            next_funding_ts=fr.get("next_funding_ts"),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "REST funding poll failed for %s: %s: %s",
                            symbol,
                            exc.__class__.__name__,
                            exc,
                        )
                    try:
                        oi = await self.rest_client.fetch_open_interest(symbol)
                        await self.repos.insert_open_interest(
                            exchange=oi["exchange"],
                            symbol=oi["symbol"],
                            ts=oi["ts"],
                            oi=oi["oi"],
                            oi_ccy=oi.get("oi_ccy"),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "REST OI poll failed for %s: %s: %s",
                            symbol,
                            exc.__class__.__name__,
                            exc,
                        )
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    async def _run_retention_cleaner(self) -> None:
        """
        数据保留清理后台任务
        ---------------------------------------------------------------
        说明：
            - 周期性按 ``settings.retention_*_seconds`` 删除各表的旧数据，
              防止 trades / orderbook_snapshots 这种高频表把硬盘吃满。
            - 启动后先 sleep 一个完整周期再开始干活，避免冷启动瞬间和
              第一次因子计算抢资源。
            - 任何一张表清理失败只 warn，不影响其他表 / 不影响主流程。
            - 不显式 VACUUM：依赖 PG 的 autovacuum 渐进回收死元组，避免
              引入额外锁与 CPU 抖动；如需立即释放磁盘，运维侧可手动
              ``VACUUM (ANALYZE) trades;``。
            - 用 ``self._stopping.wait`` + ``wait_for`` 实现可被取消的 sleep，
              进程关闭时不会拖时间。
        """
        interval = int(self.settings.retention_run_interval_seconds)
        logger.info("Retention cleaner started (every %ds)", interval)
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
        """
        执行一次全量清理
        ---------------------------------------------------------------
        说明：
            按配置好的保留秒数计算 cutoff（UTC），分别清理三张表。
            ``retention_*_seconds <= 0`` 表示该表永不清理（跳过）。
        """
        now = datetime.now(timezone.utc)
        targets = (
            ("trades", int(self.settings.retention_trades_seconds),
             self.repos.delete_trades_older_than),
            ("orderbook_snapshots", int(self.settings.retention_orderbook_seconds),
             self.repos.delete_orderbook_older_than),
            ("signals", int(self.settings.retention_signals_seconds),
             self.repos.delete_signals_older_than),
        )
        for table, retain_seconds, deleter in targets:
            if retain_seconds <= 0:
                continue
            cutoff = now - timedelta(seconds=retain_seconds)
            try:
                deleted = await deleter(cutoff)
                if deleted:
                    logger.info(
                        "Retention: deleted %d rows from %s (older than %s)",
                        deleted,
                        table,
                        cutoff.isoformat(timespec="seconds"),
                    )
                else:
                    logger.debug(
                        "Retention: %s nothing to delete (cutoff=%s)",
                        table,
                        cutoff.isoformat(timespec="seconds"),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Retention cleanup failed for %s: %s: %s",
                    table,
                    exc.__class__.__name__,
                    exc,
                )

    async def _run_onchain_poller(self) -> None:
        """
        链上指标轮询任务（当前已停用）
        ---------------------------------------------------------------
        说明：
            - 之前每 60 秒写一条 mock 链上指标，纯随机数据，对策略没有
              任何信息量，因此 :meth:`start` 里已经不再拉起这个任务。
            - 保留方法体便于将来接入真实链上数据源（Glassnode / Nansen /
              Etherscan 等）后只需替换 ``self.onchain`` 实现并恢复 start
              中的 ``create_task`` 调用即可。
        """
        if self.onchain is None:
            logger.info("Onchain provider not configured, poller skipped")
            return
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
