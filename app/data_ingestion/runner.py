"""异步采集编排器。

每个进程下并发运行下列 worker：

1. WebSocket 消费者：每个 ``ExchangeWebSocketClient`` 一个任务，把
   ``trade`` / ``orderbook`` / ``funding_rate`` / ``open_interest``
   事件落库。
2. 成交批量 flusher：把内存里的成交缓冲区每秒清空一次。
3. REST watchdog：funding-rate 与 open-interest 的兜底写入入口，
   仅在 WS 长时间没有推送对应频道时才发一次 REST，并且对每个 symbol
   并发发起；正常情况下完全不发请求，避免在 ``www.okx.com`` 网络抖动
   时把控制台刷成失败日志。
4. 数据保留清理任务（可选）。

订单簿写入按 symbol 通过 ``settings.orderbook_min_interval_seconds``
节流，避免 books5 推送把数据库打爆。

注意：原本还有一个写入 mock 链上指标的 onchain poller，已经下线，
等接入真实链上数据源后再启用。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from app.config import Settings
from app.data_ingestion.base import (
    ExchangeRestClient,
    ExchangeWebSocketClient,
    OnchainProvider,
)
from app.data_ingestion.okx_rest import CircuitOpenError
from app.data_storage.repositories import Repositories
from app.factor_engine.orderbook import compute_orderbook_metric_row
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
        # 爆仓推送：低频 + 阵发性，与 trades 类似走批量缓冲，
        # 由独立 flusher 1s 落库一次，避免单条频繁 IO。
        self._liquidation_buffer: List[Dict[str, Any]] = []
        self._liquidation_lock = asyncio.Lock()
        self._stopping = asyncio.Event()
        # 订单簿节流用：记录每个 symbol 最近一次成功落库的单调时钟时间戳
        self._last_orderbook_write: Dict[str, float] = {}
        # P1：orderbook_metrics 写入节流（独立于 orderbook_snapshots）
        # 用 monotonic 时钟，键是 symbol，值是上一次成功写入的时刻
        self._last_orderbook_metric_write: Dict[str, float] = {}
        # WS 通道健康度：记录每个 (symbol, kind) 最近一次成功收到推送的
        # 单调时钟时间戳；REST watchdog 用它判断是否需要兜底拉一次。
        # kind 取值：'trade' / 'orderbook' / 'funding_rate' / 'open_interest'
        self._last_ws_event_at: Dict[Tuple[str, str], float] = {}
        # /healthz 等外部观测用：以 ISO 时间字符串形式暴露最近一次推送时刻
        self._last_ws_event_iso: Dict[Tuple[str, str], str] = {}

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
            "启动数据采集：%d 个 WS 客户端 / REST 轮询器（链上轮询已停用）",
            len(self.ws_clients),
        )
        for client in self.ws_clients:
            self._tasks.append(asyncio.create_task(self._run_ws(client), name=f"ws-{client.name}"))
        self._tasks.append(asyncio.create_task(self._run_trade_flusher(), name="trade-flusher"))
        # 爆仓 flusher：单独的批量落库循环；与 trade flusher 解耦，
        # 即便 liquidation 写库失败也不影响主行情链路。
        self._tasks.append(
            asyncio.create_task(self._run_liquidation_flusher(), name="liquidation-flusher")
        )
        # REST 不再无脑 60s 轮询，改成 stale-watchdog：仅当 WS 长时间没有
        # 推送对应频道时才发一次 REST 兜底，降低对外网请求频率。
        self._tasks.append(asyncio.create_task(self._run_rest_watchdog(), name="rest-watchdog"))
        # 持仓比 REST 轮询任务（LLM-First 架构下永远启动）
        self._tasks.append(
            asyncio.create_task(
                self._run_position_ratios_poller(),
                name="position-ratios-poller",
            )
        )
        # 数据保留清理任务：retention_run_interval_seconds <= 0 时彻底关闭。
        # 不启动该任务时高频表会无限增长，仅在外部已有清理脚本时才允许关闭。
        if int(getattr(self.settings, "retention_run_interval_seconds", 0) or 0) > 0:
            self._tasks.append(
                asyncio.create_task(self._run_retention_cleaner(), name="retention-cleaner")
            )
        else:
            logger.warning(
                "数据保留清理任务已禁用（retention_run_interval_seconds<=0）；"
                "高频表将无限增长"
            )

    async def stop(self) -> None:
        if not self._tasks:
            return
        logger.info("正在停止数据采集任务")
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
        await self._flush_liquidations()
        logger.info("数据采集任务已停止")

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
            logger.exception("WebSocket 客户端 %s 异常崩溃", client.name)

    async def _dispatch(self, event: Dict[str, Any]) -> None:
        """
        分发并持久化一个 WS 事件
        ---------------------------------------------------------------
        说明：
            - trade 事件进入异步缓冲，由 _run_trade_flusher 批量落库。
            - orderbook 事件按 symbol 做节流：距离上次写入不足
              ``orderbook_min_interval_seconds`` 时直接丢弃，避免高频
              快照打爆 DB。
            - funding_rate / open_interest 走 WS 主路径直接落库（表上有
              ON CONFLICT DO NOTHING，与 watchdog 兜底写入幂等共存）。
            - tickers 仅用作行情参考，不入库。
            - 任何事件成功被处理后都会刷新 _last_ws_event_at[(symbol, kind)]，
              REST watchdog 据此判断是否需要兜底。
        """
        etype = event.get("type")
        symbol = event.get("symbol")
        try:
            if etype == "trade":
                await self._buffer_trade(event)
                self._mark_ws_event(symbol, "trade")
            elif etype == "orderbook":
                # 即便被节流丢弃，也要标记"通道是活的"，避免 watchdog 误判
                self._mark_ws_event(symbol, "orderbook")
                # P0 路径：原始 bids/asks 快照入 orderbook_snapshots（5s 节流）
                if self._should_write_orderbook(event["symbol"]):
                    await self.repos.insert_orderbook(
                        exchange=event["exchange"],
                        symbol=event["symbol"],
                        ts=event["ts"],
                        bids=event["bids"],
                        asks=event["asks"],
                    )
                # 独立的 orderbook_metrics 时序指标（10s 节流，永远开启）
                # 任一计算 / 落库失败都只记 warn，不影响主路径。
                if self._should_write_orderbook_metric(event["symbol"]):
                    try:
                        metric = compute_orderbook_metric_row(
                            snapshot=event,
                            wall_multiplier=float(
                                self.settings.liquidity_wall_multiplier
                            ),
                            top_n=int(self.settings.orderbook_depth),
                        )
                        await self.repos.insert_orderbook_metric(
                            exchange=event["exchange"],
                            symbol=event["symbol"],
                            ts=event["ts"],
                            metric=metric,
                        )
                    except Exception:
                        logger.warning(
                            "orderbook_metrics 计算/落库失败 symbol=%s",
                            event.get("symbol"),
                            exc_info=True,
                        )
            elif etype == "funding_rate":
                await self.repos.insert_funding_rate(
                    exchange=event["exchange"],
                    symbol=event["symbol"],
                    ts=event["ts"],
                    funding_rate=event["funding_rate"],
                    next_funding_ts=event.get("next_funding_ts"),
                )
                self._mark_ws_event(symbol, "funding_rate")
            elif etype == "open_interest":
                await self.repos.insert_open_interest(
                    exchange=event["exchange"],
                    symbol=event["symbol"],
                    ts=event["ts"],
                    oi=event["oi"],
                    oi_ccy=event.get("oi_ccy"),
                )
                self._mark_ws_event(symbol, "open_interest")
            elif etype == "ticker":
                # ticker 不入库，但记录心跳供 watchdog 参考
                self._mark_ws_event(symbol, "ticker")
            elif etype == "liquidation":
                # P0：爆仓事件先进缓冲，由 _run_liquidation_flusher 批量落库。
                # 推送本身已在 okx_ws._emit_liquidations 里按 self.symbols 过滤，
                # 这里只做一次防御性 None 检查。
                if symbol:
                    await self._buffer_liquidation(event)
                    self._mark_ws_event(symbol, "liquidation")
        except Exception:
            logger.exception("事件持久化失败 type=%s", etype)

    def _mark_ws_event(self, symbol: Optional[str], kind: str) -> None:
        """
        刷新指定 (symbol, kind) 的最近 WS 推送时间
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码；None 时直接忽略（异常推送）
            kind:   'trade' / 'orderbook' / 'funding_rate' /
                    'open_interest' / 'ticker'
        说明：
            REST watchdog 与 /healthz 都依赖这两个字典；用单调时钟做
            staleness 判断，用 UTC ISO 串供外部观测。
        """
        if not symbol:
            return
        key = (symbol, kind)
        self._last_ws_event_at[key] = time.monotonic()
        self._last_ws_event_iso[key] = datetime.now(timezone.utc).isoformat()

    def ws_health_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        导出 WS 通道健康度，供 /healthz 路由读取
        ---------------------------------------------------------------
        返回：
            {symbol: {kind: {age_seconds, last_event_at}}}
        说明：
            age_seconds 为相对调用时刻的单调时钟差（取整到 0.1s）。
        """
        now = time.monotonic()
        out: Dict[str, Dict[str, Any]] = {}
        for (symbol, kind), ts in self._last_ws_event_at.items():
            out.setdefault(symbol, {})[kind] = {
                "age_seconds": round(now - ts, 1),
                "last_event_at": self._last_ws_event_iso.get((symbol, kind)),
            }
        return out

    def _should_write_orderbook_metric(self, symbol: str) -> bool:
        """
        判断 orderbook_metrics 是否到达可以写入的最小间隔
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码
        返回：
            True - 距上次写入已经超过 P1 节流阈值；
            False - 太密集，丢弃本次。
        说明：
            - 与 _should_write_orderbook 完全独立：
              orderbook_snapshots 5s 节流是历史快照存储，
              orderbook_metrics 10s 节流是给因子层做时序回归用，
              两者任意一个落库失败都不影响另一个。
            - 单调时钟，避免系统时间跳变。
        """
        min_interval = float(
            getattr(self.settings, "orderbook_metrics_min_interval_seconds", 0.0) or 0.0
        )
        if min_interval <= 0:
            return True
        now = time.monotonic()
        last = self._last_orderbook_metric_write.get(symbol, 0.0)
        if now - last < min_interval:
            return False
        self._last_orderbook_metric_write[symbol] = now
        return True

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
            logger.debug("已批量落库 %d 条成交", len(batch))
        except Exception:
            logger.exception("成交批量落库失败（丢失 %d 行）", len(batch))

    # ------------------------------------------------------------------
    # 爆仓事件批量入库（P0）
    # ------------------------------------------------------------------
    async def _buffer_liquidation(self, event: Dict[str, Any]) -> None:
        """
        把一条爆仓事件追加到内存缓冲区
        ----------------------------------------------------------
        参数：
            event: okx_ws 里 type='liquidation' 的事件 dict
        说明：
            爆仓推送本身较稀疏，但出现"级联爆仓"时单秒可能涌入数十条；
            与 trades 一样走 1s 批量节奏，避免每条都打开一次连接。
        """
        async with self._liquidation_lock:
            self._liquidation_buffer.append(event)

    async def _run_liquidation_flusher(self) -> None:
        """
        爆仓批量 flusher 循环
        ----------------------------------------------------------
        说明：
            每 1 秒触发一次（与 trade flusher 同节奏），关闭事件触发时
            立即退出。失败只 warn 不抛，避免阻塞主链路。
        """
        try:
            while not self._stopping.is_set():
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                await self._flush_liquidations()
        except asyncio.CancelledError:
            raise

    async def _flush_liquidations(self) -> None:
        """
        将爆仓缓冲区批量写入 liquidations 表
        ----------------------------------------------------------
        说明：
            ON CONFLICT DO NOTHING 由 repos.insert_liquidations 内置，
            重连重复推送可被静默吞掉。
        """
        async with self._liquidation_lock:
            if not self._liquidation_buffer:
                return
            batch = self._liquidation_buffer
            self._liquidation_buffer = []
        try:
            await self.repos.insert_liquidations(batch)
            logger.debug("已批量落库 %d 条爆仓", len(batch))
        except Exception:
            logger.exception("爆仓批量落库失败（丢失 %d 行）", len(batch))

    # 各 WS 频道允许的最大静默时长（秒）；超过即触发一次 REST 兜底。
    # funding-rate 推送约 1 分钟一次，给 5 分钟容忍；
    # open-interest 推送 ~3s 一次，给 60s 容忍（避免 WS 抖动几秒就触发 REST）。
    _WS_STALE_FUNDING_SECONDS = 5 * 60.0
    _WS_STALE_OI_SECONDS = 60.0
    # watchdog 自身的轮询节奏（秒）：每次循环检查所有 (symbol, kind) 是否陈旧
    _WATCHDOG_TICK_SECONDS = 15.0
    # 启动后等多久才开始触发兜底，给 WS 一个建立连接 + 推第一条的时间窗口
    _WATCHDOG_GRACE_SECONDS = 30.0

    async def _run_rest_watchdog(self) -> None:
        """
        REST 兜底看门狗
        ---------------------------------------------------------------
        说明：
            - WS 是 funding-rate / open-interest 的主路径；本任务仅在
              对应频道长时间没有推送时才发一次 REST 兜底，平时不出网。
            - 每个 symbol 的 funding 与 OI 用 asyncio.gather 并发，
              互不阻塞；REST 客户端内部带熔断，连续失败会自动 fail-fast。
            - 写库走和 WS 路径完全相同的 repos 方法，依赖唯一约束 + ON
              CONFLICT DO NOTHING 保证幂等。
        """
        # 启动宽限：避免冷启动瞬间 WS 还没握手就被 watchdog 触发兜底
        try:
            await asyncio.wait_for(
                self._stopping.wait(), timeout=self._WATCHDOG_GRACE_SECONDS
            )
        except asyncio.TimeoutError:
            pass

        try:
            while not self._stopping.is_set():
                tasks: List[Awaitable[Any]] = []
                for symbol in self.settings.symbols:
                    if self._is_ws_stale(symbol, "funding_rate", self._WS_STALE_FUNDING_SECONDS):
                        tasks.append(
                            self._fallback_one(
                                symbol,
                                "funding_rate",
                                self.rest_client.fetch_funding_rate,
                                self._persist_funding_rate,
                            )
                        )
                    if self._is_ws_stale(symbol, "open_interest", self._WS_STALE_OI_SECONDS):
                        tasks.append(
                            self._fallback_one(
                                symbol,
                                "open_interest",
                                self.rest_client.fetch_open_interest,
                                self._persist_open_interest,
                            )
                        )
                if tasks:
                    # return_exceptions=True：单个兜底失败不影响其他 symbol
                    await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await asyncio.wait_for(
                        self._stopping.wait(), timeout=self._WATCHDOG_TICK_SECONDS
                    )
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    def _is_ws_stale(self, symbol: str, kind: str, threshold_seconds: float) -> bool:
        """
        判断 (symbol, kind) 通道是否陈旧到需要 REST 兜底
        ---------------------------------------------------------------
        参数：
            symbol:            合约代码
            kind:              'funding_rate' / 'open_interest'
            threshold_seconds: WS 静默多久即视为陈旧
        返回：
            True - 从未收到过 WS 推送 或 距上次推送超过 threshold；
            False - 最近收到过推送，跳过本次兜底。
        """
        last = self._last_ws_event_at.get((symbol, kind))
        if last is None:
            return True
        return (time.monotonic() - last) >= threshold_seconds

    async def _fallback_one(
        self,
        symbol: str,
        kind: str,
        fetcher: Callable[[str], Awaitable[Dict[str, Any]]],
        persister: Callable[[Dict[str, Any]], Awaitable[None]],
    ) -> None:
        """
        执行一次 REST 兜底拉取并落库
        ---------------------------------------------------------------
        参数：
            symbol:    合约代码
            kind:      'funding_rate' / 'open_interest'，仅用于日志
            fetcher:   实际发起 REST 请求的协程，例如
                       OKXRestClient.fetch_funding_rate
            persister: 拿到 dict 后写库的协程，封装在本类中以便复用
        说明：
            - CircuitOpenError 不打 warn，已在 REST 客户端内打过摘要日志。
            - 其他异常按 warn 输出（已经过 REST 客户端的失败摘要节流）。
        """
        try:
            payload = await fetcher(symbol)
            await persister(payload)
            logger.info(
                "REST 兜底成功：%s/%s（WS 已陈旧）",
                symbol,
                kind,
            )
        except CircuitOpenError:
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "REST 兜底失败 %s/%s：%s",
                symbol,
                kind,
                exc.__class__.__name__,
            )

    async def _persist_funding_rate(self, payload: Dict[str, Any]) -> None:
        """
        把 REST 返回的 funding-rate dict 写入 funding_rates 表
        ---------------------------------------------------------------
        """
        await self.repos.insert_funding_rate(
            exchange=payload["exchange"],
            symbol=payload["symbol"],
            ts=payload["ts"],
            funding_rate=payload["funding_rate"],
            next_funding_ts=payload.get("next_funding_ts"),
        )

    async def _persist_open_interest(self, payload: Dict[str, Any]) -> None:
        """
        把 REST 返回的 open-interest dict 写入 open_interest 表
        ---------------------------------------------------------------
        """
        await self.repos.insert_open_interest(
            exchange=payload["exchange"],
            symbol=payload["symbol"],
            ts=payload["ts"],
            oi=payload["oi"],
            oi_ccy=payload.get("oi_ccy"),
        )

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
            # P1：orderbook_metrics 与 orderbook_snapshots 共享保留时长，
            # 单行更小（< 100 字节），但样本更密集（10s 一行），同样需要清理。
            ("orderbook_metrics", int(self.settings.retention_orderbook_seconds),
             self.repos.delete_orderbook_metrics_older_than),
        )
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

    # ------------------------------------------------------------------
    # P1：持仓比 REST 轮询
    # ------------------------------------------------------------------
    async def _run_position_ratios_poller(self) -> None:
        """
        OKX rubik 持仓比轮询任务
        ---------------------------------------------------------------
        说明：
            - 每 ``settings.position_ratios_poll_interval_seconds`` 秒为
              每个 symbol 拉取 3 类持仓比并落库（散户币种 / 散户合约 / 精英持仓）。
            - 全部走 ExchangeRestClient 自带的熔断 + 退避；连续失败到阈值
              会被 circuit breaker 自动停一阵，本任务无须额外退避。
            - 任何一次失败只 warn 不抛，不影响下一轮拉取与主行情链路。
            - 启动后先等一个 grace 周期再开跑，避免冷启动瞬间 REST 抖动。
        """
        rest = self.rest_client
        # 用 hasattr 兜底兼容旧版 REST 客户端（没接 P1 接口时跳过）
        fetchers: List[Tuple[str, Callable[[str], Awaitable[List[Dict[str, Any]]]]]] = []
        for fname, label in (
            ("fetch_long_short_account_ratio", "account"),
            ("fetch_long_short_account_ratio_contract", "account_contract"),
            ("fetch_top_trader_position_ratio", "top_trader_position"),
        ):
            fn = getattr(rest, fname, None)
            if callable(fn):
                fetchers.append((label, fn))
        if not fetchers:
            logger.warning(
                "REST 客户端不支持 P1 持仓比接口，position-ratios-poller 退出"
            )
            return

        interval = max(60, int(self.settings.position_ratios_poll_interval_seconds))
        period = str(getattr(self.settings, "position_ratios_period", "5m"))
        logger.info(
            "持仓比轮询启动：每 %ds，period=%s，symbols=%s",
            interval,
            period,
            list(self.settings.symbols),
        )
        # 初始 grace：与 REST watchdog 一样给 30s 让连接池起来
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            pass

        try:
            while not self._stopping.is_set():
                for symbol in self.settings.symbols:
                    for label, fn in fetchers:
                        try:
                            rows = await fn(symbol, period=period)
                        except CircuitOpenError:
                            continue
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "持仓比拉取失败 %s/%s：%s",
                                symbol,
                                label,
                                exc.__class__.__name__,
                            )
                            continue
                        if rows:
                            try:
                                await self.repos.insert_position_ratios(rows)
                                logger.debug(
                                    "持仓比已落库 %s/%s：%d 行",
                                    symbol,
                                    label,
                                    len(rows),
                                )
                            except Exception:
                                logger.warning(
                                    "持仓比入库失败 %s/%s",
                                    symbol,
                                    label,
                                    exc_info=True,
                                )
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

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
            logger.info("未配置链上数据源，跳过轮询")
            return
        try:
            while not self._stopping.is_set():
                try:
                    metrics = await self.onchain.fetch_metrics()
                    await self.repos.insert_onchain(metrics)
                except Exception:
                    logger.exception("链上数据轮询失败")
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=60)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
