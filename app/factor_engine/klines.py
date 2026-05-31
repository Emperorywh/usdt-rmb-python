"""多周期 K 线增量聚合器。

职责
=====
- 从 ``trades`` 表里读取最新一根"未封盘 bar"窗口内的成交，按周期维度
  汇总成 OHLC + buy/sell volume + cvd_close + trade_count，并以
  ``INSERT ... ON CONFLICT (exchange, symbol, ts) DO UPDATE``
  的方式持续刷新到对应的 ``klines_<tf>`` 表。
- 跨周期边界后，把上一根 bar ``closed`` 标记为 TRUE 并新建下一根
  （cvd_close 需要"前一根的 cvd_close + 本根 delta"作为新 bar 起点）。

设计取舍
========
- 不用 PG 物化视图：物化视图无法做"持续滚动 / 保留未封盘 bar"。
- 不用 LISTEN/NOTIFY：trades 表写入频率高，触发器会成为 PG 瓶颈。
- 改用应用层定时增量：每个周期一个独立 asyncio task，频率与该周期
  的最小有意义粒度匹配（1m/5m=1s，15m/1h=10s，4h/1d=60s）。
- 单次聚合保持 < 100ms：聚合工作完全交给 PG，
  ``aggregate_trades_in_window`` 走 idx_trades_symbol_ts 区间扫描 +
  两次 LIMIT 1 索引点查（open/close），单次只回 1 行；行数与 bar 长度、
  TPS 解耦，4h 高 TPS 合约也不会撞上 asyncpg command_timeout。
- 性能：网络/反序列化成本固定（1 行），CPU 端只做一次加法（cvd_close
  = cvd_start + cvd_delta），不再有按成交逐笔的 Python 循环。

线程模型
========
- 每个 (symbol, timeframe) 起一个独立 asyncio task；同 symbol 的不同
  周期间不共享状态，因此并发安全。
- 同一 (symbol, timeframe) 的连续两次 tick 之间用单条协程串行执行，
  不会出现 race。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.logging_config import get_logger

logger = get_logger(__name__)


# 周期长度（秒）。1d 不按 86400 秒按"自然日"严格对齐到 UTC 00:00。
TIMEFRAME_SECONDS: Dict[str, int] = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}


def floor_to_timeframe(ts: datetime, timeframe: str) -> datetime:
    """
    把任意 UTC 时间向下对齐到给定周期的边界
    -----------------------------------------------------------------
    参数：
        ts:        含时区的 UTC datetime
        timeframe: '1m' / '5m' / '15m' / '1h' / '4h' / '1d'
    返回：
        对齐到周期开始的 UTC datetime（依旧带 tzinfo）。
    说明：
        - 对于 1m / 5m / 15m / 1h / 4h，统一按 epoch 秒整除取地板，
          这样 5m bar 永远是 :00 / :05 / :10... 切边，与交易所 K 线
          对齐。
        - 对于 1d，按自然日对齐到 UTC 00:00。
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if timeframe == "1d":
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    sec = TIMEFRAME_SECONDS[timeframe]
    epoch = int(ts.timestamp())
    aligned = (epoch // sec) * sec
    return datetime.fromtimestamp(aligned, tz=timezone.utc)


class KlineAggregator:
    """
    多周期 K 线增量聚合器
    -----------------------------------------------------------------
    用法：
        agg = KlineAggregator(repos=..., settings=..., exchange='okx')
        await agg.start(symbols=['ETH-USDT-SWAP'])
        ...
        await agg.stop()
    说明：
        启动后会为每个 (symbol, timeframe) 创建一个后台协程，按
        ``settings.kline_tick_seconds_<tf>`` 的节奏执行 ``aggregate_incremental``。
        关闭时优雅取消所有协程。
    """

    def __init__(
        self,
        repos: Repositories,
        settings: Settings,
        exchange: str = "okx",
    ):
        """
        构造聚合器
        -------------------------------------------------------------
        参数：
            repos:    数据仓储集合
            settings: 全局配置
            exchange: 当前 OKX 单交易所，预留多交易所扩展
        """
        self.repos = repos
        self.settings = settings
        self.exchange = exchange
        self._tasks: List[asyncio.Task[Any]] = []
        self._stopping = asyncio.Event()
        # 内存里维护"上一根 bar 的 cvd_close"，用于跨边界时给新 bar
        # 提供 cvd 起点；进程重启后从 DB 读最近一根回填。
        self._last_cvd_close: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self, symbols: List[str]) -> None:
        """
        为每个 (symbol, timeframe) 启动一个后台增量聚合协程
        -------------------------------------------------------------
        参数：
            symbols: 需要聚合的合约列表
        说明：
            周期与 tick 节奏由 settings.kline_tick_seconds_<tf> 控制。
        """
        if self._tasks:
            return
        tick_map: Dict[str, float] = {
            "1m": float(self.settings.factor.kline_tick_seconds_1m),
            "5m": float(self.settings.factor.kline_tick_seconds_5m),
            "15m": float(self.settings.factor.kline_tick_seconds_15m),
            "1h": float(self.settings.factor.kline_tick_seconds_1h),
            "4h": float(self.settings.factor.kline_tick_seconds_4h),
            "1d": float(self.settings.factor.kline_tick_seconds_1d),
        }
        for symbol in symbols:
            await self._warmup_last_cvd(symbol)
            for tf, tick in tick_map.items():
                self._tasks.append(
                    asyncio.create_task(
                        self._run_loop(symbol, tf, tick),
                        name=f"kline-{symbol}-{tf}",
                    )
                )
        logger.info(
            "K 线增量聚合器已启动：%d 个 (symbol×timeframe) 协程",
            len(self._tasks),
        )

    async def stop(self) -> None:
        """
        取消所有聚合协程并等待退出
        """
        if not self._tasks:
            return
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

    async def _warmup_last_cvd(self, symbol: str) -> None:
        """
        进程启动时，从 DB 把每个周期的"最近一根 bar 的 cvd_close"读回内存
        -------------------------------------------------------------
        参数：
            symbol: 合约代码
        说明：
            未读到（首次部署 / 表为空）时把 cvd 起点设为 0；后续的
            aggregate_incremental 会以此为基线累加 delta。
        """
        for tf in TIMEFRAME_SECONDS.keys():
            try:
                latest = await self.repos.fetch_latest_kline(tf, symbol)
            except Exception:
                latest = None
            cvd_prev = 0.0
            if latest and latest.get("cvd_close") is not None:
                try:
                    cvd_prev = float(latest["cvd_close"])
                except (TypeError, ValueError):
                    cvd_prev = 0.0
            self._last_cvd_close[(symbol, tf)] = cvd_prev

    async def _run_loop(self, symbol: str, timeframe: str, tick: float) -> None:
        """
        单个 (symbol, timeframe) 的增量聚合循环
        -------------------------------------------------------------
        参数：
            symbol:    合约代码
            timeframe: 周期标签
            tick:      每隔多少秒触发一次增量聚合
        说明：
            - 单次执行时间在 100ms 量级，远小于 tick；不会堆积。
            - 任何聚合异常仅 warn，不影响主链路；下个 tick 会重试。
        """
        try:
            while not self._stopping.is_set():
                try:
                    await self.aggregate_incremental(symbol, timeframe)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "K 线增量聚合失败 %s/%s：%s",
                        symbol,
                        timeframe,
                        exc.__class__.__name__,
                    )
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=tick)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    # 核心增量聚合
    # ------------------------------------------------------------------
    async def aggregate_incremental(self, symbol: str, timeframe: str) -> None:
        """
        对单个 (symbol, timeframe) 执行一次增量聚合
        -------------------------------------------------------------
        参数：
            symbol:    合约代码
            timeframe: 周期标签
        逻辑：
            1) 当前时间向下对齐得到"当前未封盘 bar"的开始时间 cur_ts；
            2) 让 PG 在服务端对 [cur_ts, cur_ts + tf_seconds) 内的 trades
               做 OHLC + 量能 + cvd_delta 聚合，单次只回 1 行；
            3) 用 _last_cvd_close[(symbol, tf)] 作为本根 bar 的 cvd 起点，
               叠加 cvd_delta 得到 cvd_close；
            4) UPSERT 到 klines_<tf>，closed=FALSE；
            5) 若 DB 里已经存在比 cur_ts 更早且 closed=FALSE 的 bar，
               说明跨过了 bar 边界，把它们补充封盘（closed=TRUE）。
        关键边界处理：
            - 进程刚启动时 _last_cvd_close 已由 _warmup_last_cvd 回填；
            - 跨周期边界时，前一根 bar 的最终 cvd_close 通过 fetch_latest_kline
              重新读取，避免内存里残留旧值；之后再以该值为新 bar 的 cvd 起点。
        性能：
            - 老实现："读全部成交回 Python 再 for-loop 累加"，行数随 bar 长度
              和 TPS 线性放大，1h/4h 高 TPS 合约下会触发 asyncpg
              command_timeout（30s）。
            - 新实现：单次聚合 SQL，固定回 1 行；走 idx_trades_symbol_ts
              区间扫描 + 两次 LIMIT 1 索引点查，远低于 100ms。
            - 数值结果与老实现等价：相同的 max/min/sum/first/last 语义，
              且全程走 NUMERIC，比原先 float() 还少一次精度损失。
        """
        if timeframe not in TIMEFRAME_SECONDS:
            return

        now = datetime.now(timezone.utc)
        cur_ts = floor_to_timeframe(now, timeframe)
        tf_seconds = TIMEFRAME_SECONDS[timeframe]
        next_ts = cur_ts + timedelta(seconds=tf_seconds)

        # ---- 跨边界检测：把所有在 cur_ts 之前但 closed=FALSE 的 bar 收尾 ----
        await self._close_stale_bars(symbol, timeframe, cur_ts)

        # ---- 在 PG 端聚合当前 bar 范围内的所有成交（只回 1 行）----
        agg = await self.repos.aggregate_trades_in_window(
            symbol=symbol,
            start=cur_ts,
            end=next_ts,
        )

        # 当前 bar 内还没有任何成交：不创建空 bar，避免大量空 OHLC 行；
        # 等出现第一笔成交时再 INSERT。这样 fetch_recent_klines 的"最近 N
        # 根"语义保持紧凑。
        if agg is None:
            return

        # ---- 拿到本根 bar 的 cvd 起点，叠加本窗口的 delta ----
        cvd_start = self._last_cvd_close.get((symbol, timeframe), 0.0)
        try:
            cvd_delta = float(agg["cvd_delta"]) if agg["cvd_delta"] is not None else 0.0
        except (TypeError, ValueError):
            cvd_delta = 0.0
        cvd_close = float(cvd_start) + cvd_delta

        # ---- 落库（upsert_kline 会自己 _to_dec，原值原样传即可）----
        await self.repos.upsert_kline(
            timeframe=timeframe,
            exchange=self.exchange,
            symbol=symbol,
            ts=cur_ts,
            ohlc={
                "open": agg["open"],
                "high": agg["high"],
                "low": agg["low"],
                "close": agg["close"],
                "volume": agg["volume"],
                "buy_volume": agg["buy_volume"],
                "sell_volume": agg["sell_volume"],
                "cvd_close": cvd_close,
                "trade_count": int(agg["trade_count"] or 0),
            },
            closed=False,
        )

    async def _close_stale_bars(
        self, symbol: str, timeframe: str, cur_ts: datetime
    ) -> None:
        """
        把已经跨过周期边界、但仍标记为 closed=FALSE 的旧 bar 封盘
        -------------------------------------------------------------
        参数：
            symbol:    合约代码
            timeframe: 周期标签
            cur_ts:    当前未封盘 bar 的开始时间
        说明：
            - 简化实现：只看最新一条；如果它的 ts < cur_ts 且 closed=FALSE，
              就把它标 closed=TRUE，并把它的 cvd_close 同步到内存里给新 bar
              做起点。
            - 极端场景（程序停机超过一整根 bar）下，更早的 bar 仍是 closed=FALSE，
              需要后续手工或运维侧补封盘；本函数不在循环内做"无界 backfill"，
              避免掩盖故障。
        """
        latest = await self.repos.fetch_latest_kline(timeframe, symbol)
        if not latest:
            return
        latest_ts: datetime = latest["ts"]
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.replace(tzinfo=timezone.utc)
        if latest_ts >= cur_ts:
            # 还在当前 bar 上，无需封盘
            return
        if bool(latest.get("closed")):
            # 已经封盘，但内存可能丢；同步一下 cvd 起点
            try:
                self._last_cvd_close[(symbol, timeframe)] = float(
                    latest.get("cvd_close") or 0.0
                )
            except (TypeError, ValueError):
                pass
            return

        # 把上一根 bar 标记为封盘
        await self.repos.upsert_kline(
            timeframe=timeframe,
            exchange=self.exchange,
            symbol=symbol,
            ts=latest_ts,
            ohlc={
                "open": latest.get("open"),
                "high": latest.get("high"),
                "low": latest.get("low"),
                "close": latest.get("close"),
                "volume": latest.get("volume"),
                "buy_volume": latest.get("buy_volume"),
                "sell_volume": latest.get("sell_volume"),
                "cvd_close": latest.get("cvd_close"),
                "trade_count": latest.get("trade_count") or 0,
            },
            closed=True,
        )
        try:
            self._last_cvd_close[(symbol, timeframe)] = float(
                latest.get("cvd_close") or 0.0
            )
        except (TypeError, ValueError):
            pass
