"""信号生命周期跟踪任务（P2 升级核心组件）。

职责
====
- 在后台周期任务中扫描 ``signal_lifecycle`` 表里 status ∈ {pending, triggered}
  的行，依据最新 mark price 推进它们的状态机：

      pending  --(price ∈ entry_zone)-->  triggered
      triggered --(touch SL)-->  sl_hit
      triggered --(touch TP1)--> tp1_hit （继续跟踪 TP2）
      triggered --(touch TP2)--> tp2_hit （结算）
      triggered --(超过 expires_at)--> 强制按 mark exit + expired
      pending  --(超过 expires_at 仍未进区间)--> expired
      bias='neutral' 或入场区间 / SL / TP 缺失 --> 直接 expired（任务首轮扫到时）

- 持续滚动 ``max_favorable_pct`` / ``max_adverse_pct`` 两个统计字段，
  给后续 IC 校准 / 信号质量复盘提供数据。

mark price 来源
================
- 取 ``klines_1m`` 最新一根 close 作为标记价格——已经在采集层 1 秒级落库，
  最大年龄 = K 线封盘节奏（1m）+ 写库延迟（≤ 5s）。
- ``settings.lifecycle_mark_price_max_age_seconds``（默认 60s）：超过此值
  视为 K 线断流，**跳过本轮**所有 symbol，避免脏价误结算。
- 不引入 ticker WS 直推：避免再开一个数据通道；1m close 在多数行情下与
  最新 last_trade 的偏差 < 0.05%（信号粒度足够）。

性能预算
========
- 每 ``settings.lifecycle_tick_seconds``（默认 30s）跑一轮；
- 单 symbol 最多 ~几十条未结算信号（按 24h TTL × 15min/条节奏估算）；
- 每轮单次 DB roundtrip 拉所有未结算行 + N 次 update（≤ 几十次），
  整轮 < 200ms 可控。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.logging_config import get_logger

logger = get_logger(__name__)


class LifecycleTracker:
    """
    信号生命周期后台跟踪任务驱动器
    --------------------------------------------------------------
    用法：
        tracker = LifecycleTracker(settings, repos, symbols=["ETH-USDT-SWAP"])
        await tracker.start()  # FastAPI lifespan 启动
        await tracker.stop()
    """

    def __init__(
        self,
        settings: Settings,
        repos: Repositories,
        symbols: List[str],
    ):
        self.settings = settings
        self.repos = repos
        self.symbols = list(symbols)
        self._task: Optional[asyncio.Task[Any]] = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        """启动后台循环。"""
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._loop(), name="signal-lifecycle")

    async def stop(self) -> None:
        """优雅关停：通知循环退出 + 等任务结束。"""
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _loop(self) -> None:
        """后台循环主体。"""
        interval = max(1, int(self.settings.lifecycle_tick_seconds))
        logger.info(
            "信号生命周期跟踪任务已启动（symbols=%s，interval=%ds）",
            self.symbols,
            interval,
        )
        while not self._stopping.is_set():
            try:
                await self.tick_once()
            except Exception:
                logger.exception("信号生命周期任务一轮执行失败，下个周期继续")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def tick_once(self) -> Dict[str, int]:
        """
        手动 / 周期触发一轮跟踪（admin / 单元测试可直接调用）
        ---------------------------------------------------------------
        返回：
            {symbol: 本轮处理的未结算行数}
        说明：
            返回值用于日志 / 测试断言；任务循环里只关心是否抛异常。
        """
        result: Dict[str, int] = {}
        for symbol in self.symbols:
            mark = await self._fetch_mark_price(symbol)
            if mark is None:
                logger.debug(
                    "lifecycle: symbol=%s 标记价格不可用（K 线断流？），本轮跳过",
                    symbol,
                )
                result[symbol] = 0
                continue
            try:
                open_rows = await self.repos.fetch_open_signal_lifecycles(
                    symbol=symbol
                )
            except Exception:
                logger.warning(
                    "lifecycle: 拉取 %s 的未结算行失败，下轮重试", symbol, exc_info=True
                )
                result[symbol] = 0
                continue
            count = 0
            for row in open_rows:
                try:
                    await self._advance_one(row, mark_price=mark)
                    count += 1
                except Exception:
                    logger.warning(
                        "lifecycle: 推进 signal_id=%s 失败，跳过本条",
                        row.get("signal_id"),
                        exc_info=True,
                    )
            result[symbol] = count
        return result

    async def _fetch_mark_price(self, symbol: str) -> Optional[float]:
        """
        取最新 mark price：klines_1m 最新一根 close + 60s 陈旧度门槛
        """
        max_age = float(
            getattr(self.settings, "lifecycle_mark_price_max_age_seconds", 60.0)
        )
        try:
            row = await self.repos.fetch_latest_kline_close_within(
                timeframe="1m", symbol=symbol, max_age_seconds=max_age
            )
        except Exception:
            logger.warning(
                "lifecycle: 拉取 %s 标记价格异常，本轮跳过", symbol, exc_info=True
            )
            return None
        if row is None or row.get("close") is None:
            return None
        try:
            return float(row["close"])
        except (TypeError, ValueError):
            return None

    async def _advance_one(self, row: Dict[str, Any], mark_price: float) -> None:
        """
        推进单条 lifecycle 行的状态机
        ---------------------------------------------------------------
        参数：
            row        : fetch_open_signal_lifecycles 返回的一行
            mark_price : 当前标记价格（已通过陈旧度校验）
        """
        signal_id: int = int(row["signal_id"])
        bias: str = row["bias"]
        status: str = row["status"]
        now = datetime.now(timezone.utc)

        # 1) 首轮处理 neutral / 缺关键价位的旧信号 → 直接置 expired
        ez_low = _to_float(row.get("entry_zone_low"))
        ez_high = _to_float(row.get("entry_zone_high"))
        sl = _to_float(row.get("stop_loss"))
        tp_list = _coerce_tp_list(row.get("take_profit"))
        expires_at = row.get("expires_at")
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if bias == "neutral" or ez_low is None or ez_high is None or sl is None or len(tp_list) < 1:
            await self.repos.update_signal_lifecycle(
                signal_id=signal_id,
                status="expired",
                exit_at=now,
                exit_price=mark_price,
            )
            logger.info(
                "lifecycle: signal_id=%s 缺少结构化字段或 neutral，置 expired", signal_id
            )
            return

        # 2) 超时强制结算
        if expires_at is not None and now >= expires_at:
            triggered_price = _to_float(row.get("triggered_price"))
            pnl_pct: Optional[float] = None
            if status == "triggered" and triggered_price and triggered_price > 0:
                pnl_pct = _compute_pnl_pct(
                    bias=bias, entry=triggered_price, exit_=mark_price
                )
            new_max_fav, new_max_adv = self._update_extremes(
                row=row,
                bias=bias,
                triggered_price=triggered_price,
                mark_price=mark_price,
            )
            await self.repos.update_signal_lifecycle(
                signal_id=signal_id,
                status="expired",
                exit_at=now,
                exit_price=mark_price,
                pnl_pct=pnl_pct,
                max_favorable_pct=new_max_fav,
                max_adverse_pct=new_max_adv,
            )
            logger.info(
                "lifecycle: signal_id=%s 超过 expires_at，强制按 %.4f 结算（pnl_pct=%s）",
                signal_id,
                mark_price,
                f"{pnl_pct:+.4%}" if pnl_pct is not None else "N/A",
            )
            return

        # 3) 状态机推进
        if status == "pending":
            # 价格进入入场区间 → triggered
            if ez_low <= mark_price <= ez_high:
                await self.repos.update_signal_lifecycle(
                    signal_id=signal_id,
                    status="triggered",
                    triggered_at=now,
                    triggered_price=mark_price,
                    max_favorable_pct=0.0,
                    max_adverse_pct=0.0,
                )
                logger.info(
                    "lifecycle: signal_id=%s 进入入场区间 [%.4f, %.4f]，置 triggered@%.4f",
                    signal_id,
                    ez_low,
                    ez_high,
                    mark_price,
                )
            return

        # status == 'triggered' 之后再判断 SL / TP / 滚动统计
        triggered_price = _to_float(row.get("triggered_price")) or mark_price
        new_max_fav, new_max_adv = self._update_extremes(
            row=row,
            bias=bias,
            triggered_price=triggered_price,
            mark_price=mark_price,
        )

        # 3a) 触发 SL
        if (bias == "long" and mark_price <= sl) or (bias == "short" and mark_price >= sl):
            pnl_pct = _compute_pnl_pct(
                bias=bias, entry=triggered_price, exit_=mark_price
            )
            await self.repos.update_signal_lifecycle(
                signal_id=signal_id,
                status="sl_hit",
                exit_at=now,
                exit_price=mark_price,
                pnl_pct=pnl_pct,
                max_favorable_pct=new_max_fav,
                max_adverse_pct=new_max_adv,
            )
            logger.info(
                "lifecycle: signal_id=%s 触发止损 mark=%.4f sl=%.4f pnl=%s",
                signal_id, mark_price, sl,
                f"{pnl_pct:+.4%}" if pnl_pct is not None else "N/A",
            )
            return

        # 3b) tp1 / tp2 触达
        tp1 = tp_list[0] if len(tp_list) >= 1 else None
        tp2 = tp_list[1] if len(tp_list) >= 2 else None
        # tp2 优先（一旦 tp2 也触发即结算）
        if tp2 is not None and (
            (bias == "long" and mark_price >= tp2)
            or (bias == "short" and mark_price <= tp2)
        ):
            pnl_pct = _compute_pnl_pct(
                bias=bias, entry=triggered_price, exit_=mark_price
            )
            await self.repos.update_signal_lifecycle(
                signal_id=signal_id,
                status="tp2_hit",
                exit_at=now,
                exit_price=mark_price,
                pnl_pct=pnl_pct,
                max_favorable_pct=new_max_fav,
                max_adverse_pct=new_max_adv,
            )
            logger.info(
                "lifecycle: signal_id=%s 触达 TP2 mark=%.4f tp2=%.4f pnl=%s",
                signal_id, mark_price, tp2,
                f"{pnl_pct:+.4%}" if pnl_pct is not None else "N/A",
            )
            return

        if tp1 is not None and (
            (bias == "long" and mark_price >= tp1)
            or (bias == "short" and mark_price <= tp1)
        ):
            # tp1_hit 不退出，仍跟踪 tp2；如果之前已经标 tp1_hit 这里幂等。
            if status != "tp1_hit":
                await self.repos.update_signal_lifecycle(
                    signal_id=signal_id,
                    status="tp1_hit",
                    max_favorable_pct=new_max_fav,
                    max_adverse_pct=new_max_adv,
                )
                logger.info(
                    "lifecycle: signal_id=%s 触达 TP1 mark=%.4f tp1=%.4f（继续跟踪 TP2）",
                    signal_id, mark_price, tp1,
                )
            else:
                # 仅刷新极值
                await self.repos.update_signal_lifecycle(
                    signal_id=signal_id,
                    max_favorable_pct=new_max_fav,
                    max_adverse_pct=new_max_adv,
                )
            return

        # 3c) 既没触发 SL 也没触达 TP → 仅刷新极值
        await self.repos.update_signal_lifecycle(
            signal_id=signal_id,
            max_favorable_pct=new_max_fav,
            max_adverse_pct=new_max_adv,
        )

    @staticmethod
    def _update_extremes(
        row: Dict[str, Any],
        bias: str,
        triggered_price: Optional[float],
        mark_price: float,
    ) -> "tuple[Optional[float], Optional[float]]":
        """
        刷新 max_favorable_pct / max_adverse_pct
        ---------------------------------------------------------------
        说明：
            triggered_price 缺失时（pending 阶段就被超时调用）跳过统计，
            返回旧值即可。
        """
        if not triggered_price or triggered_price <= 0:
            return _to_float(row.get("max_favorable_pct")), _to_float(row.get("max_adverse_pct"))
        delta = (mark_price - triggered_price) / triggered_price
        if bias == "short":
            delta = -delta  # 翻转：做空时下跌为有利
        prev_max_fav = _to_float(row.get("max_favorable_pct")) or 0.0
        prev_max_adv = _to_float(row.get("max_adverse_pct")) or 0.0
        new_max_fav = max(prev_max_fav, delta)
        new_max_adv = min(prev_max_adv, delta)
        return new_max_fav, new_max_adv


# ----------------------------------------------------------------------
# 模块级辅助函数
# ----------------------------------------------------------------------
def compute_expires_at(
    now: Optional[datetime] = None, ttl_hours: int = 24
) -> datetime:
    """
    给一条新信号计算 expires_at（一般 = now + ttl_hours 小时）
    ---------------------------------------------------------------
    参数：
        now       : 基准时间；None 取当前 UTC
        ttl_hours : 默认 24h
    返回：
        UTC datetime
    """
    base = now or datetime.now(timezone.utc)
    return base + timedelta(hours=max(1, int(ttl_hours)))


def _to_float(v: Any) -> Optional[float]:
    """
    asyncpg 取出来的 NUMERIC 是 Decimal，统一成 float / None
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _coerce_tp_list(v: Any) -> List[float]:
    """
    把 take_profit JSONB 转成 list[float]（最多保留前 2 档）
    """
    if not v:
        return []
    if isinstance(v, list):
        out: List[float] = []
        for x in v:
            f = _to_float(x)
            if f is not None:
                out.append(f)
            if len(out) >= 2:
                break
        return out
    return []


def _compute_pnl_pct(bias: str, entry: float, exit_: float) -> Optional[float]:
    """
    PnL% 计算：long = (exit-entry)/entry；short = (entry-exit)/entry
    ---------------------------------------------------------------
    返回：
        百分比表示（0.0123 = +1.23%）；entry 非法时返回 None。
    """
    if not entry or entry <= 0:
        return None
    raw = (exit_ - entry) / entry
    if bias == "short":
        raw = -raw
    return round(raw, 6)
