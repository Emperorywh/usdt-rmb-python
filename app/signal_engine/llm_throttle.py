"""LLM 调用节流管理。

职责：
- per-symbol 并发锁（防同进程竞态）
- DB 节流：查 signals 表判断是否在窗口内
- 缓存命中时从 DB 行重建完整 TradingSignal
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.data_storage.repos.signal_repo import SignalRepo
from app.logging_config import get_logger
from app.signal_engine.schemas import LLMAnalysisResult, TradingSignal
from app.utils import safe_float

logger = get_logger(__name__)


class LLMThrottleManager:
    """LLM 调用节流管理器。

    节流策略（固定窗口）：
    - "上一次判断时间"取自 signals 表里该 symbol 最后一条记录的 ts。
    - 在 min_interval 窗口内的请求会从 signals 表读出最近一条 LLM 判断，
      重建成 LLMAnalysisResult 返回（from_cache=True）。
    - 通过 per-symbol asyncio.Lock 防止同进程内并发调用。
    """

    def __init__(self, signal_repo: SignalRepo, settings: Settings):
        self._signal_repo = signal_repo
        self._settings = settings
        self._locks: Dict[str, asyncio.Lock] = {}

    @property
    def min_interval(self) -> int:
        """LLM 调用最小间隔（秒），从 settings 读取。"""
        return max(0, int(self._settings.llm.llm_min_interval_seconds))

    def get_lock(self, symbol: str) -> asyncio.Lock:
        """获取或创建指定 symbol 对应的并发锁。"""
        lock = self._locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[symbol] = lock
        return lock

    async def check_throttle(
        self,
        symbol: str,
        min_interval: Optional[int] = None,
    ) -> Optional[LLMAnalysisResult]:
        """检查节流。

        返回：
            None - 未命中缓存，应发起 LLM 调用
            LLMAnalysisResult(from_cache=True) - 命中缓存，直接复用

        步骤：
            1. 快速路径：锁外查 DB
            2. 慢速路径：拿锁后 double-checked locking 再查一次
            3. 命中时从 DB 行重建 TradingSignal
        """
        effective_interval = (
            int(min_interval) if min_interval is not None else self.min_interval
        )

        # 快速路径：锁外查一次
        cached = await self._load_recent_judgment(symbol, effective_interval)
        if cached is not None:
            return cached

        # 慢速路径：拿锁后再查一次（double-checked locking）
        async with self.get_lock(symbol):
            return await self._load_recent_judgment(symbol, effective_interval)

    async def _load_recent_judgment(
        self,
        symbol: str,
        effective_interval: int,
    ) -> Optional[LLMAnalysisResult]:
        """若 symbol 在节流窗口内已有 LLM 判断，重建成 LLMAnalysisResult 返回。"""
        if effective_interval <= 0:
            return None
        try:
            row = await self._signal_repo.fetch_latest_signal_judgment(symbol)
        except Exception:
            logger.warning(
                "查询 %s 最近一条信号时间戳失败；按未命中缓存处理",
                symbol,
                exc_info=True,
            )
            return None
        if row is None:
            return None

        last_ts: Optional[datetime] = row.get("ts")
        if last_ts is None:
            return None
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds()
        if elapsed >= effective_interval:
            return None

        cached_plan_kwargs = self.collect_cached_plan_kwargs(row)
        try:
            cached_signal = TradingSignal(
                bias=row.get("bias") or "neutral",
                confidence=float(row["confidence"]),
                reason=row.get("reason") or "",
                risk=row.get("risk") or "",
                suggestion=row.get("suggestion") or "",
                **cached_plan_kwargs,
            )
        except Exception:
            logger.warning(
                "无法从 signals 表行重建 %s 的 TradingSignal（多半是历史脏行"
                " / 结构化字段缺失）；将按未命中缓存重新调用 LLM",
                symbol,
                exc_info=True,
            )
            return None

        remaining = effective_interval - elapsed
        logger.debug(
            "LLM 数据库节流命中 %s（已过 %.0fs，下次调用还需 %.0fs，min_interval=%ds）",
            symbol,
            elapsed,
            remaining,
            effective_interval,
        )
        return LLMAnalysisResult(
            signal=cached_signal,
            reasoning_content=row.get("reasoning_content"),
            from_cache=True,
        )

    @staticmethod
    def collect_cached_plan_kwargs(row: Dict[str, Any]) -> Dict[str, Any]:
        """把 signals 表行的结构化交易计划列规整成 TradingSignal 构造 kwargs。"""
        kwargs: Dict[str, Any] = {}

        ez = row.get("entry_zone")
        if isinstance(ez, (list, tuple)) and len(ez) == 2:
            ez_low = safe_float(ez[0])
            ez_high = safe_float(ez[1])
            if ez_low is not None and ez_high is not None:
                kwargs["entry_zone"] = (ez_low, ez_high)

        sl = safe_float(row.get("stop_loss"))
        if sl is not None:
            kwargs["stop_loss"] = sl

        tp_raw = row.get("take_profit")
        if isinstance(tp_raw, (list, tuple)) and tp_raw:
            tp_list: List[float] = []
            for t in tp_raw:
                tv = safe_float(t)
                if tv is not None:
                    tp_list.append(tv)
            if tp_list:
                kwargs["take_profit"] = tp_list

        rr = safe_float(row.get("risk_reward_ratio"))
        if rr is not None:
            kwargs["risk_reward_ratio"] = rr

        psp = safe_float(row.get("position_size_pct"))
        if psp is not None:
            kwargs["position_size_pct"] = psp

        tfa = row.get("timeframe_alignment")
        if isinstance(tfa, dict) and tfa:
            kwargs["timeframe_alignment"] = {
                str(k): str(v) for k, v in tfa.items() if isinstance(v, str)
            }

        ic = row.get("invalidation_conditions")
        if isinstance(ic, list) and ic:
            kwargs["invalidation_conditions"] = [
                str(x) for x in ic if isinstance(x, str)
            ]

        return kwargs
