"""因子聚合器：多周期矩阵 + 共振 + 爆仓滚动窗口（P0 升级）。

结构概览
=========
新版输出 dict 结构（``enable_mtf_factors=True`` 时）：

```
{
  "symbol": "ETH-USDT-SWAP",
  "computed_at": "...",
  "by_timeframe": {
     "5m":  {"capital_flow": {...}, "orderbook": {...},
             "derivatives": {...}, "market_structure": {...}},
     "15m": {...},
     "1h":  {...},
     "4h":  {...},
     "1d":  {...}
  },
  "mtf_alignment": {
     "trend_votes":     {"long": 3, "short": 0, "neutral": 2},
     "alignment_score": 0.6,
     "dominant_bias":   "long"
  },
  "liquidations": {
     "long_5m": 1.2, "short_5m": 0.0, ...,
     "imbalance_15m": 0.3,
     "cascade_signal": false
  }
}
```

兼容性
======
- ``enable_mtf_factors=False``：直接走老聚合器（`_compute_legacy`），
  输出仍是单层 dict（`capital_flow / orderbook / derivatives /
  market_structure`），让规则引擎与现有 LLM prompt 不需要改也能跑。

性能预算
========
- 多周期 5 个周期 × 3 类计算（capital / derivatives / market）
  + orderbook 1 次共享读取，整体单次 ``compute`` < 500ms。
- 数据全部来自 K 线表 + funding/OI 表（已经在采集层落库），
  本层不做任何 trades 重采样。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.factor_engine.capital_flow import (
    compute_capital_flow,
    compute_capital_flow_from_klines,
)
from app.factor_engine.derivatives import (
    compute_derivatives_factors,
    compute_derivatives_per_timeframe,
)
from app.factor_engine.market_structure import (
    compute_market_structure,
    compute_market_structure_from_klines,
)
from app.factor_engine.orderbook import compute_orderbook_factors
from app.logging_config import get_logger

logger = get_logger(__name__)


# 新版多周期矩阵覆盖的 5 个周期（按从快到慢）
MTF_TIMEFRAMES: List[str] = ["5m", "15m", "1h", "4h", "1d"]


class FactorAggregator:
    """
    因子聚合器
    -----------------------------------------------------------------
    职责：
        - 当 settings.enable_mtf_factors=True：按 MTF_TIMEFRAMES 拉取多周期
          K 线表 + funding/OI + 最新订单簿 + 最近 1h 爆仓，组合成多周期
          因子矩阵 + mtf_alignment 共振 + liquidations 滚动窗口。
        - 当 settings.enable_mtf_factors=False：回退到老聚合器路径
          （30 分钟单一窗口 + trades 重采样），保证灰度回滚一键生效。
    """

    def __init__(self, repos: Repositories, settings: Settings):
        """
        构造因子聚合器
        -------------------------------------------------------------
        参数：
            repos:    数据仓储集合
            settings: 全局配置
        """
        self.repos = repos
        self.settings = settings

    async def compute(self, symbol: str) -> Dict[str, Any]:
        """
        计算并返回当前因子快照
        -------------------------------------------------------------
        参数：
            symbol: 合约代码，例如 'ETH-USDT-SWAP'
        返回：
            多周期模式：见模块顶部的 by_timeframe / mtf_alignment / liquidations 结构
            老模式：单层 dict（capital_flow / orderbook / derivatives / market_structure）
        """
        if not self.settings.enable_mtf_factors:
            return await self._compute_legacy(symbol)
        return await self._compute_mtf(symbol)

    # ------------------------------------------------------------------
    # 老聚合（回滚通道）
    # ------------------------------------------------------------------
    async def _compute_legacy(self, symbol: str) -> Dict[str, Any]:
        """
        老版聚合：30 分钟窗口的单层因子 dict
        -------------------------------------------------------------
        参数：
            symbol: 合约代码
        返回：
            单层 dict，与 P0 升级前的 schema 完全一致。
        """
        window = self.settings.factor_window_seconds
        since = datetime.now(timezone.utc) - timedelta(seconds=window)

        trades = await self.repos.fetch_recent_trades(symbol, since=since, limit=20000)
        orderbook = await self.repos.fetch_latest_orderbook(symbol)
        funding = await self.repos.fetch_latest_funding(symbol)
        oi_history = await self.repos.fetch_recent_oi(symbol, since=since, limit=2000)

        capital = compute_capital_flow(trades)
        ob = compute_orderbook_factors(
            orderbook, wall_multiplier=self.settings.liquidity_wall_multiplier
        )
        deriv = compute_derivatives_factors(funding, oi_history)
        struct = compute_market_structure(trades)

        return {
            "symbol": symbol,
            "window_seconds": window,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "capital_flow": capital,
            "orderbook": ob,
            "derivatives": deriv,
            "market_structure": struct,
        }

    # ------------------------------------------------------------------
    # 多周期聚合（P0 主路径）
    # ------------------------------------------------------------------
    async def _compute_mtf(self, symbol: str) -> Dict[str, Any]:
        """
        多周期因子矩阵 + mtf_alignment + liquidations 滚动窗口
        -------------------------------------------------------------
        参数：
            symbol: 合约代码
        返回：
            完整的 P0 因子快照。
        """
        now = datetime.now(timezone.utc)

        # ---- 共享数据：orderbook / funding / OI / 爆仓 ----
        orderbook = await self.repos.fetch_latest_orderbook(symbol)
        funding = await self.repos.fetch_latest_funding(symbol)
        # OI 取近 24 小时，覆盖 1d 周期的起点
        oi_since = now - timedelta(seconds=24 * 3600)
        oi_history = await self.repos.fetch_recent_oi(
            symbol, since=oi_since, limit=20000
        )
        # 爆仓最多算 1h 滚动窗口，但保留余量便于 cascade_signal 比较
        liq_since = now - timedelta(seconds=2 * 3600)
        liquidations = await self.repos.fetch_liquidations_since(symbol, liq_since)

        ob_factors = compute_orderbook_factors(
            orderbook, wall_multiplier=self.settings.liquidity_wall_multiplier
        )

        lookback = int(self.settings.mtf_lookback_bars)

        by_timeframe: Dict[str, Dict[str, Any]] = {}
        for tf in MTF_TIMEFRAMES:
            klines = await self.repos.fetch_recent_klines(
                timeframe=tf, symbol=symbol, limit=lookback
            )
            cap = compute_capital_flow_from_klines(
                klines,
                volume_zscore_window=int(self.settings.mtf_volume_zscore_window),
                divergence_lookback=int(self.settings.mtf_divergence_lookback),
            )
            deriv = compute_derivatives_per_timeframe(
                funding=funding,
                oi_history=oi_history,
                klines=klines,
            )
            struct = compute_market_structure_from_klines(klines, levels_count=3)
            # orderbook：P0 阶段不分周期，所有周期挂同一份快照，
            # P1 再补按 N 秒平均的时序版。
            by_timeframe[tf] = {
                "capital_flow": cap,
                "orderbook": ob_factors,
                "derivatives": deriv,
                "market_structure": struct,
            }

        mtf_alignment = self._compute_alignment(by_timeframe)
        liquidation_summary = self._summarize_liquidations(
            liquidations=liquidations, now=now,
        )

        return {
            "symbol": symbol,
            "computed_at": now.isoformat(),
            "by_timeframe": by_timeframe,
            "mtf_alignment": mtf_alignment,
            "liquidations": liquidation_summary,
        }

    # ------------------------------------------------------------------
    # mtf_alignment 共振
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_alignment(
        by_timeframe: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        计算多周期共振指标
        -------------------------------------------------------------
        参数：
            by_timeframe: 多周期因子矩阵
        返回：
            trend_votes / alignment_score / dominant_bias
        说明：
            trend 标签到方向的映射：
                uptrend   → long
                downtrend → short
                range / neutral → neutral
            alignment_score = (long_votes - short_votes) / total_tf
            dominant_bias 取 votes 最多的方向；并列时偏 neutral。
        """
        votes: Dict[str, int] = {"long": 0, "short": 0, "neutral": 0}
        total = 0
        for tf in MTF_TIMEFRAMES:
            tf_block = by_timeframe.get(tf) or {}
            struct = (tf_block.get("market_structure") or {})
            trend = struct.get("trend") or "neutral"
            if trend == "uptrend":
                votes["long"] += 1
            elif trend == "downtrend":
                votes["short"] += 1
            else:
                votes["neutral"] += 1
            total += 1

        score = 0.0
        if total > 0:
            score = round((votes["long"] - votes["short"]) / total, 4)

        if votes["long"] > votes["short"] and votes["long"] >= votes["neutral"]:
            dominant = "long"
        elif votes["short"] > votes["long"] and votes["short"] >= votes["neutral"]:
            dominant = "short"
        else:
            dominant = "neutral"

        return {
            "trend_votes": votes,
            "alignment_score": score,
            "dominant_bias": dominant,
        }

    # ------------------------------------------------------------------
    # 爆仓滚动窗口
    # ------------------------------------------------------------------
    def _summarize_liquidations(
        self,
        liquidations: List[Dict[str, Any]],
        now: datetime,
    ) -> Dict[str, Any]:
        """
        汇总爆仓滚动窗口因子
        -------------------------------------------------------------
        参数：
            liquidations: fetch_liquidations_since 返回的事件列表（升序）
            now:          当前 UTC 时间
        返回：
            long_<tf> / short_<tf> / imbalance_<tf> + cascade_signal
        说明：
            - 数值单位采用基础币种（ETH）的 size 累计；如果想要 USD
              金额，可换成 notional。这里选 size 是为了与 K 线 buy/sell
              volume 单位对齐，便于人脑跨表对比。
            - cascade_signal：取最近 1 分钟的爆仓总 size，与"过去 1 小时
              内每分钟均值"比较，> N 倍即 True。
            - 性能：单次遍历事件，线性时间，事件数 1 小时上限 ~10 万即可。
        """
        windows_min = list(self.settings.liquidation_windows_minutes or [5, 15, 60])
        out: Dict[str, Any] = {}
        # 预先把每条事件的 (epoch_seconds, side, size) 转成简单元组，
        # 避免后面循环里反复访问 dict 字段
        events: List[tuple] = []
        for ev in liquidations:
            ts = ev.get("ts")
            if ts is None:
                continue
            try:
                size = float(ev.get("size") or 0.0)
            except (TypeError, ValueError):
                size = 0.0
            side = ev.get("side")
            events.append((ts, side, size))

        for w in windows_min:
            since = now - timedelta(minutes=w)
            long_size = 0.0
            short_size = 0.0
            for ts, side, size in events:
                if ts < since:
                    continue
                if side == "long":
                    long_size += size
                elif side == "short":
                    short_size += size
            total = long_size + short_size
            imbalance = (
                round((long_size - short_size) / total, 4) if total > 0 else 0.0
            )
            out[f"long_{w}m"] = round(long_size, 6)
            out[f"short_{w}m"] = round(short_size, 6)
            out[f"imbalance_{w}m"] = imbalance

        # cascade_signal：最近 1 分钟 vs 过去 1h 每分钟均值
        last_minute_since = now - timedelta(minutes=1)
        last_minute_size = sum(
            size for ts, _, size in events if ts >= last_minute_since
        )
        last_hour_since = now - timedelta(minutes=60)
        last_hour_size = sum(
            size for ts, _, size in events if ts >= last_hour_since
        )
        # 每分钟均值；不足 60 分钟样本时仍按 60 估算（保守）
        per_min_avg = last_hour_size / 60.0 if last_hour_size > 0 else 0.0
        threshold = per_min_avg * float(self.settings.liquidation_cascade_multiplier)
        cascade = bool(per_min_avg > 0 and last_minute_size > threshold)
        out["cascade_signal"] = cascade
        out["last_minute_size"] = round(last_minute_size, 6)
        out["last_hour_avg_per_min"] = round(per_min_avg, 6)
        return out
