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

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.factor_engine.capital_flow import compute_capital_flow_from_klines
from app.factor_engine.derivatives import (
    compute_derivatives_per_timeframe,
)
from app.factor_engine.liquidity import build_liquidity_map
from app.factor_engine.market_structure import compute_market_structure_from_klines
from app.factor_engine.orderbook import compute_orderbook_factors_timeseries
from app.factor_engine.regime import detect_regime
from app.logging_config import get_logger

logger = get_logger(__name__)


# 新版多周期矩阵覆盖的 5 个周期（按从快到慢）
MTF_TIMEFRAMES: List[str] = ["5m", "15m", "1h", "4h", "1d"]


def _ob_static_view(ob_factors: Dict[str, Any]) -> Dict[str, Any]:
    """
    给大周期挂"订单簿静态视图"：去掉时序字段，避免 prompt / JSON 体积膨胀
    -----------------------------------------------------------------
    参数：
        ob_factors: 完整的 P1 订单簿因子（含时序字段）
    返回：
        仅保留 P0 静态字段的 dict（available / imbalance / best_bid /
        best_ask / spread / bid_qty / ask_qty / bid_walls / ask_walls）。
    说明：
        订单簿是高频指标，挂在 1h/4h/1d 上没有时序解释力（这些周期
        bar 内可能已经发生几十次盘口结构变化）。所以大周期只挂静态
        切片，时序字段全留给 5m / 15m。
    """
    keep_keys = (
        "available", "imbalance", "best_bid", "best_ask", "spread",
        "bid_qty", "ask_qty", "bid_walls", "ask_walls",
    )
    return {k: ob_factors.get(k) for k in keep_keys}


class FactorAggregator:
    """
    因子聚合器（LLM-First 架构下永远走多周期路径）
    -----------------------------------------------------------------
    职责：
        按 MTF_TIMEFRAMES 拉取多周期 K 线表 + funding/OI + 最新订单簿
        + 最近 1h 爆仓，组合成多周期因子矩阵 + mtf_alignment 共振
        + liquidations 滚动窗口 + regime + 流动性地图。

    LLM-First 重构：
        删除 ``_compute_legacy`` 回滚通道与 ``enable_mtf_factors`` /
        ``enable_orderbook_timeseries`` /
        ``enable_regime`` 3 个灰度 flag。MTF / 订单簿时序 /
        regime 全部永远开启。
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
            模块顶部说明的 by_timeframe / mtf_alignment / liquidations /
            regime / liquidity 完整结构。
        """
        return await self._compute_mtf(symbol)

    # ------------------------------------------------------------------
    # 多周期聚合（唯一路径）
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
        # 连接池利用率监控：>80% 时打 warn，帮助提前发现池耗尽
        stats = self.repos.db.pool_stats()
        usage_pct = (stats["size"] - stats["idle"]) / stats["max"] if stats["max"] > 0 else 0
        if usage_pct > 0.8:
            logger.warning(
                "连接池利用率偏高 symbol=%s：%d/%d 在用（空闲 %d）",
                symbol,
                stats["size"] - stats["idle"],
                stats["max"],
                stats["idle"],
            )

        now = datetime.now(timezone.utc)

        # ---- 第 1 组：共享数据并行拉取（6 个查询 → 1 次 gather）----
        # 使用 return_exceptions=True 保证单个查询失败不阻塞其他查询。
        oi_since = now - timedelta(seconds=24 * 3600)
        liq_since = now - timedelta(seconds=2 * 3600)
        ob_metric_window = int(self.settings.factor.orderbook_metrics_baseline_seconds)
        funding_window = int(self.settings.factor.funding_pct_rank_window_seconds)

        (
            orderbook_raw,
            funding_raw,
            oi_history_raw,
            liquidations_raw,
            ob_metrics_raw,
            funding_history_raw,
        ) = await asyncio.gather(
            self.repos.fetch_latest_orderbook(symbol),
            self.repos.fetch_latest_funding(symbol),
            self.repos.fetch_recent_oi(symbol, since=oi_since, limit=20000),
            self.repos.fetch_liquidations_since(symbol, liq_since),
            self.repos.fetch_orderbook_metrics_since(
                symbol=symbol,
                since=now - timedelta(seconds=ob_metric_window),
            ),
            self.repos.fetch_funding_rates_since(
                symbol=symbol, since=now - timedelta(seconds=funding_window)
            ),
            return_exceptions=True,
        )

        # 核心数据：失败必须显式 raise，还原"串行版直接抛异常"的行为
        if isinstance(orderbook_raw, Exception):
            raise orderbook_raw
        if isinstance(funding_raw, Exception):
            raise funding_raw
        if isinstance(oi_history_raw, Exception):
            raise oi_history_raw
        if isinstance(liquidations_raw, Exception):
            raise liquidations_raw

        orderbook = orderbook_raw
        funding = funding_raw
        oi_history = oi_history_raw
        liquidations = liquidations_raw

        # 可降级数据：Exception → 空值，与当前逐个 try/except 行为一致
        recent_orderbook_metrics = (
            ob_metrics_raw if not isinstance(ob_metrics_raw, Exception) else []
        )
        if isinstance(ob_metrics_raw, Exception):
            logger.warning(
                "拉取 orderbook_metrics 失败，退化为单快照模式 symbol=%s",
                symbol,
                exc_info=ob_metrics_raw,
            )
        funding_history = (
            funding_history_raw if not isinstance(funding_history_raw, Exception) else []
        )
        if isinstance(funding_history_raw, Exception):
            logger.warning(
                "拉取 funding_rates 历史失败，funding 分位数将为 None symbol=%s",
                symbol,
                exc_info=funding_history_raw,
            )

        ob_factors = compute_orderbook_factors_timeseries(
            latest_snapshot=orderbook,
            recent_metrics=recent_orderbook_metrics,
            wall_multiplier=self.settings.factor.liquidity_wall_multiplier,
            now=now,
            window_seconds=int(self.settings.factor.orderbook_metrics_window_seconds),
            baseline_seconds=int(self.settings.factor.orderbook_metrics_baseline_seconds),
        )

        lookback = int(self.settings.factor.mtf_lookback_bars)

        # ---- 第 2 组：5 周期 K 线并行拉取 ----
        klines_map = await asyncio.gather(*[
            self.repos.fetch_recent_klines(timeframe=tf, symbol=symbol, limit=lookback)
            for tf in MTF_TIMEFRAMES
        ])

        by_timeframe: Dict[str, Dict[str, Any]] = {}
        for idx, tf in enumerate(MTF_TIMEFRAMES):
            klines = klines_map[idx]
            cap = compute_capital_flow_from_klines(
                klines,
                volume_zscore_window=int(self.settings.factor.mtf_volume_zscore_window),
                divergence_lookback=int(self.settings.factor.mtf_divergence_lookback),
            )
            deriv = compute_derivatives_per_timeframe(
                funding=funding,
                oi_history=oi_history,
                klines=klines,
                funding_history=funding_history,
            )
            struct = compute_market_structure_from_klines(klines, levels_count=3)
            # orderbook：P1 在 5m/15m 周期挂"时序版"指标，
            # 1h/4h/1d 周期仍挂同一份（订单簿是高频，不下放到大周期）。
            ob_for_tf = ob_factors if tf in ("5m", "15m") else _ob_static_view(ob_factors)
            by_timeframe[tf] = {
                "capital_flow": cap,
                "orderbook": ob_for_tf,
                "derivatives": deriv,
                "market_structure": struct,
            }

        mtf_alignment = self._compute_alignment(by_timeframe)
        liquidation_summary = self._summarize_liquidations(
            liquidations=liquidations, now=now,
        )

        # regime 检测（永远开启；失败时挂 None）
        regime: Optional[str] = None
        try:
            regime = detect_regime(
                by_timeframe=by_timeframe,
                bb_width_history_4h=self._collect_bb_width_history(by_timeframe, "4h"),
                bb_width_history_15m=self._collect_bb_width_history(by_timeframe, "15m"),
                adx_trending_threshold=float(self.settings.factor.regime_adx_trending_threshold),
                adx_ranging_threshold=float(self.settings.factor.regime_adx_ranging_threshold),
            )
        except Exception:
            logger.warning(
                "regime 检测失败，回退到 None symbol=%s", symbol, exc_info=True
            )
            regime = None

        # P1：流动性地图
        try:
            liquidity = build_liquidity_map(
                by_timeframe=by_timeframe,
                round_step_usd=float(self.settings.factor.liquidity_round_level_step_usd),
                max_levels_per_side=int(self.settings.factor.liquidity_max_levels_per_side),
            )
        except Exception:
            logger.warning(
                "流动性地图构造失败，回退到空 symbol=%s", symbol, exc_info=True
            )
            liquidity = {
                "liquidity_pool_above": [],
                "liquidity_pool_below": [],
                "nearest_above_pct": None,
                "nearest_below_pct": None,
                "current_price": None,
            }

        return {
            "symbol": symbol,
            "computed_at": now.isoformat(),
            "by_timeframe": by_timeframe,
            "mtf_alignment": mtf_alignment,
            "liquidations": liquidation_summary,
            # P1 根节点新增字段
            "regime": regime,
            "liquidity": liquidity,
        }

    @staticmethod
    def _collect_bb_width_history(
        by_timeframe: Dict[str, Dict[str, Any]],
        tf: str,
    ) -> List[float]:
        """
        从指定周期的 market_structure 块中收集"过去 N 根 bar 的 bb_width 序列"
        -----------------------------------------------------------------
        参数：
            by_timeframe: 多周期因子矩阵
            tf:           '4h' / '15m' 等
        返回：
            float 列表；当前 P0 阶段 market_structure 只输出"最新一根的
            bb_width 标量"，因此这里只返回 [当前值]，给 regime 一个保底。
            下阶段如果接入按周期的 bb_width 历史窗口，把序列填进来即可。
        """
        block = (by_timeframe or {}).get(tf) or {}
        ms = block.get("market_structure") or {}
        v = ms.get("bb_width")
        try:
            return [float(v)] if v is not None else []
        except (TypeError, ValueError):
            return []

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
        windows_min = list(self.settings.factor.liquidation_windows_minutes or [5, 15, 60])
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
        threshold = per_min_avg * float(self.settings.factor.liquidation_cascade_multiplier)
        cascade = bool(per_min_avg > 0 and last_minute_size > threshold)
        out["cascade_signal"] = cascade
        out["last_minute_size"] = round(last_minute_size, 6)
        out["last_hour_avg_per_min"] = round(per_min_avg, 6)
        return out
