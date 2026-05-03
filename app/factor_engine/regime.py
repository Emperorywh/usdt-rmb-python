"""市场状态（regime）检测器。

输入
====
多周期因子矩阵中的 1h / 4h / 15m / 5m 的 ``market_structure`` 块，
依赖以下字段（来源：market_structure.py）：

    1h.market_structure.adx_14
    4h.market_structure.trend
    4h.market_structure.bb_width
    15m.market_structure.bb_width
    5m.market_structure.swept_high_recent / swept_low_recent

输出
====
单字符串挂在因子矩阵根节点 ``regime``，取值之一：

    'trending_up'   - 强趋势上行（1h ADX >= 25 且 4h trend=uptrend 且 4h bb_width 上升）
    'trending_down' - 强趋势下行（同上但 4h trend=downtrend）
    'ranging'       - 区间震荡（1h ADX < 18 且 4h bb_width 较中位数明显收窄）
    'breakout'      - 向上突破（15m bb_width 在过去 1h 暴增 + 5m swept_high）
    'breakdown'     - 向下突破（同上但 5m swept_low）
    'transitional'  - 过渡 / 不明确

设计原则
========
- 规则全部用阈值 + 字段判断；不引入额外历史序列读取，
  数据完全来自已经聚合好的 by_timeframe 矩阵，单次计算 < 1ms。
- 任意输入字段缺失时不抛错，回退到 'transitional'，让 LLM 仍可往下走。
- 阈值通过 settings 注入，便于上线后调整（例如不同合约 ADX 灵敏度差异）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def detect_regime(
    by_timeframe: Dict[str, Dict[str, Any]],
    bb_width_history_4h: Optional[list[float]] = None,
    bb_width_history_15m: Optional[list[float]] = None,
    adx_trending_threshold: float = 25.0,
    adx_ranging_threshold: float = 18.0,
    bb_compress_ratio: float = 0.7,
    bb_expand_ratio: float = 1.5,
) -> str:
    """
    判定当前市场处于哪种 regime
    -----------------------------------------------------------------
    参数：
        by_timeframe:           多周期因子矩阵（来自 FactorAggregator）
        bb_width_history_4h:    过去 N 根 4h bb_width 序列（升序），可空
        bb_width_history_15m:   过去 N 根 15m bb_width 序列（升序），可空
        adx_trending_threshold: 1h ADX ≥ 此值视为强趋势（默认 25）
        adx_ranging_threshold:  1h ADX < 此值视为震荡（默认 18）
        bb_compress_ratio:      4h bb_width < 历史中位数 × 该比例视为收窄
        bb_expand_ratio:        15m bb_width 当前值 ≥ 1h 内中位数 × 该比例视为暴增
    返回：
        'trending_up' / 'trending_down' / 'ranging' / 'breakout' / 'breakdown' / 'transitional'
    说明：
        判定优先级（由高到低）：
            1) breakout / breakdown：突破信号最及时，必须先抓
            2) trending_up / trending_down：1h ADX 与 4h trend 共同支持
            3) ranging：1h ADX 弱 + 4h 波动收窄
            4) 其余 → transitional
        bb_width_history 为空时退化为"只看当前 bb_width"。
    """
    one_h = (by_timeframe or {}).get("1h") or {}
    four_h = (by_timeframe or {}).get("4h") or {}
    fifteen_m = (by_timeframe or {}).get("15m") or {}
    five_m = (by_timeframe or {}).get("5m") or {}

    ms_1h = one_h.get("market_structure") or {}
    ms_4h = four_h.get("market_structure") or {}
    ms_15m = fifteen_m.get("market_structure") or {}
    ms_5m = five_m.get("market_structure") or {}

    adx_1h = _safe_num(ms_1h.get("adx_14"))
    trend_4h = ms_4h.get("trend") or "neutral"
    bb_4h = _safe_num(ms_4h.get("bb_width"))
    bb_15m = _safe_num(ms_15m.get("bb_width"))
    swept_high_5m = bool(ms_5m.get("swept_high_recent"))
    swept_low_5m = bool(ms_5m.get("swept_low_recent"))

    # ---- 1) 优先判定 breakout / breakdown ----
    if bb_15m is not None and bb_width_history_15m:
        median_15m = _median(bb_width_history_15m)
        if median_15m and bb_15m >= median_15m * bb_expand_ratio:
            if swept_high_5m and not swept_low_5m:
                return "breakout"
            if swept_low_5m and not swept_high_5m:
                return "breakdown"

    # ---- 2) 强趋势 ----
    if adx_1h is not None and adx_1h >= adx_trending_threshold:
        bb_expanding = True
        if bb_4h is not None and bb_width_history_4h:
            # 用 4h 历史中位数与当前比对：当前 ≥ 中位数则视为"未收窄"
            median_4h = _median(bb_width_history_4h)
            bb_expanding = bool(median_4h and bb_4h >= median_4h)
        if trend_4h == "uptrend" and bb_expanding:
            return "trending_up"
        if trend_4h == "downtrend" and bb_expanding:
            return "trending_down"

    # ---- 3) 震荡 ----
    if adx_1h is not None and adx_1h < adx_ranging_threshold:
        if bb_4h is not None and bb_width_history_4h:
            median_4h = _median(bb_width_history_4h)
            if median_4h and bb_4h < median_4h * bb_compress_ratio:
                return "ranging"
        else:
            # 缺少历史时仅依赖 ADX：兜底成 ranging，
            # 只要 4h 不是明显单边都视为震荡的可能性偏高。
            if trend_4h in ("range", "neutral"):
                return "ranging"

    # ---- 4) 其他 ----
    return "transitional"


def _safe_num(v: Any) -> Optional[float]:
    """安全 float 转换：None / 异常 → None，便于上层做 is None 判断。"""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _median(values: list[float]) -> Optional[float]:
    """
    朴素中位数实现（不引入 statistics 单纯为了少一次 import）
    """
    arr = [float(v) for v in values if v is not None]
    if not arr:
        return None
    arr.sort()
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return arr[mid]
    return (arr[mid - 1] + arr[mid]) / 2.0
